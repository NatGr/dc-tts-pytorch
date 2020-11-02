import torch
from torch.utils.data import Dataset, Sampler
from audio_processing import load_spectrograms
import os
import pandas as pd
import re
import random


class TTSDataset(Dataset):
    """custom dataset to load mal and mel spectrograms as well as processed text for text 2 speech data"""
    def __init__(self, folder: str, is_text2mel: bool, vocab: str, max_num_chars: int, max_num_mag_time_frames: int):
        """
        :param folder: the folder where the data lays
        :param is_text2mel: if True we train text2mel and we need text + mels,
        otherwise we need mels + mags to train ssrn
        :param vocab: vocabulary used to load data
        :param max_num_chars: max num of chars allowed in a sentence datapoint
        :param max_num_mag_time_frames: max number of time frames for the mag transform
        """
        super().__init__()
        self.folder = folder
        self.is_text2mel = is_text2mel
        self.char2id = {char: id for id, char in enumerate(vocab)}
        transcript = os.path.join(folder, "transcript.csv")
        audio = os.path.join(folder, "audio")
        if not os.path.isfile(transcript):
            raise IOError(f"unable to find transcript file in folder {folder}")
        if not os.path.isdir(audio):
            raise IOError(f"unable to find audio folder in folder {folder}")
        self.data = pd.read_csv(transcript, sep=";")

        # create mels and mags if they do not exist
        self.mags = os.path.join(audio, "mags")
        self.mels = os.path.join(audio, "mels")
        if not os.path.exists(self.mags):
            if "mag_size" in self.data.columns:
                print("There is a column named mag_size in the csv but no mags folder, this is abnormal")
                self.data.drop(columns="mag_size", inplace=True)
            os.mkdir(self.mags)
            os.mkdir(self.mels)
            mag_sizes = []
            for file in self.data["file"]:
                mel, mag = load_spectrograms(os.path.join(audio, file))
                mag_sizes.append(mag.shape[1])
                torch.save(mel, os.path.join(self.mels, file[:-3] + "pt"))
                torch.save(mag, os.path.join(self.mags, file[:-3] + "pt"))
            self.data["mag_size"] = mag_sizes
            self.data.to_csv(transcript, index=False, sep=";")

        # treats text
        invalid_char = re.compile(f"[^{vocab}]")
        several_spaces = re.compile("[ ]+")
        if any(char.isupper() for char in vocab[2:]):  # we do not count padding and end of sentence
            prepro = lambda text: re.sub(several_spaces, " ", re.sub(invalid_char, " ", text)) + vocab[1]
        else:  # if there no upper case char in vocab, we lowercase the input
            prepro = lambda text: re.sub(several_spaces, " ", re.sub(invalid_char, " ", text.lower())) + vocab[1]
        self.data["sentence"] = self.data["sentence"].apply(prepro)

        # filter out long sequences
        num_data = self.data.shape[0]
        self.data = self.data.assign(sent_len=lambda x: x.sentence.apply(lambda y: len(y)))\
            .query("sent_len <= @max_num_chars")
        num_data_after_long_sent_out = len(self.data)
        print(f"{num_data - num_data_after_long_sent_out} ({(num_data - num_data_after_long_sent_out) / num_data * 100: .2f}%)"
              f" samples of the dataset were deleted because the sentences were too long")

        self.data = self.data.query("mag_size <= @max_num_mag_time_frames")
        print(f"{num_data_after_long_sent_out - len(self.data)} "
              f"({(num_data_after_long_sent_out - len(self.data)) / num_data * 100: .2f}%) samples of the dataset were "
              f"deleted because the sentences were too long")

    def __getitem__(self, index):
        """
        :param index: index of data to return
        :return: sentence. (Num_chars), mel (n_mels, T) OR mel (n_mels, T), mag (n_mags, T)
        """
        file = self.data.at[index, "file"][:-3] + "pt"
        mel = torch.load(os.path.join(self.mels, file))

        if self.is_text2mel:
            sentence = self.data.at[index, "sentence"]
            sentence = torch.tensor([self.char2id[char] for char in sentence], dtype=torch.long)
            return sentence, mel
        else:
            mag = torch.load(os.path.join(self.mags, file))
            return mel, mag

    def __len__(self):
        return len(self.data)


class BatchSampler(Sampler):
    """samples the data in batches where data whoose audio has the same size is grouped together, batch order and data
    order within batch are random"""
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        index_list = data_source.data["mag_size"].sample(frac=1).sort_values().index
        self.indexes = []
        batch = []
        for offset in index_list:
            if len(batch) < batch_size:
                batch.append(offset)
            else:
                self.indexes.append(batch)
                batch = [offset]
        self.indexes.append(batch)
        random.shuffle(self.indexes)

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


def collate_batch(batch, is_text2mel):
    """regroups the different tensors forming a batch together
    :param batch: List of tuples.
    :param is_text2mel: True if we train text2mel and not ssrn
    :return: Tuple.
    When is_text2mel, The batch concatenation of Text, the batch concatenation of Mel (shifted - to be feed to the net)
    and the batch concatenation of Mels not shifted (target)
    When not is_text2mel, The batch concatenation of Mel (not shifted), the batch concatenation of Mag (not shifted)
    In both cases, some zero padding will be applied among the last dimention to concatenate the elements together.
    """
    nbr_elem = len(batch[0])
    max_sizes = [0] * nbr_elem
    return_val = []

    # get max length
    for data in batch:
        for i in range(nbr_elem):
            max_sizes[i] = max(max_sizes[i], data[i].shape[-1])

    # padding and grouping
    device = batch[0][0].device

    for i, max_size in enumerate(max_sizes):
        padded_tensor_shape = [len(batch)] + [dim for dim in batch[0][i].shape]
        padded_tensor_shape[-1] = max_size
        dtype = batch[0][i].dtype
        return_val.append(torch.zeros(*padded_tensor_shape, dtype=dtype, device=device))

    for i, data in enumerate(batch):
        for j in range(nbr_elem):
            if len(return_val[j].shape) == 2:
                return_val[j][i, :data[j].shape[0]] = data[j]
            elif len(return_val[j].shape) == 3:
                return_val[j][i, :, :data[j].shape[1]] = data[j]

    if is_text2mel:
        text, mels = return_val[0], return_val[1]
        shifted_mels = torch.zeros_like(mels)
        shifted_mels[:, :, 1:] = mels[:, :, :-1]
        return text, shifted_mels, mels
    else:
        return return_val


def load_sentences(sentences_file, vocab, max_num_chars):
    """loads and treats the sentences from the sentence file
    returns: errors_list, sentences, sentences_offsets, sentences_tensor
    If errors_list is empty: returns [], the list of sentences, the list of lists of offset(s) of each sentence
    within the tensor and the sentence tensor. If one sentence was splitted, it has several sentence_offsets.
    If error_treating_sentences is True: the three others are None and we should stop the program
    """
    def add_new_tensor(current_tensor_offset, sentences_tensors, sentence, char2id):
        """add a new sentence to the list of curated sentences tensors"""
        current_tensor_offset += 1
        sentences_tensors.append(torch.tensor([char2id[char] for char in sentence], dtype=torch.long))
        return current_tensor_offset, sentences_tensors

    char2id = {char: id for id, char in enumerate(vocab)}
    sentences_offsets = []
    sentences_tensors = []
    errors = []
    current_tensor_offset = 0

    with open(sentences_file, "r") as file:
        sentences = file.readlines()

    invalid_char = re.compile(f"[^{vocab}]")
    several_spaces = re.compile("[ ]+")
    if any(char.isupper() for char in vocab[2:]):  # we do not count padding and end of sentence
        sentences = list(map(
            lambda text: re.sub(several_spaces, " ", re.sub(invalid_char, " ", text)) + vocab[1], sentences))
    else:  # if there no upper case char in vocab, we lowercase the input
        sentences = list(map(
            lambda text: re.sub(several_spaces, " ", re.sub(invalid_char, " ", text.lower())) + vocab[1], sentences))

    # cut sentences that are too long in smaller pieces
    punctuation_regex = re.compile("[.?!,]")
    for sentence in sentences:
        if len(sentence) <= max_num_chars:
            sentences_offsets.append([current_tensor_offset])
            current_tensor_offset, sentences_tensors = add_new_tensor(current_tensor_offset, sentences_tensors,
                                                                      sentence, char2id)
        else:  # we will have to split the sentence
            splitted_sentence = re.split(punctuation_regex, sentence)
            new_error = False
            for sub_sent in splitted_sentence:
                if len(sub_sent) > max_num_chars:
                    new_error = True

            if new_error:
                errors.append(sentence)
            else:  # splits the sentence
                new_sentence, len_new_sentence, sentence_tensors_offsets = "", 0, []
                for sub_sent in splitted_sentence:
                    if len_new_sentence == 0:
                        new_sentence = sub_sent
                        len_new_sentence += len(sub_sent)
                    elif len_new_sentence + 1 + len(sub_sent) <= max_num_chars:
                        new_sentence += ',' + sub_sent  # we always use comma here
                        len_new_sentence += 1 + len(sub_sent)
                    else:
                        sentence_tensors_offsets.append(current_tensor_offset)
                        current_tensor_offset, sentences_tensors = add_new_tensor(
                            current_tensor_offset, sentences_tensors, new_sentence, char2id)
                        new_sentence, len_new_sentence = "", 0
                if len_new_sentence != 0:
                    sentence_tensors_offsets.append(current_tensor_offset)
                    current_tensor_offset, sentences_tensors = add_new_tensor(
                        current_tensor_offset, sentences_tensors, new_sentence, char2id)
                sentences_offsets.append(sentence_tensors_offsets)

    if len(errors) > 0:
        return errors, None, None, None
    else:
        # merge all tensors
        complete_tensor = torch.zeros(len(sentences_tensors), max_num_chars, dtype=torch.long)
        for i, tensor in enumerate(sentences_tensors):
            complete_tensor[i, :tensor.shape[0]] = tensor
        return errors, sentences, sentences_offsets, complete_tensor
