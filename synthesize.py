import torch
from argparse import ArgumentParser
from Text2Mel import Text2Mel
from SSRN import SSRN
from tqdm import tqdm
from audio_processing import N_MELS, N_MAGS, REDUCTION_FACTOR, SAMPLING_RATE, spectrogram2wav
from data_loading import load_sentences
import os
from scipy.io import wavfile


if __name__ == "__main__":
    parser = ArgumentParser(description="synthesize audio in a folder named synthesized_audio that will lay next to "
                                        "sentences_file")
    parser.add_argument('--t2m_ckpt', type=str, required=True, help="if a file, pytorch checkpoint of text to mel, if a"
                                                                    " folder tensorflow checkpoint for text to mel")
    parser.add_argument('--ssrn_ckpt', type=str, required=True, help="same as t2m_ckpt but for SSRN")
    parser.add_argument('--sentences_file', type=str, required=True,
                        help="file containing the sentences to speek separated by a line return")
    parser.add_argument('--max_num_chars', type=int, default=250,
                        help="sentences with more chars than that will be splitted")
    parser.add_argument('--max_num_mag_time_frames', type=int, default=800,
                        help="we will not generate mags with more samples than that")
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--num_t2m_hidden_units', type=int, default=256)
    parser.add_argument('--num_ssrn_hidden_units', type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=.05)
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help="max batch size to use, the bigger, the faster the inference will be for a lot of data at "
                             "the same time. If it's too big, you will have VRAM/RAM overflows")
    parser.add_argument('--vocab', type=str,
                        help="authorized text token, the first must always stand for padding and the second for end of "
                             "sentence, if no uppercase letter is present in vocab, the input text will be lowercased",
                        default="PE abcdefghijklmnopqrstuvwxyz'.?")
    args = parser.parse_args()
    print("\n")
    print(args)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_folder = os.path.join(os.path.dirname(args.sentences_file), "synthesized_audio")
        os.makedirs(audio_folder, exist_ok=True)

        t2m = Text2Mel(len(args.vocab), args.embed_size, args.num_t2m_hidden_units, N_MELS, args.dropout_rate)
        ssrn = SSRN(N_MELS, N_MAGS, args.num_ssrn_hidden_units, args.dropout_rate)

        if os.path.isfile(args.t2m_ckpt):
            checkpoint = torch.load(args.t2m_ckpt)
            t2m.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            from load_tf_models import load_t2m_from_tf
            load_t2m_from_tf(t2m, args.t2m_ckpt)
        t2m.eval()
        if os.path.isfile(args.ssrn_ckpt):
            checkpoint = torch.load(args.ssrn_ckpt)
            ssrn.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            from load_tf_models import load_ssrn_from_tf
            load_ssrn_from_tf(ssrn, args.ssrn_ckpt)
        ssrn.eval()

        print("Models loaded")

        errors, sentences, sentences_offsets, sentences_tensor = load_sentences(
            args.sentences_file, args.vocab, args.max_num_chars)
        if len(errors) > 0:
            print("The following sentences are too long and cannot be automatically splitted by punctuation, "
                  "please split them manually and retry:\n")
            print(errors)
            exit()

        batches = torch.split(sentences_tensor, args.max_batch_size, dim=0)
        print("Data loaded")

        # t2m predictions
        t2m = t2m.to(device)
        mel_specs = []
        num_mel_time_frames = args.max_num_mag_time_frames // REDUCTION_FACTOR
        for batch in tqdm(batches, desc="t2m batches"):
            batch = batch.to(device)
            mel_specs.append(t2m.synthesize(batch, num_mel_time_frames).to("cpu"))
        del t2m
        del batch

        # ssrn predictions
        ssrn = ssrn.to(device)
        mags = []
        for mel in tqdm(mel_specs, desc="ssrn batches"):
            mag, _ = ssrn.forward(mel.to(device))
            mags.append(mag.to("cpu"))

        # merge back tensors and saves wav files
        for i, (sentence, sentence_offsets) in tqdm(enumerate(zip(sentences, sentences_offsets)),
                                                    desc="postprocessing", total=len(sentences)):
            wav = spectrogram2wav(torch.cat([mags[i // args.max_batch_size][i % args.max_batch_size, :].squeeze(0)
                                  for i in sentence_offsets], 1).numpy())
            file_name = os.path.join(audio_folder, f"{sentence[:20]}_{i}.wav")
            wavfile.write(file_name, SAMPLING_RATE, wav)
