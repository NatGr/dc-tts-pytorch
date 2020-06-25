import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Conv1DNormAct, HighwayConv
from itertools import chain
from tqdm import tqdm


class Text2Mel(nn.Module):
    """module representing the full Text2Mel model"""
    def __init__(self, vocab_size, embed_size, num_hidden_units, n_mels, dropout_rate):
        super().__init__()
        self.attn_norm = embed_size ** .5
        self.textEnc = TextEnc(vocab_size, embed_size, num_hidden_units, dropout_rate)
        self.audioEnc = AudioEnc(n_mels, num_hidden_units, dropout_rate)
        self.audioDec = AudioDec(n_mels, num_hidden_units, dropout_rate)
        self.n_mels = n_mels

    def forward(self, in_text: torch.Tensor, input_mel_spec: torch.Tensor):
        """ Predicts the mel spectrograms at time t given the input text and the previous mel spectrograms
        for all times t in one pass
        :param in_text: the input text. (Batch, Num_chars)
        :param input_mel_spec: the mel spectrograms of the input. (Batch, n_mels, Time_dim_size)
        :return: Tuple: generated_mel_specs (Batch, n_mels, Time_dim_size), generated_mel_specs_logits
        (Batch, n_mels, Time_dim_size), attention (Batch, Num_chars, Time_dim_size)
        """
        input_mel_spec = torch.cat((torch.zeros_like(input_mel_spec[:, :, :1]), input_mel_spec[:, :, :-1]), dim=2)
        key, value = self.textEnc(in_text)
        query = self.audioEnc(input_mel_spec)
        final_enc, attention = self.__compute_attention(key, value, query)
        mel, mel_logits = self.audioDec(final_enc)
        return mel, mel_logits, attention

    def __compute_attention(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor,
                            enforce_incr_att_offset: int = 0):
        """computes the concatenation of the attended value and the query
        for the givens key, value and query tensors
        :param key: (Batch, num_hidden_units, Num_chars)
        :param value: (Batch, num_hidden_units, Num_chars)
        :param query: (Batch, num_hidden_units, Time_dim_size)
        :param enforce_incr_att_offset: set to > 1 to enforce the attention to be incremental until the given offset,
        see paper (page 3)
        :return: Tuple: concat_attended_query. (Batch, 2*num_hidden_units, Time_dim_size)
            attention. (Batch, Num_chars, Time_dim_size)"""
        attention = F.softmax(torch.bmm(key.transpose(1, 2), query) / self.attn_norm, dim=1)  # (Batch, Num_chars,
        # Time_dim_size)
        if enforce_incr_att_offset > 1:
            attention = attention.cpu()
            offset_last_char = attention.shape[1] - 1
            should_enforce_again = True
            while should_enforce_again:  # in general, this happens < 10 times
                attended_chars = attention[:, :, :enforce_incr_att_offset].argmax(dim=1)  # (Batch, Time_dim_size)
                delta_attended_chars = attended_chars[:, 1:] - attended_chars[:, :-1]
                should_replace = (delta_attended_chars < -1) | (delta_attended_chars > 3)
                should_replace_offsets = should_replace.nonzero()

                for offset in should_replace_offsets:  # under those conditions, replace the attention by a
                    # kroeneker_delta
                    n_tminus1_plus1 = min(attended_chars[offset[0], offset[1]] + 1, offset_last_char)
                    kroeneker_delta = torch.zeros(attention.shape[1], device=attention.device, dtype=attention.dtype)
                    kroeneker_delta[n_tminus1_plus1] = 1
                    attention[offset[0], :, offset[1] + 1] = kroeneker_delta
                should_enforce_again = len(should_replace_offsets) != 0
            attention = attention.to(key.device)

        seed = torch.bmm(value, attention)
        return torch.cat((seed, query), dim=1), attention

    def synthesize(self, in_text: torch.Tensor, synthesize_length: int):  # TODO: I think we could spare a lot of computations by reusing the results of the previous loop iteration
        """
        synthesizes mel spectrograms for a given text. The generation stops once the attention has been on padding data
        for at least 3 steps
        :param in_text: the input text. (Batch, Num_chars)
        :param synthesize_length: the time dimention length of the synthesized mel spectrograms
        :return: generated_mel_specs. (Batch, n_mels, Time_dim_size)
        """
        key, value = self.textEnc(in_text)
        full_mel_spec = torch.zeros(in_text.shape[0], self.n_mels, synthesize_length+1, device=in_text.device)
        # starts with 1 zero-padding on the left of the time-axis
        for i in tqdm(range(synthesize_length), desc="t2m synthetisation of batch"):
            current_mel_spec = full_mel_spec[:, :, :i+1]
            query = self.audioEnc(current_mel_spec)
            final_enc, attention = self.__compute_attention(key, value, query, enforce_incr_att_offset=i+1)
            out_mel_spec, _ = self.audioDec(final_enc)
            full_mel_spec[:, :, i+1] = out_mel_spec[:, :, i]
            del current_mel_spec, query, final_enc, out_mel_spec

        # removes the useless noise while looking at the padding
        input_is_pad = in_text == 0  # (Batch, Num_Chars)
        attention_over_pad = (attention * input_is_pad.unsqueeze(-1)).sum(axis=1)
        return full_mel_spec[:, :, 1:] * (attention_over_pad <= .1).unsqueeze(1)


class TextEnc(nn.Module):
    """module representing the full text encoder model"""
    def __init__(self, vocab_size, embed_size, num_hidden_units, dropout_rate):
        super().__init__()
        twice_num_hidden_units = 2 * num_hidden_units
        self.num_hidden_units = num_hidden_units
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.C_1 = Conv1DNormAct(embed_size, twice_num_hidden_units, dropout_rate, activation=nn.ReLU(inplace=True))
        self.C_2 = Conv1DNormAct(twice_num_hidden_units, twice_num_hidden_units, dropout_rate)
        self.HCs_1 = torch.nn.ModuleList([HighwayConv(twice_num_hidden_units, twice_num_hidden_units, dropout_rate,
                                                      kernel_size=3, dilation=3 ** i)
                                          for i in chain(range(4), range(4))])
        self.HCs_2 = torch.nn.ModuleList([HighwayConv(twice_num_hidden_units, twice_num_hidden_units, dropout_rate,
                                                      kernel_size=3, dilation=1) for _ in range(2)])
        self.HCs_3 = torch.nn.ModuleList([HighwayConv(twice_num_hidden_units, twice_num_hidden_units, dropout_rate,
                                                      kernel_size=1, dilation=1) for _ in range(2)])

    def forward(self, x: torch.Tensor):
        """
        :param x: Text inputs. (Batch, Num_chars)
        :returns: Keys (Batch, num_hidden_units, Num_chars) and Values (Batch, num_hidden_units, Num_chars) as a Pair"""
        x = self.embed(x).transpose(1, 2)  # (batch, num_chars, embed_size) -> (batch, embed_size, num_chars)
        x = self.C_2(self.C_1(x))
        for hcs in [self.HCs_1, self.HCs_2, self.HCs_3]:
            for hc in hcs:
                x = hc(x)
        return x[:, :self.num_hidden_units, :], x[:, self.num_hidden_units:, :]


class AudioEnc(nn.Module):
    """module representing the full audio encoder model"""
    def __init__(self, n_mels, num_hidden_units, dropout_rate):
        super().__init__()
        self.C_1 = Conv1DNormAct(n_mels, num_hidden_units, dropout_rate, activation=nn.ReLU(inplace=True),
                                 causal_conv=True)
        self.C_2 = Conv1DNormAct(num_hidden_units, num_hidden_units, dropout_rate, activation=nn.ReLU(inplace=True),
                                 causal_conv=True)
        self.C_3 = Conv1DNormAct(num_hidden_units, num_hidden_units, dropout_rate,
                                 causal_conv=True)
        self.HCs_1 = nn.ModuleList([HighwayConv(num_hidden_units, num_hidden_units, dropout_rate, kernel_size=3,
                                                dilation=3 ** i, causal_conv=True) for i in chain(range(4), range(4))])
        self.HCs_2 = nn.ModuleList([HighwayConv(num_hidden_units, num_hidden_units, dropout_rate, kernel_size=3,
                                                dilation=3, causal_conv=True) for _ in range(2)])

    def forward(self, x: torch.Tensor):
        """
        :param x: mel spectrogram. (Batch, n_mels, Time_dim_size)
        :return: queries. (Batch, num_hidden_units, Time_dim_size)
        """
        x = self.C_3(self.C_2(self.C_1(x)))
        for hcs in [self.HCs_1, self.HCs_2]:
            for hc in hcs:
                x = hc(x)
        return x


class AudioDec(nn.Module):
    """module representing the full audio decoder model"""
    def __init__(self, n_mels, num_hidden_units, dropout_rate):
        super().__init__()
        self.C_1 = Conv1DNormAct(2 * num_hidden_units, num_hidden_units, dropout_rate, causal_conv=True)
        self.HCs_1 = nn.ModuleList([HighwayConv(num_hidden_units, num_hidden_units, dropout_rate, kernel_size=3,
                                                dilation=3 ** i, causal_conv=True) for i in range(4)])
        self.HCs_2 = nn.ModuleList([HighwayConv(num_hidden_units, num_hidden_units, dropout_rate, kernel_size=3,
                                                dilation=1, causal_conv=True) for _ in range(2)])
        self.Cs_2 = nn.ModuleList([Conv1DNormAct(num_hidden_units, num_hidden_units, dropout_rate, kernel_size=1,
                                                 dilation=1, causal_conv=True, activation=nn.ReLU(inplace=True))
                                   for _ in range(3)])
        self.C_3 = Conv1DNormAct(num_hidden_units, n_mels, dropout_rate, causal_conv=True)

    def forward(self, x: torch.Tensor):
        """
        :param x: encoded audio. (Batch, 2*num_hidden_units, Time_dim_size)
        :return: generated_mel_specs, generated_mel_specs_logits. Both of shape (Batch, n_mels, Time_dim_size)
        """
        x = self.C_1(x)
        for hcs_or_cs in [self.HCs_1, self.HCs_2, self.Cs_2]:
            for hc_or_c in hcs_or_cs:
                x = hc_or_c(x)
        x = self.C_3(x)
        return torch.sigmoid(x), x
