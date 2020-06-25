import torch
import torch.nn as nn
from modules import Conv1DNormAct, HighwayConv, ConvTranspose1DNormAct
from itertools import chain


class SSRN(nn.Module):
    """module representing the SSRN model"""
    def __init__(self, n_mels, n_mags, num_ssrn_hidden_units, dropout_rate):
        super().__init__()
        twice_num_ssrn_hidden_units = 2 * num_ssrn_hidden_units
        self.C_1 = Conv1DNormAct(n_mels, num_ssrn_hidden_units, dropout_rate)
        self.HCs_1 = nn.ModuleList([HighwayConv(num_ssrn_hidden_units, num_ssrn_hidden_units, dropout_rate,
                                                kernel_size=3, dilation=3 ** i) for i in range(2)])
        self.D_HCs_1 = nn.ModuleList([
            ConvTranspose1DNormAct(num_ssrn_hidden_units, num_ssrn_hidden_units, dropout_rate) if i == 0 else
            HighwayConv(num_ssrn_hidden_units, num_ssrn_hidden_units, dropout_rate, kernel_size=3, dilation=3 ** (i-1))
            for i in chain(range(3), range(3))
        ])
        self.C_2 = Conv1DNormAct(num_ssrn_hidden_units, twice_num_ssrn_hidden_units, dropout_rate)
        self.HCs_2 = nn.ModuleList([HighwayConv(twice_num_ssrn_hidden_units, twice_num_ssrn_hidden_units, dropout_rate,
                                                kernel_size=3, dilation=1) for _ in range(2)])
        self.C_3 = Conv1DNormAct(twice_num_ssrn_hidden_units, n_mags, dropout_rate)
        self.Cs_4 = nn.ModuleList([Conv1DNormAct(n_mags, n_mags, dropout_rate, activation=nn.ReLU(inplace=True))
                                   for _ in range(2)])
        self.C_5 = Conv1DNormAct(n_mags, n_mags, dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        :param x: generated_mel_specs. (Batch, n_mels, Time_dim_size)
        :return: a pair: spectrograms, spectrograms logits. Both of shape (Batch, n_mags, Time_dim_size * 4)
        """
        x = self.C_1(x)
        for d_or_hcs in [self.HCs_1, self.D_HCs_1]:
            for d_or_hc in d_or_hcs:
                x = d_or_hc(x)
        x = self.C_2(x)
        for hc in self.HCs_2:
            x = hc(x)
        x = self.C_3(x)
        for c in self.Cs_4:
            x = c(x)
        x = self.C_5(x)
        return torch.sigmoid(x), x
