import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelNorm(nn.Module):
    """Layer normalization, where we only normalize along the channel axis, see
    https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/layer_norm for details
    (we are equivalent to begin_norm_axis=-1 since tf uses NHWC and pytorch NCHW)
    """
    def __init__(self, num_channels, eps=1e-12, dim=1):
        """
        :param num_channels: the size of the channel dimention of the input tensors
        :param eps: epsilon value added to denominator during normalization for numerical stability
        :param dim: number of dimensions of the input tensors (not counting batch and channel axes)
        """
        super().__init__()
        self.shape = [1, num_channels] + [1] * dim
        self.gamma = nn.Parameter(torch.ones(*self.shape))
        self.beta = nn.Parameter(torch.zeros(*self.shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Conv1DNormAct(nn.Module):
    """class representing a 1D conv with dropout, layernorm and activation. The 1d conv applies padding so that the
    length of the time dimension stays the same"""
    def __init__(self, in_channels, out_channels, dropout_rate, kernel_size=1, dilation=1,
                 activation: Optional[nn.Module] = None, causal_conv=False):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        if causal_conv:  # putting all padding on the left is sufficient to get causal convs, otherwise we put
            # the same +-1 on both sides
            self.padding = (padding, 0)
        else:
            self.padding = (padding // 2 + padding % 2, padding // 2)

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.normalize = ChannelNorm(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        :param x: inputs (Batch, in_channels, time)
        :return: outputs (Batch, out_channels, time)"""
        x = self.conv1d(F.pad(x, self.padding))
        x = self.normalize(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.dropout(x)


class ConvTranspose1DNormAct(nn.Module):
    """class representing a 1D transposed conv with dropout, layernorm and activation. The 1d transposed conv applies
    padding and stride so that the length of the time dimension doubles. Kernel size is 3"""
    def __init__(self, in_channels, out_channels, dropout_rate, activation: Optional[nn.Module] = None):
        super().__init__()

        self.convtranspose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)
        self.normalize = ChannelNorm(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        :param x: inputs (Batch, in_channels, time)
        :return: outputs (Batch, out_channels, time)"""
        x = self.convtranspose1d(x)
        x = self.normalize(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.dropout(x)


class HighwayConv(nn.Module):
    """class representing a 1D conv followed by a highway layer"""
    def __init__(self, in_channels, out_channels, dropout_rate, kernel_size=1, dilation=1, causal_conv=False):
        super().__init__()
        self.out_channels = out_channels
        padding = dilation * (kernel_size - 1)
        if causal_conv:  # putting all padding on the left is sufficient to get causal convs, otherwise we put
            # the same +-1 on both sides
            self.padding = (padding, 0)
        else:
            self.padding = (padding // 2 + padding % 2, padding // 2)

        self.conv1d = nn.Conv1d(in_channels, 2*out_channels, kernel_size, dilation=dilation)
        # layer norm for X, H1 and H2
        self.normalize_1, self.normalize_2 = ChannelNorm(out_channels), ChannelNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        :param x: inputs (Batch, in_channels, time)
        :return: outputs (Batch, out_channels, time)"""
        out_conv = self.conv1d(F.pad(x, self.padding))
        h_1, h_2 = out_conv[:, :self.out_channels, :], out_conv[:, self.out_channels:, :]
        h_1 = torch.sigmoid(self.normalize_1(h_1))
        h_2 = self.normalize_2(h_2)
        return self.dropout(h_1 * h_2 + (1.0 - h_1) * x)
