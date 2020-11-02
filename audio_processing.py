import numpy as np
import torch
import torch.nn.functional as F
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio  # we are using sox as backend
torchaudio.set_audio_backend("sox_io")
from torchaudio.transforms import GriffinLim, Resample, MelScale
import sys
from scipy import signal


SAMPLING_RATE = 22050
N_FFT = 2048  # fft points (samples)
FRAME_SHIFT = 0.0125  # seconds
FRAME_LENGTH = 0.05  # seconds
HOP_LENGTH = int(SAMPLING_RATE * FRAME_SHIFT)  # samples. =276.
WIN_LENGTH = int(SAMPLING_RATE * FRAME_LENGTH)  # samples. =1102.
N_MELS = 80  # Number of Mel banks to generate
N_MAGS = 1 + N_FFT // 2
POWER = 1.5  # Exponent for amplifying the predicted magnitude
N_ITER = 50  # Number of inversion iterations
PREEMPHASIS = .97
MAX_DB = 100
REF_DB = 20
REDUCTION_FACTOR = 4
DB_TRIM_THRESHOLD = 70  # more than default to be sure no speech audio is cut


def get_spectrograms(fpath):
    """Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    This would be faster if we batched it.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d pytorch tensor of shape (n_mels, T) and dtype of float32.
      mag: A 2d pytorch tensor of shape (n_mags, T) and dtype of float32.
    """
    # Loading sound file
    y, sr = torchaudio.load_wav(fpath).squeeze()  # since there is one channel, we remove it
    if len(y.shape) != 1:  # handle the case where there are two channels
        if y.shape[0] == 2:
            y = y.mean(axis=0)
        elif y.shape[1] == 2:
            y = y.mean(axis=1)
        else:
            print(f"file {fpath} has wrong dimensions, exciting...")
            sys.exit(-1)
    if sr != SAMPLING_RATE:
        y = Resample(sr, SAMPLING_RATE)(y)

    # Preemphasis
    y = torch.cat((y[0], y[1:] - PREEMPHASIS * y[:-1]))

    # stft
    linear = torch.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, return_complex=True)

    # magnitude spectrogram
    mag = torch.abs(linear)  # (n_mags, T)

    # mel spectrogram
    mel = MelScale(n_mels=N_MELS, sample_rate=SAMPLING_RATE, n_stft=N_FFT)(mag)

    # to decibel
    mel = 20 * torch.log10(torch.maximum(mel, torch.tensor(1e-5)))
    mag = 20 * torch.log10(torch.maximum(mag, torch.tensor(1e-5)))

    # normalize
    mel = torch.clamp((mel - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)
    mag = torch.clamp((mag - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    return mel.type(torch.float32), mag.type(torch.float32)


def spectrogram2wav(mag):
    """ Generate wave file from linear magnitude spectrogram

    Args:
      mag: A tensor of (n_mags, T)

    Returns:
      wav: A 1-D numpy array.
    """
    # de-noramlize
    mag = (torch.clamp(mag, 0, 1) * MAX_DB) - MAX_DB + REF_DB

    # to amplitude
    mag = torch.pow(10.0, mag * 0.05)

    # griffin-lim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # deprecation warnings depending on GriffinLim implementation
        wav = GriffinLim(n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, power=1/POWER, normalized=True)(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -PREEMPHASIS], wav.numpy())  # for some reason, this is much faster than
    # torchaudio.functional.lfilter(wav, torch.tensor([1, -PREEMPHASIS], dtype=torch.float32),
    # torch.tensor([1, 1], dtype=torch.float32))

    return wav.astype(np.float32)


def load_spectrograms(fpath):
    """Read the wave file in `fpath`
    and extracts spectrograms, mel spectrograms time dimention is reduced"""
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[1]

    # Marginal padding for reduction shape sync.
    num_paddings = REDUCTION_FACTOR - (t % REDUCTION_FACTOR) if t % REDUCTION_FACTOR != 0 else 0
    mel = F.pad(mel, [0, num_paddings, 0, 0], mode="constant")
    mag = F.pad(mag, [0, num_paddings, 0, 0], mode="constant")

    # Reduction
    mel = mel[:, ::REDUCTION_FACTOR]
    return mel, mag
