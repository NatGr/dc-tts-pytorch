import numpy as np
import torch
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio  # we are using sox as backend
torchaudio.set_audio_backend("sox_io")
from torchaudio.transforms import GriffinLim, Resample, MelScale
from scipy import signal
import librosa

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
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (n_mels, T) and dtype of float32.
      mag: A 2d array of shape (n_mags, T) and dtype of float32.
    """
    # Loading sound file
    y, sr = librosa.load(fpath, sr=SAMPLING_RATE)

    # removes silence at beginning and end of audio
    y, _ = librosa.effects.trim(y, top_db=DB_TRIM_THRESHOLD, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)

    # Preemphasis
    y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

    # stft
    linear = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    # magnitude spectrogram
    mag = np.abs(linear)  # (n_mags, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(SAMPLING_RATE, N_FFT, N_MELS)  # (n_mels, n_mags)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)
    mag = np.clip((mag - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    return mel.astype(np.float32), mag.astype(np.float32)


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

    # trim
    wav, _ = librosa.effects.trim(wav, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)

    return wav.astype(np.float32)


def load_spectrograms(fpath):
    """Read the wave file in `fpath`
    and extracts spectrograms, mel spectrograms time dimention is reduced"""
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[1]

    # Marginal padding for reduction shape sync.
    num_paddings = REDUCTION_FACTOR - (t % REDUCTION_FACTOR) if t % REDUCTION_FACTOR != 0 else 0
    mel = np.pad(mel, [[0, 0], [0, num_paddings]], mode="constant")
    mag = np.pad(mag, [[0, 0], [0, num_paddings]], mode="constant")

    # Reduction
    mel = mel[:, ::REDUCTION_FACTOR]
    return mel, mag
