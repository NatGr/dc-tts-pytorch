"""the contents of this file were basically copied from https://github.com/Kyubyong/dc_tts/blob/master/utils.py"""
import numpy as np
import librosa
import copy
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

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (n_mels, T) and dtype of float32.
      mag: A 2d array of shape (n_mags, T) and dtype of float32.
    """
    # Loading sound file
    y, sr = librosa.load(fpath, sr=SAMPLING_RATE)

    # removes silence at beginning and end of audio
    y, _ = librosa.effects.trim(y, top_db=DB_TRIM_THRESHOLD)

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
      mag: A numpy array of (n_mags, T)

    Returns:
      wav: A 1-D numpy array.
    """
    # de-noramlize
    mag = (np.clip(mag, 0, 1) * MAX_DB) - MAX_DB + REF_DB

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**POWER)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -PREEMPHASIS], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    """Applies Griffin-Lim's raw."""
    x_best = copy.deepcopy(spectrogram)
    for i in range(N_ITER):
        x_t = invert_spectrogram(x_best)
        est = librosa.stft(x_t, N_FFT, HOP_LENGTH, win_length=WIN_LENGTH)
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    x_t = invert_spectrogram(x_best)
    y = np.real(x_t)

    return y


def invert_spectrogram(spectrogram):
    """Applies inverse fft.
    Args:
      spectrogram: [n_mags, t]
    """
    return librosa.istft(spectrogram, HOP_LENGTH, win_length=WIN_LENGTH, window="hann")


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
