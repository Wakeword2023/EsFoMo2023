#based on: https://github.com/tugstugi/pytorch-speech-commands

"""Transforms on raw wav samples."""

import random
import numpy as np
import librosa

import torch
#import torchaudio
from torch.utils.data import Dataset
#from torchaudio.transforms import *

def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob

class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path']
        if path:
            samples, sample_rate = librosa.load(path, sr=self.sample_rate)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class LoadAudioRT(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        samples = np.asarray(data,dtype=np.float32)
        return samples

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class FixAudioLengthRT(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data, sample_rate=16000):
        samples = data
        length = int(self.time * sample_rate)
        if length < len(samples):
            data = samples[:length]
        elif length > len(samples):
            data = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1), prob=0.5):
        self.amplitude_range = amplitude_range
        self.prob = prob

    def __call__(self, data):
        if not should_apply_transform(self.prob):
            return data

        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data

class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2, prob=0.5):
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, data):
        if not should_apply_transform(self.prob):
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0  / (1 + scale)
        data['samples'] = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0,len(samples)), samples).astype(np.float32)
        return data

class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2, prob=0.5):
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, data):
        if not should_apply_transform(self.prob):
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data['samples'] = librosa.effects.time_stretch(data['samples'], 1+scale)
        return data

class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2, prob=0.5):
        self.max_shift_seconds = max_shift_seconds
        self.prob = prob

    def __call__(self, data):
        if not should_apply_transform(self.prob):
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        max_shift = (sample_rate * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        samples = np.pad(samples, (a, b), "constant")
        data['samples'] = samples[:len(samples) - a] if a else samples[b:]
        return data

class AddBackgroundNoise(Dataset):
    """Adds a random background noise."""

    def __init__(self, bg_dataset, max_percentage=0.45, prob=0.5):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage
        self.prob = prob

    def __call__(self, data):
        if not should_apply_transform(self.prob):
            return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        percentage = random.uniform(0, self.max_percentage)
        data['samples'] = samples * (1 - percentage) + noise * percentage
        return data

class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        s = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=self.n_mels)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

class ToMelSpectrogramRT(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32, sample_rate=16000):
        self.n_mels = n_mels
        self.sample_rate=sample_rate

    def __call__(self, data):
        samples = data
        s = librosa.feature.melspectrogram(y=samples, sr=self.sample_rate, n_mels=self.n_mels)
        data = librosa.power_to_db(s, ref=np.max)
        return data

class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, np_name, tensor_name, normalize=None):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data

class ToTensorRT(object):
    """Converts into a tensor."""

    def __init__(self, normalize=None):
        self.normalize = normalize

    def __call__(self, data):
        tensor = torch.FloatTensor(data)
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data = tensor
        return data
