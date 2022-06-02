import io
import os
import random

import av
from torch.utils.data import Dataset as TorchDataset
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from audioset.audiodatasets import PreprocessDataset
import h5py
import augly.audio as audaugs

LMODE = os.environ.get("LMODE", False)
if LMODE:
    def LMODE_default_config():
        cache_root_path = "/system/user/publicdata/CP/DCASE/cached_datasets/"


def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    # print(stream)
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]

def pydub_augment(waveform, gain_augment=7):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1-x1.mean()
            x2 = x2-x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class DCASE18Dataset(TorchDataset):
    def __init__(self, hdf5_file, sample_rate=32000, classes_num=10, clip_length=10, augment=False, in_mem=False, extra_augment=False):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.sample_rate = sample_rate
        self.hdf5_file = hdf5_file
        if in_mem:
            print("\nPreloading in memory\n")
            with open(hdf5_file, 'rb') as f:
                self.hdf5_file = io.BytesIO(f.read())
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['audio_name'])
            print(f"Dataset from {hdf5_file} with length {self.length}.")
        self.dataset_file = None  # lazy init
        self.clip_length = clip_length * sample_rate
        self.classes_num = classes_num
        self.augment = augment
        self.extra_augment = extra_augment
        if augment:
            print(f"Will agument data from {hdf5_file}")

    def open_hdf5(self):
        self.dataset_file = h5py.File(self.hdf5_file, 'r')

    def __len__(self):
        return self.length

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()
            self.dataset_file = None

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        if self.dataset_file is None:
            self.open_hdf5()

        audio_name = self.dataset_file['audio_name'][index].decode()
        try:
            waveform = decode_mp3(self.dataset_file['mp3'][index])
        except:
            print("Read Error:" + audio_name)
            index = random.randint(1,self.length-1)
            audio_name = self.dataset_file['audio_name'][index].decode()
            waveform = decode_mp3(self.dataset_file['mp3'][index])
        #else:
        #    waveform = decode_mp3(self.dataset_file['mp3'][index])
        #waveform = decode_mp3(self.dataset_file['mp3'][index])
        if self.augment:
            waveform = pydub_augment(waveform)
        
        waveform = self.resample(waveform)
        if self.extra_augment:
            Transforms = audaugs.Compose([
                audaugs.AddBackgroundNoise(snr_level_db=random.uniform(0.0, 15.0), p=random.random()),
                audaugs.ChangeVolume(volume_db=random.uniform(-2.0, 2.0), p=random.random()),
                audaugs.HighPassFilter(cutoff_hz=random.sample([5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0], 1)[0], p=random.random()),
                audaugs.LowPassFilter(cutoff_hz=random.sample([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], 1)[0], p=random.random()),
                audaugs.Speed(factor=random.uniform(0.8, 1.2), p=random.random()),
            ])
            waveform, _ = Transforms(waveform, self.sample_rate)
            if waveform.ndim > 1:
                waveform = waveform[0, :]
        waveform = pad_or_truncate(waveform, self.clip_length)
        target = self.dataset_file['target'][index]
        target = np.unpackbits(target, axis=-1,
                               count=self.classes_num).astype(np.float32)
        return waveform.reshape(1, -1), target

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0:: 2]
        elif self.sample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')

def get_roll_func(axis=1, shift=None, shift_range=50):
    print("rolling...")

    def roll_func(b):
        x, y = b
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-shift_range, shift_range))
        global FirstTime

        return x.roll(sf, axis), y

    return roll_func

def get_training_set(train_hdf5, sample_rate=32000, classes_num=10, clip_length=10, augment=False, in_mem=False, extra_augment=True, roll=True, wavmix=True):
    ds = DCASE18Dataset(
        hdf5_file=train_hdf5, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length, 
        augment=augment, in_mem=in_mem, extra_augment=extra_augment)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_test_set(eval_hdf5, sample_rate=32000, classes_num=10, clip_length=10):
    ds = DCASE18Dataset(
        hdf5_file=eval_hdf5, sample_rate=sample_rate, classes_num=classes_num, 
        clip_length=clip_length, 
        augment=False, in_mem=False, extra_augment=False)
    return ds

def get_validation_set(validation_hdf5, sample_rate=32000, classes_num=10, clip_length=10):
    ds = DCASE18Dataset(
        hdf5_file=validation_hdf5, sample_rate=sample_rate, classes_num=classes_num, 
        clip_length=clip_length, 
        augment=False, in_mem=False, extra_augment=False)
    return ds


if __name__ == "__main__":
    validation_hdf5 = './dcase18/data/dcase18_evaluation_mp3.hdf'
    eval_hdf5 = './dcase18/data/dcase18_testing_mp3.hdf'
    train_hdf5 = './dcase18/data/dcase18_train_mp3.hdf'

    print("get_test_set", len(get_test_set(eval_hdf5)))
    print("get_train_set", len(get_training_set(train_hdf5)))
    print("get_validation_set", len(get_validation_set(validation_hdf5)))
