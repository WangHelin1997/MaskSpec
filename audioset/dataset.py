import io
import os
import random

import av
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DistributedSampler, WeightedRandomSampler, RandomSampler
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


class AudioSetDataset(TorchDataset):
    def __init__(self, hdf5_file, sample_rate=32000, classes_num=527, clip_length=10, augment=False, in_mem=False, extra_augment=False):
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
        if 'target' in self.dataset_file.keys():
            target = self.dataset_file['target'][index]
            target = np.unpackbits(target, axis=-1,
                                count=self.classes_num).astype(np.float32)
        else:
            target = None
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



def preload_mp3(balanced_train_hdf5, unbalanced_train_hdf5, num_of_classes):
    for hdf5_file in [balanced_train_hdf5, unbalanced_train_hdf5]:
        print(f"\n \n will now preload {hdf5_file} \n\n ")
        with h5py.File(hdf5_file, 'r') as dataset_file:
            target = dataset_file['mp3'][:]
            print(len(target))
            print(f"\n \n done with  {hdf5_file} \n\n ")
    return target[1000]


def get_ft_cls_balanced_sample_weights(balanced_train_hdf5, unbalanced_train_hdf5, num_of_classes,
                                       sample_weight_offset=100, sample_weight_sum=True):
    """
    :return: float tenosr of shape len(full_training_set) representing the weights of each sample.
    """
    # the order of balanced_train_hdf5,unbalanced_train_hdf5 is important.
    # should match get_full_training_set
    all_y = []
    for hdf5_file in [balanced_train_hdf5, unbalanced_train_hdf5]:
        with h5py.File(hdf5_file, 'r') as dataset_file:
            target = dataset_file['target']
            target = np.unpackbits(target, axis=-1,
                                   count=num_of_classes)
            all_y.append(target)
    all_y = np.concatenate(all_y, axis=0)
    all_y = torch.as_tensor(all_y)
    per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class

    per_class = sample_weight_offset + per_class  # offset low freq classes
    if sample_weight_offset > 0:
        print(f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}")
    per_class_weights = 1000. / per_class
    all_weight = all_y * per_class_weights
    if sample_weight_sum:
        print("\nsample_weight_sum\n")
        all_weight = all_weight.sum(dim=1)
    else:
        all_weight, _ = all_weight.max(dim=1)
    return all_weight


def get_ft_weighted_sampler(balanced_train_hdf5, unbalanced_train_hdf5, num_of_classes,
                            epoch_len=100000, sampler_replace=False):
    samples_weights=get_ft_cls_balanced_sample_weights(balanced_train_hdf5, unbalanced_train_hdf5, num_of_classes)
    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    print("num_nodes= ", num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    return DistributedSamplerWrapper(sampler=WeightedRandomSampler(samples_weights,
                                                                   num_samples=epoch_len, replacement=sampler_replace),
                                     dataset=range(epoch_len),
                                     num_replicas=num_nodes,
                                     rank=rank,
                                     )

def get_random_sampler(dataset, epoch_len=100000, sampler_replace=True):
    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    print("num_nodes= ", num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    return DistributedSamplerWrapper(sampler=RandomSampler(data_source=dataset, num_samples=epoch_len, replacement=sampler_replace),
                                     dataset=range(epoch_len),
                                     num_replicas=num_nodes,
                                     rank=rank,
                                     )

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

def get_base_training_set(balanced_train_hdf5, sample_rate=32000, classes_num=527, clip_length=10, augment=False, in_mem=False, extra_augment=True, roll=True, wavmix=True):
    ds = AudioSetDataset(
        hdf5_file=balanced_train_hdf5, sample_rate=sample_rate, classes_num=classes_num, 
        clip_length=clip_length, 
        augment=augment, in_mem=in_mem, extra_augment=extra_augment)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds

def get_full_training_set(balanced_train_hdf5, unbalanced_train_hdf5, sample_rate=32000, classes_num=527, clip_length=10, augment=False, in_mem=False, extra_augment=True, roll=True, wavmix=True):
    sets = [
        AudioSetDataset(
        hdf5_file=balanced_train_hdf5, sample_rate=sample_rate, classes_num=classes_num, 
        clip_length=clip_length, 
        augment=augment, in_mem=in_mem, extra_augment=extra_augment
        ), 
        AudioSetDataset(
        hdf5_file=unbalanced_train_hdf5, sample_rate=sample_rate, classes_num=classes_num, 
        clip_length=clip_length, 
        augment=augment, in_mem=in_mem, extra_augment=extra_augment
        )]
    ds = ConcatDataset(sets)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_test_set(eval_hdf5, sample_rate=32000, classes_num=527, clip_length=10):
    ds = AudioSetDataset(
        hdf5_file=eval_hdf5, sample_rate=sample_rate, classes_num=classes_num, 
        clip_length=clip_length, 
        augment=False, in_mem=False, extra_augment=False)
    return ds

def get_other_sets(others_hdf5_path, use_audioset, balanced_train_hdf5, unbalanced_train_hdf5, sample_rate=32000, classes_num=527, clip_length=10, augment=False, in_mem=False, extra_augment=True, roll=True, wavmix=True):
    sets = []
    for root, dirs, files in os.walk(others_hdf5_path, topdown=False):
        for name in files:
            if name[-3:] == 'hdf':
                sets.append(AudioSetDataset(
                    hdf5_file=os.path.join(root, name), sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length, 
                    augment=augment, in_mem=in_mem, extra_augment=extra_augment))
    if use_audioset:
        sets.append(AudioSetDataset(
                    hdf5_file=balanced_train_hdf5, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length, 
                    augment=augment, in_mem=in_mem, extra_augment=extra_augment))
        sets.append(AudioSetDataset(
                    hdf5_file=unbalanced_train_hdf5, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length, 
                    augment=augment, in_mem=in_mem, extra_augment=extra_augment))
        sets.append(AudioSetDataset(
                    hdf5_file=eval_hdf5, sample_rate=sample_rate, classes_num=classes_num, clip_length=clip_length, 
                    augment=augment, in_mem=in_mem, extra_augment=extra_augment))

    ds = ConcatDataset(sets)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        #print(self.sampler)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


if __name__ == "__main__":

    name = 'audioset'  # dataset name
    roll = True  # apply roll augmentation
    wavmix = True # apply wave-level mixup
    base_dir = "/data/dean/whl/audioset_Kong/"  # base directory of the dataset, change it or make a link
    if LMODE:
        base_dir = "/system/user/publicdata/CP/audioset/audioset_hdf5s/"

    balanced_train_hdf5 = base_dir + "mp3/balanced_train_segments_mp3.hdf"
    eval_hdf5 = base_dir + "mp3/eval_segments_mp3.hdf"
    unbalanced_train_hdf5 = base_dir + "mp3/unbalanced_train_segments_mp3.hdf"

    if LMODE:
        balanced_train_hdf5 = balanced_train_hdf5.replace(base_dir, os.environ.get("TMPDIR", base_dir)+"/")
        unbalanced_train_hdf5 = unbalanced_train_hdf5.replace(base_dir, os.environ.get("TMPDIR", base_dir)+"/")
        eval_hdf5 = eval_hdf5.replace(base_dir, os.environ.get("TMPDIR", base_dir)+"/")
    
    num_of_classes = 527

    print("get_base_test_set", len(get_test_set(eval_hdf5)))
    print("get_full_training_set", len(get_full_training_set(balanced_train_hdf5, unbalanced_train_hdf5)))
