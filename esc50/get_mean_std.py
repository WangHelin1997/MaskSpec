import torch
import numpy as np
import h5py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from esc50.dataset import decode_mp3, pad_or_truncate
from models.models_mae import AugmentMelSTFT 

def get_mean_std(n_mel=128):
    print('Start...')
    hdf5_file1 = './esc50/data/esc50_mp3_fold1.hdf'
    hdf5_file2 = './esc50/data/esc50_mp3_fold2.hdf'
    hdf5_file3 = './esc50/data/esc50_mp3_fold3.hdf'
    hdf5_file4 = './esc50/data/esc50_mp3_fold4.hdf'
    hdf5_file5 = './esc50/data/esc50_mp3_fold5.hdf'
    dataset1 = h5py.File(hdf5_file1, 'r')
    dataset2 = h5py.File(hdf5_file2, 'r')
    dataset3 = h5py.File(hdf5_file3, 'r')
    dataset4 = h5py.File(hdf5_file4, 'r')
    dataset5 = h5py.File(hdf5_file5, 'r')
    length1 = len(dataset1['audio_name'])
    length2 = len(dataset2['audio_name'])
    length3 = len(dataset3['audio_name'])
    length4 = len(dataset4['audio_name'])
    length5 = len(dataset5['audio_name'])
    Mel = AugmentMelSTFT(
        n_mels=n_mel, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0,
        timem=0, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
        fmax_aug_range=2000)
    all_mean = 0.
    all_std = 0.
    count = 0
    frames = []
    for i in range(length1):
        count += 1
        if count % 200 == 0:
            print(count, length1+length2+length3+length4+length5)
        waveform = decode_mp3(dataset1['mp3'][i])
        waveform = torch.Tensor(pad_or_truncate(waveform, 160000))
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += mel.std()
        frames.append(mel[0])
    for i in range(length2):
        count += 1
        if count % 200 == 0:
            print(count, length1+length2+length3+length4+length5)
        waveform = decode_mp3(dataset2['mp3'][i])
        waveform = torch.Tensor(pad_or_truncate(waveform, 160000))
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += mel.std()
        frames.append(mel[0])
    for i in range(length3):
        count += 1
        if count % 200 == 0:
            print(count, length1+length2+length3+length4+length5)
        waveform = decode_mp3(dataset3['mp3'][i])
        waveform = torch.Tensor(pad_or_truncate(waveform, 160000))
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += mel.std()
        frames.append(mel[0])
    for i in range(length4):
        count += 1
        if count % 200 == 0:
            print(count, length1+length2+length3+length4+length5)
        waveform = decode_mp3(dataset4['mp3'][i])
        waveform = torch.Tensor(pad_or_truncate(waveform, 160000))
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += mel.std()
        frames.append(mel[0])
    for i in range(length5):
        count += 1
        if count % 200 == 0:
            print(count, length1+length2+length3+length4+length5)
        waveform = decode_mp3(dataset5['mp3'][i])
        waveform = torch.Tensor(pad_or_truncate(waveform, 160000))
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += mel.std()
        frames.append(mel[0])
    dataset1.close()
    dataset2.close()
    dataset3.close()
    dataset4.close()
    dataset5.close()
    all_mean = all_mean / (length1+length2+length3+length4+length5)
    all_std = all_std / (length1+length2+length3+length4+length5)
    print('all mean', all_mean)
    print('all std', all_std)
    frames = np.concatenate(frames, 1)
    frame_mean = np.mean(frames, 1)
    frame_std = np.std(frames, 1)
    print('frame mean', frame_mean)
    print('frame std', frame_std)
    print(frame_mean.shape, frames.shape)
    dict = {
        'all_mean':all_mean,'all_std':all_std, 
        'frame_mean':frame_mean, 'frame_std':frame_std}
    np.save('./esc50/mean_std_'+str(n_mel)+'.npy', dict)
    return {"done": True}

def test(n_mel=128):
    mean_std_file = np.load('./esc50/mean_std_'+str(n_mel)+'.npy',allow_pickle=True).item()
    frame_mean = torch.Tensor(mean_std_file['frame_mean']).cuda()
    frame_std = torch.Tensor(mean_std_file['frame_std']).cuda()
    all_mean = mean_std_file['all_mean']
    all_std = mean_std_file['all_std']
    print(all_mean, all_std)

if __name__ == '__main__':
    get_mean_std(n_mel=128)
    get_mean_std(n_mel=64)
    test(n_mel=128)
    test(n_mel=64)
