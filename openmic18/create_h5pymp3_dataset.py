import h5py
import numpy as np
import os

# %%
base_dir = "./dcase18/data/"
mp3_path = "./dcase18/data/mp3_audio/fold1_train/"
save_file = 'dcase18_train'
classes_num = 10
# %%
print("now working on ", mp3_path)

all_count = 0
available_files = []
for root, dirs, files in os.walk(mp3_path, topdown=False):
    for name in files:
        if name[-3:] == 'mp3':
            available_files.append(name)
targets = np.zeros((len(available_files), classes_num), dtype=bool)
for i in range(targets.shape[0]):
    label = int(available_files[i].split('_')[0])
    targets[i, label] = 1
y = np.packbits(targets, axis=-1)
packed_len = y.shape[1]
print(available_files[0], "classes: ",packed_len, y.dtype)
available_size = len(available_files)
dt = h5py.vlen_dtype(np.dtype('uint8'))
with h5py.File(base_dir + save_file + "_mp3.hdf", 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
    waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
    target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
    for i, file in enumerate(available_files):
        if i % 100 == 0:
            print(f"{i}/{available_size}")
        f = file
        a = np.fromfile(mp3_path + f, dtype='uint8')
        audio_name[i] = f.encode()
        waveform[i] = a
        target[i] = y[i]

# %%
base_dir = "./dcase18/data/"
mp3_path = "./dcase18/data/mp3_audio/fold1_test/"
save_file = 'dcase18_testing'
classes_num = 10
# %%
print("now working on ", mp3_path)

all_count = 0
available_files = []
for root, dirs, files in os.walk(mp3_path, topdown=False):
    for name in files:
        if name[-3:] == 'mp3':
            available_files.append(name)
targets = np.zeros((len(available_files), classes_num), dtype=bool)
for i in range(targets.shape[0]):
    label = int(available_files[i].split('_')[0])
    targets[i, label] = 1
y = np.packbits(targets, axis=-1)
packed_len = y.shape[1]
print(available_files[0], "classes: ",packed_len, y.dtype)
available_size = len(available_files)
dt = h5py.vlen_dtype(np.dtype('uint8'))
with h5py.File(base_dir + save_file + "_mp3.hdf", 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
    waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
    target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
    for i, file in enumerate(available_files):
        if i % 100 == 0:
            print(f"{i}/{available_size}")
        f = file
        a = np.fromfile(mp3_path + f, dtype='uint8')
        audio_name[i] = f.encode()
        waveform[i] = a
        target[i] = y[i]

# %%
base_dir = "./dcase18/data/"
mp3_path = "./dcase18/data/mp3_audio/fold1_evaluate/"
save_file = 'dcase18_evaluation'
classes_num = 10
# %%
print("now working on ", mp3_path)

all_count = 0
available_files = []
for root, dirs, files in os.walk(mp3_path, topdown=False):
    for name in files:
        if name[-3:] == 'mp3':
            available_files.append(name)
targets = np.zeros((len(available_files), classes_num), dtype=bool)
for i in range(targets.shape[0]):
    label = int(available_files[i].split('_')[0])
    targets[i, label] = 1
y = np.packbits(targets, axis=-1)
packed_len = y.shape[1]
print(available_files[0], "classes: ",packed_len, y.dtype)
available_size = len(available_files)
dt = h5py.vlen_dtype(np.dtype('uint8'))
with h5py.File(base_dir + save_file + "_mp3.hdf", 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
    waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
    target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
    for i, file in enumerate(available_files):
        if i % 100 == 0:
            print(f"{i}/{available_size}")
        f = file
        a = np.fromfile(mp3_path + f, dtype='uint8')
        audio_name[i] = f.encode()
        waveform[i] = a
        target[i] = y[i]
print("Done!")
