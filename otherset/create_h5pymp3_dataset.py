import h5py
import numpy as np
import os

# %%
base_dir = "/data/dean/whl/hdf5s_audioset/"
mp3_path = "/data/dean/whl/mp3_audioset_20w/"
save_file = 'audioset_20w'
# %%
print("now working on ", mp3_path)
all_count = 0
available_files = []
for root, dirs, files in os.walk(mp3_path, topdown=False):
    for name in files:
        if name[-3:] == 'mp3':
            available_files.append(name)
available_size = len(available_files)
dt = h5py.vlen_dtype(np.dtype('uint8'))
with h5py.File(base_dir + save_file + "_mp3.hdf", 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
    waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
    for i, file in enumerate(available_files):
        if i % 1000 == 0:
            print(f"{i}/{available_size}")
        f = file
        a = np.fromfile(mp3_path + f, dtype='uint8')
        audio_name[i] = f.encode()
        waveform[i] = a
print("Done!")
