import argparse
import multiprocessing
import os

from torch import float64
import wget
import numpy as np
import csv
import soundfile as sf

# prepare the data of the dcase19t1a dataset.
print('Now download and process dcase19t1a dataset, it will take a few moments...')

# download the dcase19t1a dataset
if os.path.exists('./dcase19/data/TAU-urban-acoustic-scenes-2019-development') == False:
    os.mkdir('./dcase19/data')
    sc_urls = [
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.1.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.2.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.3.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.4.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.5.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.6.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.7.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.8.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.9.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.10.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.11.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.12.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.13.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.14.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.15.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.16.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.17.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.18.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.19.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.20.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.21.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.doc.zip',
        'https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip'
        ]
    for sc_url in sc_urls:
        wget.download(sc_url, out='./dcase19/data/')
        sc_name = sc_url.split('/')[-1]
        os.system('unzip -n ./dcase19/data/' + sc_name + ' -d ./dcase19/data')
        os.remove('./dcase19/data/' + sc_name)
    print('Finish download.')

scene_label = {
    'airport':0,
    'bus':1,
    'metro':2,
    'metro_station':3,
    'park':4,
    'public_square':5,
    'shopping_mall':6,
    'street_pedestrian':7,
    'street_traffic':8,
    'tram':9
}

source_path = './dcase19/data/TAU-urban-acoustic-scenes-2019-development/'
# Replace with the output directory
out_path_left = "./dcase19/data/mp3_audio_left/"
out_path_right = "./dcase19/data/mp3_audio_right/"
out_path_mid = "./dcase19/data/mp3_audio_mid/"

if os.path.exists('./dcase19/data/tmp') == False:
    os.mkdir('./dcase19/data/tmp')

def process_folder(fol="fold1_train.csv"):
    print("now working on ", fol)
    with open(os.path.join(source_path, 'evaluation_setup', fol), 'r') as f:
        rows = csv.reader(f)
        data_list = []
        for row in rows:
            if '.wav' in row[0]:
                data_list.append(row[0])
        name = fol.split('.')[0]
        os.makedirs(out_path_left + name, exist_ok=True)
        os.makedirs(out_path_right + name, exist_ok=True)
        os.makedirs(out_path_mid + name, exist_ok=True)
        print(f"it has {len(data_list)}")
        all_files = []
        for i in data_list:
            tag = scene_label[i.split('-')[0].split('/')[-1]]
            all_files.append(str(tag) + '_' + i.split('wav')[0].split('audio/')[-1])
        cmds = [(i, os.path.join(source_path, file.split('wav')[0] + 'wav').split('wav')[0] + 'wav', out_path_left + name + "/" + all_files[i], out_path_right + name + "/" + all_files[i], out_path_mid + name + "/" + all_files[i]) for i, file in enumerate(data_list)]
        print(cmds[0])
        with multiprocessing.Pool(processes=20) as pool:
            pool.starmap(process_one, cmds)


def process_one(i, f1, f2, f3, f4):
    if i % 500 == 0:
        print(i)
    tmp_pth = './dcase19/data/tmp/' + f1.split('/')[-1]
    with sf.SoundFile(f1) as f:
        data = f.read(dtype=np.float32)
    sf.write(tmp_pth, data[:, 0], 48000, 'PCM_16')
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {tmp_pth} -codec:a mp3 -ar 32000 {f2}mp3")
    os.remove(tmp_pth)
    sf.write(tmp_pth, data[:, 1], 48000, 'PCM_16')
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {tmp_pth} -codec:a mp3 -ar 32000 {f3}mp3")
    os.remove(tmp_pth)
    sf.write(tmp_pth, (data[:, 0] + data[:, 1])/2, 48000, 'PCM_16')
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {tmp_pth} -codec:a mp3 -ar 32000 {f4}mp3")
    os.remove(tmp_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=False, default=None,
                        help='Path of folder containing the wave files, expected format to be as downloaded '
                             'using the PANNs script, containing balanced_train_segments, eval_segments, '
                             'unbalanced_train_segments folders.')
    parser.add_argument('--out', type=str, required=False, default=None,
                        help='Directory to save out the converted mp3s.')

    args = parser.parse_args()

    source_path = args.source or source_path
    out_path = args.out or out_path_left
    process_folder('fold1_train.csv')
    process_folder('fold1_evaluate.csv')
    process_folder('fold1_test.csv')
    os.system('stty sane')
