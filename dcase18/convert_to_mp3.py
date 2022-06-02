import argparse
import multiprocessing
import glob
import os
import wget
import zipfile
import numpy as np

# prepare the data of the dcase18t1a dataset.
print('Now download and process dcase18t1a dataset, it will take a few moments...')

# download the dcase18t1a dataset
if os.path.exists('./dcase18/data/TUT-urban-acoustic-scenes-2018-development') == False:
    os.mkdir('./dcase18/data')
    sc_urls = [
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.1.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.2.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.3.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.4.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.5.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.6.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.7.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.8.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.9.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.10.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.11.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.12.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.audio.13.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.doc.zip',
        'https://zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development.meta.zip'
        ]
    for sc_url in sc_urls:
        wget.download(sc_url, out='./dcase18/data/')
        sc_name = sc_url.split('/')[-1]
        os.system('unzip -n ./dcase18/data/' + sc_name + ' -d ./dcase18/data')
        os.remove('./dcase18/data/' + sc_name)
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

source_path = './dcase18/data/TUT-urban-acoustic-scenes-2018-development/'
# Replace with the output directory
out_path = "./dcase18/data/mp3_audio/"

def process_folder(fol="fold1_train.txt"):
    print("now working on ", fol)
    with open(os.path.join(source_path, 'evaluation_setup', fol), 'r') as f:
        data_list = f.readlines()
        name = fol.split('.')[0]
        os.makedirs(out_path + name, exist_ok=True)
        print(f"it has {len(data_list)}")
        all_files = []
        for i in data_list:
            tag = scene_label[i.split('-')[0].split('/')[-1]]
            all_files.append(str(tag) + '_' + i.split('wav')[0].split('audio/')[-1])
        cmds = [(i, os.path.join(source_path, file.split('wav')[0] + 'wav').split('wav')[0] + 'wav', out_path + name + "/" + all_files[i]) for i, file in enumerate(data_list)]
        print(cmds[0])
        with multiprocessing.Pool(processes=20) as pool:
            pool.starmap(process_one, cmds)


def process_one(i, f1, f2):
    if i % 500 == 0:
        print(i)
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {f1} -codec:a mp3 -ar 32000 {f2}mp3")


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
    out_path = args.out or out_path
    process_folder('fold1_train.txt')
    process_folder('fold1_evaluate.txt')
    process_folder('fold1_test.txt')
    os.system('stty sane')
