import argparse
import multiprocessing
import glob
import os
import wget
import zipfile
import numpy as np

# prepare the data of the speechcommands dataset.
print('Now download and process speechcommands dataset, it will take a few moments...')

# download the speechcommands dataset
if os.path.exists('./scv2/data/speech_commands_v0.02') == False:
    # we use the 35 class v2 dataset, which is used in torchaudio https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html
    os.mkdir('./scv2/data')
    os.mkdir('./scv2/data/speech_commands_v0.02')
    sc_url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    wget.download(sc_url, out='./scv2/data/')
    os.system('tar -xzvf ./scv2/data/speech_commands_v0.02.tar.gz -C ./scv2/data/speech_commands_v0.02')
    os.remove('./scv2/data/speech_commands_v0.02.tar.gz')
    print('Finish download.')

# generate training list = all samples - validation_list - testing_list
if os.path.exists('./scv2/data/speech_commands_v0.02/train_list.txt')==False:
    with open('./scv2/data/speech_commands_v0.02/validation_list.txt', 'r') as f:
        val_list = f.readlines()

    with open('./scv2/data/speech_commands_v0.02/testing_list.txt', 'r') as f:
        test_list = f.readlines()

    val_test_list = list(set(test_list+val_list))

    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

    base_path = './scv2/data/speech_commands_v0.02/'
    all_cmds = get_immediate_subdirectories(base_path)
    all_list = []
    for cmd in all_cmds:
        if cmd != '_background_noise_':
            cmd_samples = get_immediate_files(base_path+'/'+cmd)
            for sample in cmd_samples:
                all_list.append(cmd + '/' + sample+'\n')

    training_list = [x for x in all_list if x not in val_test_list]

    with open('./scv2/data/speech_commands_v0.02/train_list.txt', 'w') as f:
        f.writelines(training_list)
    print('Finish train list generation.')

label_dicts = {}
count = 0
for root, dirs, files in os.walk('./scv2/data/speech_commands_v0.02/'):
    for dir in dirs:
        if dir != '_background_noise_':
            label_dicts[dir] = count
            count += 1
print(label_dicts)
print(len(label_dicts))

source_path = './scv2/data/speech_commands_v0.02/'
# Replace with the output directory
out_path = "./scv2/data/mp3_audio/"

def process_folder(fol="testing_list.txt"):
    print("now working on ", fol)
    with open(os.path.join(source_path, fol), 'r') as f:
        data_list = f.readlines()
        name = fol.split('_')[0]
        os.makedirs(out_path + name, exist_ok=True)
        print(f"it has {len(data_list)}")
        all_files = []
        for i in data_list:
            tag = str(label_dicts[i.split('/')[0]])
            all_files.append(tag + '_' + i.replace('/', '_').split('wav')[0])
        cmds = [(i, os.path.join(source_path, file).split('wav')[0] + 'wav', out_path + name + "/" + all_files[i]) for i, file in enumerate(data_list)]
        print(cmds[0])
        with multiprocessing.Pool(processes=20) as pool:
            pool.starmap(process_one, cmds)


def process_one(i, f1, f2):
    if i % 1000 == 0:
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
    process_folder('train_list.txt')
    process_folder('validation_list.txt')
    process_folder('testing_list.txt')
    os.system('stty sane')
