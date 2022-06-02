import argparse
import multiprocessing
import glob
import os
import soundfile as sf
import numpy as np

# Replace this with the dataset downloaded using PANN scripts
#source_path = "/data/dean/avspeech/"
source_path = "/data/dean/unbalanced_audioset_20W"
# Replace with the output directory
out_path = "/data/dean/whl/mp3_audioset_20w/"

all_num = 0
audio_length = 10

def process_folder(source_path):
    print("now working on : " + source_path)
    os.makedirs(out_path, exist_ok=True)
    all_files = list(glob.glob(source_path + "/*.flac"))
    print(f"it has {len(all_files)}")
    global all_num
    all_num = len(all_files)
    cmds = [(i, file, out_path + os.path.basename(file)[:-4]) for i, file in enumerate(all_files)]
    print(cmds[0])
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(process_one, cmds)

def process_one(i, f1, f2):
    if i % 100 == 0:
        print(f"{i}/{all_num} \t", f1)
    audio, sr = sf.read(f1)
    sample_length = audio_length * sr
    num = audio.shape[0] // sample_length
    count = 0
    for n in range(num):
        en = np.sqrt(np.sum(np.abs(audio[n * sample_length:(n + 1) * sample_length]) ** 2))
        if en >= 1.: # set 1 maybe
            count += 1
            x1 = f2.split('.')[0]+'_'+str(count)+'.flac'
            x2 = f2.split('.')[0]+'_'+str(count)+'.'
            sf.write(x1, audio[n * sample_length:(n + 1) * sample_length], sr)
            os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {x1} -codec:a mp3 -ar 32000 {x2}mp3")
            os.system(f"rm -rf {x1}")

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
    print("I will work on this folder:")
    print(source_path)
    process_folder(source_path)
    os.system('stty sane')
