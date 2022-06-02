# MaskSpec

This is the Pytorch implementation of paper: Masked Spectrogram Prediction For Self-Supervised Audio Pre-Training.

Continuously Updating :)

# Setting up the experiments environment

Our experiments are based on `cuda 11.5` and `python 3.7.10`:

```
conda create -n audiotrans python=3.7.10
conda activate audiotrans
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

# Settings

- Change ```base_dir``` in ```audioset/dataset.py``` to your own path.
- Change ```hdf5_file``` in ```audioset/get_mean_std.py``` to your own, which is the unbalanced training data of Audioset now.
- Run ```python audioset/get_mean_std.py``` to get the mean and std values in random 10000 samples, and you will get a file named ```mean_std_128.npy``` in your working dir.

# Test scripts

# Train scripts

## Preparing Dataset

Download and prepare the dataset as explained in the [audioset page](https://github.com/kkoutini/PaSST/tree/main/audioset).

You can use the audio files provided by [PANNS](https://github.com/qiuqiangkong/audioset_tagging_cnn).

That is [https://pan.baidu.com/s/13WnzI1XDSvqXZQTS-Kqujg](https://pan.baidu.com/s/13WnzI1XDSvqXZQTS-Kqujg), password: 0vc2

## Vit Training From Scratch on Audioset

The base Vit model can be trained from scratch for example like this (using 8 GPUs):
```
bash scripts/train_from_scratch_vit.sh
```

## Vit Pretraining on Audioset

The base Vit model can be pretrained for example like this (using 8 GPUs):
```
bash scripts/pretrain_vit.sh
```

## Vit Pretraining on Other Datasets (Large scale)

The base Vit model can be pretrained for example like this (using 8 GPUs):
```
bash scripts/submitit_pretrain.sh
```

## Vit Finetuning on Audioset

The base Vit model can be fintuned for example like this (using 8 GPUs):
```
bash scripts/finetune_vit.sh
```


# References

1. [PaSST: Efficient Training of Audio Transformers with Patchout](https://github.com/kkoutini/PaSST)
2. [Masked Autoencoders Are Scalable Vision Learners](https://github.com/facebookresearch/mae)
