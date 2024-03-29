# Bootstrapping Vision-Language Learning with Decoupled Language Pre-training

This repo covers implementations of VideoCaption + Pformer in **Bootstrapping Vision-Language Learning with Decoupled Language Pre-training**. The code is developed based on [LAVIS](https://github.com/salesforce/LAVIS/) project (cloned on Mar 9, 2023).

We mainly add following files in `lavis/models/blip2_models` (Pformer is named darkformer during the development):

- [x] `video_feature_opt.py`
- [x] `video_feature_opt_new.py`
- [x] `video_feature_opt_stage1.py`
- [x] `video_feature_opt_stage2.py`

## Installation

```
# install lavis based on official LAVIS guideline
conda create -n lavis python=3.8
conda activate lavis
pip install -e .

# fix package version issues, use transformers==4.26.1
pip install -r pip_freeze.txt
```
The experiments are carried out on a single RTX-A6000. We provide our environment in `pip_freeze.txt`, for closely reproducing of our results. 

## Data Preparation
I3D features of VATEX can be downloaded from [VATEX](https://eric-xw.github.io/vatex-website/download.html).

## Pre-trained Models
~~For video captioning, we use a P-former pretrained with 40M data (>12M). The pretrained P-former and the captioner weights will be released soon.~~

Please use the pretrained models from [here](https://www.dropbox.com/scl/fo/wkssgsqbqj7qqqcwlxkwp/h?rlkey=33ydfikubq6kaun74uf7lm9ge&dl=0).

- [x] P-former: `models/ours/pformer/checkpoint_60000.pth`.
- [x] Stage 1: `models/ours/Caption_vatex_stage1`.
- [x] Stage 2: `models/ours/Caption_vatex_stage2`.
- [x] You will find the generated captions here: `models/ours/Caption_vatex_stage2/20240102015/result`.

## Training
stage 1
```bash
bash run_scripts/blip2/train/train_caption_vatex_stage1.sh
```

stage 2
```bash
bash run_scripts/blip2/train/train_caption_vatex_stage2.sh
```

You could omit the `missing keys` warning, see related discussion [here](https://github.com/yiren-jian/BLIText-video/issues/1)

## Evaluation
We use [CLIPScore](https://github.com/jmhessel/clipscore). Put `compute_score.py` in the `/clipscore` and run:
```
python compute_score.py
```

## Training and Evaluation Logs
You can find our training (stage-1 and stage-2) and evaluation (w/ scores) logs [here](training_logs/)
