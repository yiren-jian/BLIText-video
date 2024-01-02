# Bootstrapping Vision-Language Learning with Decoupled Language Pre-training

This repo covers implementations of VideoCaption + Pformer in **Bootstrapping Vision-Language Learning with Decoupled Language Pre-training**. The code is developed based on [LAVIS](https://github.com/salesforce/LAVIS/) project (cloned on Mar 9, 2023).

We mainly add following files in `lavis/models/blip2_models` (Pformer is named darkformer during the development):

- [x] `video_feature_opt.py`
- [x] `video_feature_opt_new.py`
- [x] `video_feature_opt_stage1.py`
- [x] `video_feature_opt_stage2.py`

## Installation

```bash
conda create -n lavis python=3.8
conda activate lavis
pip install -e .
```

The experiments are carried out on a single RTX-A6000. We provide our environment in `pip_freeze.txt`, for closely reproducing of our results. 

## Data Preparation
I3D features of VATEX can be downloaded from [VATEX](https://eric-xw.github.io/vatex-website/download.html).

## Pre-trained Models
For video captioning, we use a P-former pretrained with 40M data (>12M). The pretrained P-former and the captioner weights will be released soon.

## Training
stage 1
```bash
bash run_scripts/blip2/train/train_caption_vatex_stage1.sh
```

stage 2
```bash
bash run_scripts/blip2/train/train_caption_vatex_stage2.sh
```

## Evaluation
We use [CLIPScore](https://github.com/jmhessel/clipscore). Put `compute_score.py` in the `/clipscore` and run:
```
python compute_score.py
```

## Training and Evaluation Logs
You can find our training (stage-1 and stage-2) and evaluation (w/ scores) logs [here](training_logs/)
