# DreamRec 2.0

Consistency and Distillation Models for Sequential Recommendation

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Training](#training)

  * [Consistency Training](#consistency-training)
  * [Distillation Training](#distillation-training)
* [Evaluation](#evaluation)
* [Scripts](#scripts)
* [Configuration](#configuration)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)

## Overview

DreamRec-2.0 implements consistency models and diffusion-based distillation techniques for sequential recommendation tasks, offering both one-step consistency training and multi-step teacher-student distillation.

## Features

* One-step sampling consistency model for rapid inference
* Teacher-student distillation with customizable rollout steps
* Support for linear, exponential, and cosine beta schedules
* Modular architecture leveraging PyTorch

## Prerequisites

* Python 3.7 or higher
* PyTorch 2.7.0 or higher
* pandas
* numpy
* tqdm

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository_url>
cd dreamrec
pip install -r requirements.txt
```

## Data Preparation

Prepare your dataset under `data/<dataset_name>` with the following files:

* `data_statis.df` (pickle of dataset statistics)
* `train_data.df`, `val_data.df`, `test_data.df` (pickled DataFrames of sequences)

## Training

### Consistency Training

```bash
python -u isolation.py \
  --timesteps 500 \
  --beta_sche cosine \
  --w 0 \
  --optimizer adamw \
  --diffuser_type mlp1 \
  --lr 0.001 \
  --random_seed 100
```

### Distillation Training

```bash
python -u distillation.py \
  --teacher_ckpt path/to/teacher.ckpt \
  --timesteps 500 \
  --beta_sche cosine \
  --w 0 \
  --infer_steps 4 \
  --lr 0.005 \
  --diffuser_type mlp1 \
  --optimizer adamw
  --random_seed 100
```

## Evaluation

Run evaluation with the `--eval` flag:

```bash
python isolation.py --eval --load_model_num <ckpt_num>
python distillation.py --eval --teacher_ckpt path/to/teacher.ckpt --load_model_num <ckpt_num>
```

## Scripts

* `diffusion.py`: Defines beta schedules and diffusion classes (`Teacher`, `Student`, `Consistency`).
* `isolation.py`: Script for training and evaluating the consistency model.
* `distillation.py`: Script for teacher-student distillation training.
* `tenc.py`: Transformer encoder (`Tenc`) and sequence processing logic.
* `modules.py`: Multi-head attention and position-wise feed-forward modules.
* `utility.py`: Helper functions for evaluation, model saving, and reproducibility.

## Configuration

All scripts support command-line arguments. Use `--help` to see available options:

```bash
python isolation.py --help
```

## Project Structure

```
.
├── diffusion.py
├── distillation.py
├── isolation.py
├── modules.py
├── tenc.py
├── utility.py
├── data/
│   └── yc/
│       ├── data_statis.df
│       ├── train_data.df
│       ├── val_data.df
│       └── test_data.df
├── ckpt/
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
