# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a skin lesion classification project (BIPproject) using multi-stage feature extraction and machine learning. It classifies lesions into 3 types: mel (melanoma), nv (nevus), vasc (vascular). Data is 224×224 RGB images with corresponding masks.

## Directory Structure

- `data/original_data/` — input images (`image/`), masks (`mask/`), and `label.csv`
- `code/src/` — all feature extraction and preprocessing code
- `code/main.py` — entry point for color+shape feature extraction
- `result/` — outputs: features CSVs, confusion matrices, trained models, visualizations
- `deeplearning/` — ResNet18 training (`resnet_train.py`), outputs (Grad-CAM, confusion matrix, train curves)

## Key Files and Architecture

### Central Data Loading
[code/src/preprocess.py](code/src/preprocess.py) — `load_data(data_dir)` returns `(images, masks, info)` where `info` contains `image_id`, `dx` (label), and `is_augmented`. This is the standard interface all other modules depend on.

### Feature Extraction (run via `code/main.py`)
- **Color features** (`extract_color_features`) — RGB and HSV channel statistics (mean, std, skew) over masked regions
- **Shape features** (`extract_shape_features`) — area_ratio, perimeter, eccentricity, circularity, solidity from regionprops
- Output: `result/color_shape_features.csv`

### Texture Features
[code/src/texture.py](code/src/texture.py) — `imageLoader` class with `LBP_process()` (256-bin histogram) and `GLCM_process()` (contrast, ASM, IDM, correlation, entropy, sum_variance at distances 1 and 2).

### Classifier Training
[code/src/classifier.py](code/src/classifier.py) — reads `result/features.csv` (texture features), trains 5 classifiers (RandomForest, SVM, KNN, GradientBoosting, LogisticRegression), saves models to `result/classifiers/<Name>/model.pkl` and confusion matrices to each classifier's folder.

### Deep Learning
[deeplearning/resnet_train.py](deeplearning/resnet_train.py) — ResNet18 fine-tuning with 70% frozen layers, class-weighted loss (mel class weighted 1.65×), data augmentation, saves best model to `best_resnet.pth`. Dataset lives at `./skin_dataset/train` and `./skin_dataset/test` (not in this repo).

## Commands

```bash
# Feature extraction (color + shape)
python code/main.py

# Texture feature extraction (interactive - supply image path)
python code/src/texture.py <image_path>

# Classifier training
python code/src/classifier.py

# Deep learning training (requires skin_dataset in working directory)
python deeplearning/resnet_train.py

# Run preprocessing test/visualization
python code/test.py
```

## Data Format Notes

- All images and masks are 224×224, normalized. Original images are RGB.
- Augmented images have `_aug1` or `_aug2` suffixes in filename.
- `load_data` returns PIL-loaded images (RGB format) — no BGR conversion needed for RGB operations.
- For GLCM texture features, images are quantized to 16 gray levels.
