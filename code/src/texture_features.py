"""
提取 LBP + GLCM 特征，保存为 features.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))
from preprocess import load_data, apply_mask, rgb_to_gray
from texture import imageLoader

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "original_data"
OUTPUT_DIR = ROOT / "result"

def extract(image, mask):
    masked = apply_mask(image, mask).astype(np.uint8)
    gray = rgb_to_gray(masked).astype(np.uint8)
    loader = imageLoader(gray)
    lbp = loader.LBP_process()
    glcm = loader.GLCM_process()
    glcm_flat = np.array(list(glcm[0]) + list(glcm[1]))
    return np.concatenate([lbp, glcm_flat])

def main():
    images, masks, info = load_data(DATA_DIR)
    X, y = [], []
    for i, (img, mask, inf) in enumerate(zip(images, masks, info)):
        if i % 100 == 0:
            print(f"进度: {i}/{len(images)}")
        X.append(extract(img, mask))
        y.append(inf["dx"])
    
    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv(OUTPUT_DIR / "features.csv", index=False)

if __name__ == "__main__":
    main()
