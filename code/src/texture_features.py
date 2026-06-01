"""
提取 LBP + GLCM 特征（仅在 mask 区域内统计），保存为 features.csv

输出格式与 color_shape_features.csv 对齐：
  - 第一行为列名
  - 前 3 列：image_id, is_augmented, label_dx
  - 之后为 256 个 LBP 直方图 bin + 12 个 GLCM 特征（d=1, d=2 各 6 个）

LBP 中心像素必须位于 mask 内才计入直方图；
GLCM 仅累加两个像素同时位于 mask 内的邻接对，避免背景 (0,0) 共现淹没结果。
"""
from pathlib import Path
import numpy as np
import pandas as pd

from preprocess import load_data, apply_mask, rgb_to_gray
from texture import imageLoader

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "original_data"
OUTPUT_DIR = ROOT / "result"

# GLCM 标量特征顺序与 GLCM_process 返回值保持一致
GLCM_FEATURE_NAMES = ['contrast', 'asm', 'idm', 'correlation', 'entropy', 'sum_variance']
GLCM_DISTANCES = [1, 2]

def _build_feature_names():
    """生成 256 + 12 个特征的列名。"""
    names = [f'LBP_{i}' for i in range(256)]
    for d in GLCM_DISTANCES:
        for stat in GLCM_FEATURE_NAMES:
            names.append(f'GLCM_d{d}_{stat}')
    return names

FEATURE_NAMES = _build_feature_names()
N_LBP = 256
N_GLCM_PER_D = len(GLCM_FEATURE_NAMES)

def extract(image, mask):
    """
    在 mask 区域内提取 LBP 直方图(256) + GLCM 6 特征 × 2 距离

    参数:
        image: np.ndarray, 形状 (H, W, 3) 的 RGB 图像
        mask:  np.ndarray, 形状 (H, W) 的 0/255 灰度 mask

    返回:
        np.ndarray, 长度为 256 + 12 = 268 的特征向量
    """
    masked = apply_mask(image, mask).astype(np.uint8)
    gray = rgb_to_gray(masked).astype(np.uint8)
    loader = imageLoader(gray, mask=mask)
    lbp = loader.LBP_process()
    glcm = loader.GLCM_process()
    glcm_flat = np.concatenate([np.asarray(glcm[0]), np.asarray(glcm[1])])
    return np.concatenate([lbp, glcm_flat])

def main():
    images, masks, info = load_data(DATA_DIR)
    rows = []
    for i, (img, mask, inf) in enumerate(zip(images, masks, info)):
        if i % 100 == 0:
            print(f"进度: {i}/{len(images)}")
        feature_vec = extract(img, mask)
        row = {
            'image_id': inf['image_id'],
            'is_augmented': inf['is_augmented'],
            'label_dx': inf['dx'],
        }
        for name, val in zip(FEATURE_NAMES, feature_vec):
            row[name] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=['image_id', 'is_augmented', 'label_dx'] + FEATURE_NAMES)
    out_path = OUTPUT_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"已写入 {out_path}，共 {len(df)} 行 × {df.shape[1]} 列")

if __name__ == "__main__":
    main()
