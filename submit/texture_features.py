"""
纹理特征 (LBP 直方图 + GLCM 统计量) 提取 —— submit 推理版本

只暴露推理需要的接口:
  - extract(image, mask) -> np.ndarray, 长度 268 的特征向量
  - FEATURE_NAMES:        268 个特征列名（仅供诊断 / 输出 CSV 时使用）
  - N_LBP, N_GLCM_PER_D:  维度常量
"""
import numpy as np

from preprocess import apply_mask, rgb_to_gray
from texture import imageLoader

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
