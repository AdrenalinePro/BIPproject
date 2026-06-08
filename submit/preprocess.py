"""
submit/ 版本的 preprocess 最小子集

只保留 submit 流程真正用到的两个函数：
  - apply_mask: texture_features.extract() 用来把图像 * mask
  - rgb_to_gray: texture_features.extract() 用来从 RGB 取灰度
其余 (load_data, remove_hair) 都和推理无关，全部省略。
"""
import numpy as np


def apply_mask(image, mask):
    """
    将图像与 mask 相乘，保留 mask 覆盖区域。

    参数:
        image: np.ndarray, 形状 (H, W, C) 的 RGB 图像或 (H, W) 的灰度图
        mask:  np.ndarray, 形状 (H, W), 值域 0-255 的灰度图
    返回:
        np.ndarray, 与 mask 相乘后的图像，形状与输入 image 一致
    """
    mask_normalized = mask / 255.0
    if image.ndim == 3:
        return image * mask_normalized[:, :, np.newaxis]
    return image * mask_normalized


def rgb_to_gray(image):
    """
    RGB -> 灰度 (uint8)。与 code/src/preprocess.py 完全一致。
    """
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
