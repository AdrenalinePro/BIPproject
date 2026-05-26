import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_data(data_dir):
    """
    加载所有图像和对应的mask到内存，并记录每张图像的信息（二维图像）。

    参数:
        data_dir: Path, 指向 original_data 文件夹的路径

    返回:
        images: np.ndarray, 形状为 (N, H, W, C)，存储所有二维RGB图像
        masks: np.ndarray, 形状为 (N, H, W)，存储所有二维mask
        info: list of dict, 每个元素包含 {'image_id': int, 'dx': str, 'is_augmented': bool}
            其中 dx 为病变类型，is_augmented 表示是否为增强图像
    """
    image_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    label_df = pd.read_csv(data_dir / "label.csv")

    image_files = sorted(image_dir.glob("*.jpg"), key=lambda x: x.stem)

    images = []
    masks = []
    info = []

    for img_path in image_files:
        stem = img_path.stem
        is_augmented = "_aug" in stem

        if is_augmented:
            base_id = int(stem.replace("_aug1", "").replace("_aug2", ""))
            label_id = stem
        else:
            base_id = int(stem)
            label_id = str(base_id)

        label_row = label_df[label_df["image_id"] == label_id].iloc[0]
        img = np.array(Image.open(img_path))

        if is_augmented:
            mask_path = mask_dir / f"mask_{stem}.jpg"
        else:
            mask_path = mask_dir / f"mask_{base_id}.jpg"
        mask = np.array(Image.open(mask_path))

        images.append(img)
        masks.append(mask)
        info.append({"image_id": base_id, "dx": label_row["dx"], "is_augmented": is_augmented})

    return np.array(images), np.array(masks), info


def apply_mask(image, mask):
    """
    将图像与mask相乘，保留mask覆盖的区域。

    参数:
        image: np.ndarray, 形状为 (H, W, C) 的RGB图像或 (H, W) 的灰度图
        mask: np.ndarray, 形状为 (H, W)，值为0-255的灰度图

    返回:
        np.ndarray, 与mask相乘后的图像，形状与输入image一致
    """
    mask_normalized = mask / 255.0
    if image.ndim == 3:
        return image * mask_normalized[:, :, np.newaxis]
    else:
        return image * mask_normalized


def rgb_to_gray(image):
    """
    将RGB图像转换为灰度图。

    参数:
        image: np.ndarray, 形状为 (H, W, C) 的RGB图像

    返回:
        np.ndarray, 灰度图像，形状为 (H, W)
    """
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def remove_hair(image):
    """
    去除图像中的毛发。

    参数:
        image: np.ndarray, 形状为 (H, W, C) 的RGB图像

    返回:
        result: np.ndarray, 去毛发后的图像
        hair_mask: np.ndarray, 检测到的毛发区域mask
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    kernel2 = np.ones((3, 3), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel2)

    result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)

    return result, hair_mask