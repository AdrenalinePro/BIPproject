import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.preprocess import load_data, apply_mask, rgb_to_gray

DATA_DIR = Path(__file__).parent.parent / "data" / "original_data"
RESULT_DIR = Path(__file__).parent.parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# 加载数据
images, masks, info = load_data(DATA_DIR)
print(f"共加载 {len(images)} 张图像")

# 随机挑选4张
indices = np.random.choice(len(images), 4, replace=False)

fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for row, idx in enumerate(indices):
    img = images[idx]
    mask = masks[idx]

    # 原图
    axes[row, 0].imshow(img)
    axes[row, 0].set_title("Original")
    axes[row, 0].axis("off")

    # 蒙版
    axes[row, 1].imshow(mask, cmap="gray")
    axes[row, 1].set_title("Mask")
    axes[row, 1].axis("off")

    # 应用蒙版后的图像
    masked = apply_mask(img, mask)
    axes[row, 2].imshow(masked.astype(np.uint8))
    axes[row, 2].set_title("Masked")
    axes[row, 2].axis("off")

    # 灰度图
    gray = rgb_to_gray(img)
    axes[row, 3].imshow(gray, cmap="gray")
    axes[row, 3].set_title("Grayscale")
    axes[row, 3].axis("off")

    # 应用蒙版后的灰度图
    gray_masked = apply_mask(gray, mask)
    axes[row, 4].imshow(gray_masked, cmap="gray")
    axes[row, 4].set_title("Gray + Mask")
    axes[row, 4].axis("off")

plt.tight_layout()
plt.savefig(RESULT_DIR / "preprocess.jpg", dpi=150)
plt.close()
print(f"可视化已保存到 {RESULT_DIR / 'preprocess.jpg'}")