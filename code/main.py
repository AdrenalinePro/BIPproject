import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.preprocess import load_data, apply_mask, rgb_to_gray

DATA_DIR = Path(__file__).parent.parent / "data" / "original_data"
RESULT_DIR = Path(__file__).parent.parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

# 加载数据
images, masks, info = load_data(DATA_DIR)
print(f"共加载 {len(images)} 张图像")