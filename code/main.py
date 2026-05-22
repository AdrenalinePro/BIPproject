import pandas as pd
from pathlib import Path
from src.preprocess import *



DATA_DIR = Path(__file__).parent.parent / "data" / "original_data"
# 以下是个例子，后面可以删除
# 读取标签
label_df = pd.read_csv(DATA_DIR / "label.csv")
print(f"标签数量: {len(label_df)}")
print(label_df.head())

# 图片目录
image_dir = DATA_DIR / "image"
print(f"\n图片数量: {len(list(image_dir.glob('*.jpg')))}")