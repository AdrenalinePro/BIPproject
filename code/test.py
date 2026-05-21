import pandas as pd
from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data" / "original_data"
RESULT_DIR = Path(__file__).parent.parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

# 读取标签
label_df = pd.read_csv(DATA_DIR / "label.csv")

# 图片目录
image_dir = DATA_DIR / "image"

# 筛选原始图像（不含_aug的）
original_images = [f for f in image_dir.glob("*.jpg") if "_aug" not in f.name]
print(f"原始图像数量: {len(original_images)}")

# 生成info
info_data = []
for img_path in sorted(original_images, key=lambda x: int(x.stem)):
    img = Image.open(img_path)
    width, height = img.size
    image_id = int(img_path.stem)
    info_data.append({"image_id": image_id, "width": width, "height": height})

info_df = pd.DataFrame(info_data)

# 与label.csv合并（保留label.csv的前两列结构）
result_df = label_df[["image_id"]].copy()
result_df["dx"] = label_df["dx"]
result_df["resolution"] = info_df["width"].astype(str) + "*" + info_df["height"].astype(str)

# 保存
result_df.to_csv(RESULT_DIR / "info.csv", index=False)
print(f"已保存到 {RESULT_DIR / 'info.csv'}")
print(result_df.head(10))