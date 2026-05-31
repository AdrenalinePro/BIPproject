import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.stats import skew

# ==========================================
# 1. 团队统一的路径配置 (使用 pathlib 更优雅)
# ==========================================
# 自动定位项目根目录 (兼容文件放在 code 或 code/src 下)
current_dir = Path(__file__).parent
ROOT = current_dir.parent.parent if current_dir.name == 'src' else current_dir.parent

DATA_DIR = ROOT / "data" / "original_data"
IMAGE_DIR = DATA_DIR / "image"
MASK_DIR = DATA_DIR / "mask"
RESULT_DIR = ROOT / "result"

# 确保输出目录存在
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 也可以像队友一样把 src 加入系统路径，方便后续调包
sys.path.append(str(ROOT / "code" / "src"))
from preprocess import load_data

# ==========================================
# 2. 颜色特征提取模块
# ==========================================
def extract_color_features(image_rgb, mask):
    features = {}
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return {f"{space}_{ch}_{stat}": 0 for space in ['RGB', 'HSV'] for ch in ['1','2','3'] for stat in ['mean', 'std', 'skew']}
    
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    pixels_rgb = image_rgb[mask_bool]
    pixels_hsv = image_hsv[mask_bool]
    
    channels_rgb = ['R', 'G', 'B']
    for i, ch in enumerate(channels_rgb):
        ch_pixels = pixels_rgb[:, i]
        features[f'RGB_{ch}_mean'] = np.mean(ch_pixels)
        features[f'RGB_{ch}_std'] = np.std(ch_pixels)
        features[f'RGB_{ch}_skew'] = skew(ch_pixels)

    channels_hsv = ['H', 'S', 'V']
    for i, ch in enumerate(channels_hsv):
        ch_pixels = pixels_hsv[:, i]
        features[f'HSV_{ch}_mean'] = np.mean(ch_pixels)
        features[f'HSV_{ch}_std'] = np.std(ch_pixels)
        features[f'HSV_{ch}_skew'] = skew(ch_pixels)
        
    return features

# ==========================================
# 3. 形状特征提取模块
# ==========================================
def extract_shape_features(mask):
    features = {}
    total_pixels = mask.shape[0] * mask.shape[1]
    
    mask_binary = (mask > 127).astype(np.uint8)
    label_img = label(mask_binary)
    props = regionprops(label_img)
    
    if not props:
        return {'area_ratio': 0, 'perimeter': 0, 'eccentricity': 0, 'circularity': 0, 'solidity': 0}
    
    lesion = max(props, key=lambda r: r.area)
    area = lesion.area
    perimeter = lesion.perimeter
    
    features['area_ratio'] = area / total_pixels
    features['perimeter'] = perimeter
    features['eccentricity'] = lesion.eccentricity
    
    if perimeter == 0:
        features['circularity'] = 0
    else:
        features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
        
    features['solidity'] = lesion.solidity
    return features

# ==========================================
# 4. 可视化模块
# ==========================================
def visualize_features(image_rgb, mask, filename):
    masked_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(masked_img)
    axes[1].set_title('Lesion Area (Shape)')
    axes[1].axis('off')
    
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [i], mask, [256], [0, 256])
        axes[2].plot(hist, color=col)
    axes[2].set_title('Color Distribution (RGB)')
    axes[2].set_xlim([0, 256])
    
    plt.tight_layout()
    save_path = RESULT_DIR / f'vis_{filename}'
    plt.savefig(str(save_path))
    plt.close()
    print(f"可视化结果已保存至: {save_path}")

# ==========================================
# 5. 主程序流
# ==========================================
def main():
    print("开始提取颜色和形状特征 (集成预处理版)...")
    
    # 调用第一阶段同学写好的统一接口读取数据
    print("正在通过 preprocess 加载数据，请稍候...")
    images, masks, infos = load_data(DATA_DIR)
    
    all_features = []
    vis_count = 0 
    
    # 遍历内存中加载好的所有数据
    for i in range(len(images)):
        image_rgb = images[i]  # 队友的 load_data 用 PIL 读取，原生就是 RGB，无需转换！
        mask = masks[i]
        info = infos[i]
        
        color_feats = extract_color_features(image_rgb, mask)
        shape_feats = extract_shape_features(mask)
        
        # 将队友解析好的标签(dx)直接放进特征里，对下游分类极其友好
        combined_features = {
            'image_id': info['image_id'],
            'is_augmented': info['is_augmented'],
            'label_dx': info['dx']  # 关键：直接附带疾病类别标签
        }
        combined_features.update(color_feats)
        combined_features.update(shape_feats)
        all_features.append(combined_features)
        
        if vis_count < 3:
            # 构造一个唯一的存图名称
            vis_name = f"id{info['image_id']}_{'aug' if info['is_augmented'] else 'orig'}_{i}.jpg"
            visualize_features(image_rgb, mask, vis_name)
            vis_count += 1
            
    df_features = pd.DataFrame(all_features)
    csv_path = RESULT_DIR / 'color_shape_features.csv'
    df_features.to_csv(csv_path, index=False)
    
    print(f"\n提取完成！共提取了 {len(all_features)} 张图片的特征。")
    print(f"特征数据表已保存至: {csv_path}")

if __name__ == "__main__":
    main()