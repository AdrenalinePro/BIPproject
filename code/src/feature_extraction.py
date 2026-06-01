"""
颜色 + 形状 + ABCD 临床规则 特征提取

输出 result/color_shape_features.csv，列结构：
    image_id, is_augmented, label_dx,
    <72 颜色特征>, <18 形状特征>, <6 ABCD 特征>

可视化：对前 4 张图做 3x3 面板综合可视化，保存到 result/vis_*.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans

# ==========================================
# 路径配置
# ==========================================
current_dir = Path(__file__).parent
ROOT = current_dir.parent.parent if current_dir.name == 'src' else current_dir.parent
DATA_DIR = ROOT / "data" / "original_data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT / "code" / "src"))
from preprocess import load_data


# ==========================================
# 1. 颜色特征提取
#    4 个色彩空间 × 3 通道 × 6 统计量 = 72 维
# ==========================================
def _safe_float(x):
    """将可能为 NaN/Inf 的标量转成 0。"""
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _stats(pixels):
    """对一维像素向量计算 6 个统计量。"""
    pixels = pixels.astype(np.float64)
    if pixels.size == 0:
        return dict(mean=0.0, std=0.0, skew=0.0, kurt=0.0, p10=0.0, p90=0.0)
    return dict(
        mean=float(np.mean(pixels)),
        std=float(np.std(pixels)),
        skew=_safe_float(skew(pixels)),
        kurt=_safe_float(kurtosis(pixels)),
        p10=float(np.percentile(pixels, 10)),
        p90=float(np.percentile(pixels, 90)),
    )


COLOR_SPACES = {
    'RGB':   lambda img: (img,                              ['R', 'G', 'B']),
    'HSV':   lambda img: (cv2.cvtColor(img, cv2.COLOR_RGB2HSV),   ['H', 'S', 'V']),
    'Lab':   lambda img: (cv2.cvtColor(img, cv2.COLOR_RGB2LAB),   ['L', 'a', 'b']),
    'YCbCr': lambda img: (cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb), ['Y', 'Cb', 'Cr']),
}

STATS = ['mean', 'std', 'skew', 'kurt', 'p10', 'p90']


def extract_color_features(image_rgb, mask):
    """在 mask 区域内的 4 色彩空间 (RGB/HSV/Lab/YCbCr) 像素上各计算 6 个统计量。"""
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return {f'{sp}_{ch}_{stat}': 0.0
                for sp in COLOR_SPACES for ch in ['1', '2', '3'] for stat in STATS}

    features = {}
    for sp_name, converter in COLOR_SPACES.items():
        img_converted, ch_names = converter(image_rgb)
        for i, ch in enumerate(ch_names):
            pixels = img_converted[..., i][mask_bool]
            s = _stats(pixels)
            for stat in STATS:
                features[f'{sp_name}_{ch}_{stat}'] = s[stat]
    return features


# ==========================================
# 2. 形状特征提取
#    5 个 regionprops + 6 个新几何量 + 7 个 Hu 矩 = 18 维
# ==========================================
def _hu_moments(mask_binary):
    """7 个 Hu 不变矩；使用 -sign(x)*log10(|x|) 平移，量级更可比。"""
    moments = cv2.moments(mask_binary.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return [_safe_float(v) for v in hu_log]


def extract_shape_features(mask):
    features = {}
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_binary = (mask > 127).astype(np.uint8)
    label_img = label(mask_binary)
    props = regionprops(label_img)

    if not props:
        empty = {f'shape_{k}': 0.0 for k in [
            'area_ratio', 'perimeter', 'eccentricity', 'circularity', 'solidity',
            'aspect_ratio', 'extent', 'equivalent_diameter', 'orientation',
            'major_axis', 'minor_axis',
        ]}
        empty.update({f'hu_{i+1}': 0.0 for i in range(7)})
        return empty

    lesion = max(props, key=lambda r: r.area)
    area = lesion.area
    perimeter = lesion.perimeter

    features['area_ratio'] = area / total_pixels
    features['perimeter'] = perimeter
    features['eccentricity'] = lesion.eccentricity
    features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    features['solidity'] = lesion.solidity

    minr, minc, maxr, maxc = lesion.bbox
    bbox_h = max(maxr - minr, 1)
    bbox_w = max(maxc - minc, 1)
    features['aspect_ratio'] = bbox_w / bbox_h
    features['extent'] = area / (bbox_h * bbox_w)
    # skimage 0.26 起 axis_major_length/equivalent_diameter_area 取代旧名；新名不可用时回退
    if hasattr(lesion, 'equivalent_diameter_area'):
        features['equivalent_diameter'] = lesion.equivalent_diameter_area
    else:
        features['equivalent_diameter'] = lesion.equivalent_diameter
    features['orientation'] = lesion.orientation
    if hasattr(lesion, 'axis_major_length'):
        features['major_axis'] = lesion.axis_major_length
        features['minor_axis'] = lesion.axis_minor_length
    else:
        features['major_axis'] = lesion.major_axis_length
        features['minor_axis'] = lesion.minor_axis_length

    for i, h in enumerate(_hu_moments(mask_binary)):
        features[f'hu_{i+1}'] = h

    return features


# ==========================================
# 3. ABCD 临床规则特征（新增支路）= 6 维
# ==========================================
def _compute_asymmetry(mask_binary):
    """A = 1 - 重叠率。把 mask 绕质心旋转 180°，计算与原 mask 的交集占原 mask 面积的比例。"""
    moments = cv2.moments(mask_binary)
    area = int(mask_binary.sum())
    if moments['m00'] == 0 or area == 0:
        return 1.0
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    h, w = mask_binary.shape
    M = cv2.getRotationMatrix2D((cx, cy), 180, 1.0)
    rotated = cv2.warpAffine(mask_binary, M, (w, h), flags=cv2.INTER_NEAREST)
    intersection = int(np.logical_and(mask_binary, rotated).sum())
    return 1.0 - intersection / area


def _largest_contour(mask_binary):
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _compute_border_irregularity(mask_binary):
    """B = 边界点到质心距离的变异系数 (std / mean)。值越大边界越不规则。"""
    contour = _largest_contour(mask_binary)
    if contour is None:
        return 0.0
    points = contour.squeeze()
    if points.ndim != 2 or len(points) < 2:
        return 0.0
    moments = cv2.moments(mask_binary)
    if moments['m00'] == 0:
        return 0.0
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    mean_d = float(dists.mean())
    if mean_d == 0:
        return 0.0
    return float(dists.std() / mean_d)


def _compute_color_variegation(image_rgb, mask_bool, k=5, min_ratio=0.05):
    """C = mask 内 KMeans 聚类后占比 > 5% 的聚类数（颜色多样性）。同时返回 HSV 三通道 std 之和。"""
    n_pix = int(mask_bool.sum())
    if n_pix < k:
        return 0.0, 0.0
    pixels = image_rgb[mask_bool]
    km = KMeans(n_clusters=k, n_init=3, random_state=0)
    labels = km.fit_predict(pixels)
    counts = np.bincount(labels, minlength=k)
    ratios = counts / counts.sum()
    n_effective = int((ratios > min_ratio).sum())
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv_std_sum = sum(float(image_hsv[..., c][mask_bool].std()) for c in range(3))
    return float(n_effective), hsv_std_sum


def _compute_diameter(mask_binary):
    """D = 等效直径 = 2 * sqrt(area / π)。"""
    area = int(mask_binary.sum())
    if area == 0:
        return 0.0
    return float(2.0 * np.sqrt(area / np.pi))


def extract_abcd_features(image_rgb, mask):
    mask_binary = (mask > 127).astype(np.uint8)
    mask_bool = mask > 0
    label_img = label(mask_binary)
    props = regionprops(label_img)

    features = {}
    features['ABCD_A_asymmetry'] = _compute_asymmetry(mask_binary)
    features['ABCD_B_border_CV'] = _compute_border_irregularity(mask_binary)
    n_clusters, hsv_std_sum = _compute_color_variegation(image_rgb, mask_bool, k=5)
    features['ABCD_C_n_color_clusters'] = n_clusters
    features['ABCD_C_hsv_std_sum'] = hsv_std_sum
    features['ABCD_D_diameter'] = _compute_diameter(mask_binary)

    if props:
        lesion = max(props, key=lambda r: r.area)
        features['ABCD_area'] = float(lesion.area)
        features['ABCD_perimeter'] = float(lesion.perimeter)
    else:
        features['ABCD_area'] = 0.0
        features['ABCD_perimeter'] = 0.0

    return features


# ==========================================
# 4. 可视化（3×3 面板）
# ==========================================
def visualize_features(image_rgb, mask, info, filename):
    mask_binary = (mask > 127).astype(np.uint8)
    mask_bool = mask > 0
    h, w = mask_binary.shape

    A = _compute_asymmetry(mask_binary)
    B = _compute_border_irregularity(mask_binary)
    n_clusters, _ = _compute_color_variegation(image_rgb, mask_bool, k=5)
    D = _compute_diameter(mask_binary)

    moments = cv2.moments(mask_binary)
    if moments['m00'] > 0:
        cx, cy = moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
    else:
        cx, cy = 0.0, 0.0

    M_rot = cv2.getRotationMatrix2D((cx, cy), 180, 1.0)
    rotated_mask = cv2.warpAffine(mask_binary, M_rot, (w, h), flags=cv2.INTER_NEAREST)

    if mask_bool.sum() >= 5:
        km = KMeans(n_clusters=5, n_init=3, random_state=0)
        labels = km.fit_predict(image_rgb[mask_bool])
        cluster_img = np.zeros_like(image_rgb)
        cluster_img[mask_bool] = km.cluster_centers_[labels].astype(np.uint8)
    else:
        cluster_img = image_rgb.copy()

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(image_rgb); ax.set_title('Original', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mask_binary, cmap='gray')
    ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
    ax.set_title('Mask + Centroid', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(cv2.bitwise_and(image_rgb, image_rgb, mask=mask))
    ax.set_title('Lesion ROI', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[1, 0])
    contour_img = image_rgb.copy()
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    ax.imshow(contour_img); ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
    ax.set_title('Contour', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    asym_vis = np.zeros((h, w, 3), dtype=np.int32)
    asym_vis[mask_binary > 0] = [255, 80, 80]
    asym_vis[rotated_mask > 0] += np.array([0, 80, 255], dtype=np.int32)
    ax.imshow(np.clip(asym_vis, 0, 255).astype(np.uint8))
    ax.set_title(f'A (Asymmetry) = {A:.3f}', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(cluster_img)
    ax.set_title(f'C (Color clusters, k=5) = {int(n_clusters)}', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[2, 0])
    contour = _largest_contour(mask_binary)
    if contour is not None:
        pts = contour.squeeze()
        if pts.ndim == 2 and len(pts) >= 2:
            dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            ax.hist(dists, bins=20, color='steelblue', edgecolor='k')
            ax.axvline(dists.mean(), color='r', linestyle='--', label=f'mean={dists.mean():.1f}')
            ax.set_xlabel('Distance from centroid (px)'); ax.set_ylabel('Count'); ax.legend()
    ax.set_title(f'B (Border CV) = {B:.3f}', fontsize=12)

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(mask_binary, cmap='gray')
    circle = plt.Circle((cx, cy), D / 2.0, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.set_title(f'D (Equivalent Diameter) = {D:.1f} px', fontsize=12); ax.axis('off')

    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    props_list = regionprops(label(mask_binary))
    lesion = max(props_list, key=lambda r: r.area) if props_list else None
    circ = (4 * np.pi * lesion.area / lesion.perimeter ** 2) if lesion and lesion.perimeter else 0.0
    text_lines = [
        f"label       : {info['dx']}",
        f"image_id    : {info['image_id']}",
        f"augmented   : {info['is_augmented']}",
        '',
        '—— Color (4 spaces) ——',
        f'RGB R mean  : {image_rgb[..., 0][mask_bool].mean():.1f}',
        f'Lab L mean  : {cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[..., 0][mask_bool].mean():.1f}',
        '',
        '—— Shape ——',
        f'area         : {lesion.area if lesion else 0}',
        f'perimeter    : {lesion.perimeter if lesion else 0:.1f}',
        f'eccentricity : {lesion.eccentricity if lesion else 0:.3f}',
        f'circularity  : {circ:.3f}',
        f'solidity     : {lesion.solidity if lesion else 0:.3f}',
        '',
        '—— ABCD ——',
        f'A (asym)   = {A:.3f}',
        f'B (border) = {B:.3f}',
        f'C (cluster)= {int(n_clusters)}',
        f'D (diam)   = {D:.1f}',
    ]
    ax.text(0.03, 0.97, '\n'.join(text_lines), transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10)

    fig.suptitle(f'Feature Extraction Visualization - {filename}', fontsize=14, fontweight='bold')
    save_path = RESULT_DIR / f'vis_{filename}'
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'可视化结果已保存至: {save_path}')


# ==========================================
# 5. 主程序流
# ==========================================
def main():
    print("开始提取颜色 + 形状 + ABCD 特征 ...")
    print("正在通过 preprocess 加载数据，请稍候...")
    images, masks, infos = load_data(DATA_DIR)

    all_features = []
    vis_count = 0
    n_visualize = 4  # 前 4 张图生成可视化

    for i in range(len(images)):
        image_rgb = images[i]
        mask = masks[i]
        info = infos[i]

        color_feats = extract_color_features(image_rgb, mask)
        shape_feats = extract_shape_features(mask)
        abcd_feats = extract_abcd_features(image_rgb, mask)

        combined = {
            'image_id': info['image_id'],
            'is_augmented': info['is_augmented'],
            'label_dx': info['dx'],
        }
        combined.update(color_feats)
        combined.update(shape_feats)
        combined.update(abcd_feats)
        all_features.append(combined)

        if vis_count < n_visualize:
            vis_name = f"id{info['image_id']}_{'aug' if info['is_augmented'] else 'orig'}_{i}.jpg"
            visualize_features(image_rgb, mask, info, vis_name)
            vis_count += 1

    df_features = pd.DataFrame(all_features)
    csv_path = RESULT_DIR / 'color_shape_features.csv'
    df_features.to_csv(csv_path, index=False)

    n_color = len(extract_color_features(images[0], masks[0]))
    n_shape = len(extract_shape_features(masks[0]))
    n_abcd = len(extract_abcd_features(images[0], masks[0]))
    print(f"\n提取完成！共 {len(all_features)} 张图片。")
    print(f"特征维度: 3 (meta) + {n_color} (color) + {n_shape} (shape) + {n_abcd} (ABCD) = {df_features.shape[1]}")
    print(f"特征数据表已保存至: {csv_path}")

if __name__ == "__main__":
    main()
