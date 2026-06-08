"""
颜色 + 形状 + ABCD 特征提取 —— submit 推理版本

只保留推理需要的三个函数 (extract_color_features / extract_shape_features /
extract_abcd_features) 以及它们依赖的私有辅助函数与常量。
完全去掉原 feature_extraction.py 中的可视化 (matplotlib) 与 main() 入口。
"""
import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans


# ==========================================
# 1. 颜色特征提取  (4 空间 × 3 通道 × 6 统计量 = 72 维)
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
    'RGB':   lambda img: (img,                                 ['R', 'G', 'B']),
    'HSV':   lambda img: (cv2.cvtColor(img, cv2.COLOR_RGB2HSV),   ['H', 'S', 'V']),
    'Lab':   lambda img: (cv2.cvtColor(img, cv2.COLOR_RGB2LAB),   ['L', 'a', 'b']),
    'YCbCr': lambda img: (cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb), ['Y', 'Cb', 'Cr']),
}

STATS = ['mean', 'std', 'skew', 'kurt', 'p10', 'p90']


def extract_color_features(image_rgb, mask):
    """在 mask 区域内的 4 色彩空间像素上各计算 6 个统计量，共 72 维。"""
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
# 2. 形状特征提取  (5 regionprops + 6 几何 + 7 Hu 矩 = 18 维)
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
# 3. ABCD 临床规则特征  (= 7 维)
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
