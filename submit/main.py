"""
submit/main.py  —  推理入口

读取 submit/image/ 中的所有 .jpg 图片（按 stem 字典序遍历），
对每张图在 submit/mask/ 中找对应的 mask（命名约定: mask_<stem>.jpg），
按训练时的特征顺序 (纹理 268 + 颜色 72 + 形状 18 + ABCD 7 = 365 维)
提取特征向量，喂入预训练的 Stacking 模型，预测类别，输出 submit/output.csv。

输出 CSV 列: image_id, dx
  - image_id  使用图片完整 stem (例如 "1" / "1_aug1" / "100_aug2")
  - dx        预测标签 ∈ {mel, nv, vasc}

用法:
    python submit/main.py
"""
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from PIL import Image

# 关闭 sklearn 训练时残留的版本警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 让同目录下的模块可被 import
SUBMIT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SUBMIT_DIR))

from texture_features import extract as extract_texture                 # 268 维
from features_cs import (
    extract_color_features,                                             # 72 维
    extract_shape_features,                                             # 18 维
    extract_abcd_features,                                              # 7 维
)

# ==========================================
# 颜色 + 形状 + ABCD 的列名固定顺序 (97 维)
# 与 code/src/feature_extraction.py 中 main() 写入 color_shape_features.csv 的
# 列顺序严格一致: 先 72 color, 再 18 shape, 再 7 ABCD。
# 训练好的 Stacking bundle 期望 [268:365] 这 97 列按此顺序输入。
# ==========================================
COLOR_NAMES = [
    f'{sp}_{ch}_{stat}'
    for sp, ch in [
        ('RGB',   'R'), ('RGB',   'G'), ('RGB',   'B'),
        ('HSV',   'H'), ('HSV',   'S'), ('HSV',   'V'),
        ('Lab',   'L'), ('Lab',   'a'), ('Lab',   'b'),
        ('YCbCr', 'Y'), ('YCbCr', 'Cb'), ('YCbCr', 'Cr'),
    ]
    for stat in ['mean', 'std', 'skew', 'kurt', 'p10', 'p90']
]

# 形状特征: 11 几何 + 7 Hu = 18
SHAPE_NAMES = [
    'area_ratio', 'perimeter', 'eccentricity', 'circularity', 'solidity',
    'aspect_ratio', 'extent', 'equivalent_diameter', 'orientation',
    'major_axis', 'minor_axis',
] + [f'hu_{i+1}' for i in range(7)]

# ABCD 特征: 7
ABCD_NAMES = [
    'ABCD_A_asymmetry', 'ABCD_B_border_CV', 'ABCD_C_n_color_clusters',
    'ABCD_C_hsv_std_sum', 'ABCD_D_diameter', 'ABCD_area', 'ABCD_perimeter',
]

CS_ABCD_NAMES = COLOR_NAMES + SHAPE_NAMES + ABCD_NAMES  # 72 + 18 + 7 = 97
assert len(CS_ABCD_NAMES) == 97, f"CS_ABCD_NAMES 长度异常: {len(CS_ABCD_NAMES)}"

TOTAL_FEATURE_DIMS = 268 + len(CS_ABCD_NAMES)  # 365


# ==========================================
# 单图 -> 365 维特征向量
# ==========================================
def _extract_one(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """对单张图 (H,W,3) + 它的 mask (H,W) 提取 365 维特征向量。"""
    tex = extract_texture(image_rgb, mask)  # (268,)

    cs_dict = extract_color_features(image_rgb, mask)
    sh_dict = extract_shape_features(mask)
    ab_dict = extract_abcd_features(image_rgb, mask)

    # 强制按 CS_ABCD_NAMES 顺序拼成 97 维向量
    merged = {**cs_dict, **sh_dict, **ab_dict}
    cs_abcd = np.array([merged.get(name, 0.0) for name in CS_ABCD_NAMES],
                       dtype=np.float64)

    return np.concatenate([tex, cs_abcd])


# ==========================================
# 内联 Stacking 推理 (10 行级，等价于 code/src/stacking_classifier.predict_stacking)
# ==========================================
def _predict_stacking(fitted_base, meta_learner, X: np.ndarray):
    """X: (n, 365)  ->  (y_pred, y_proba)"""
    n_base = len(fitted_base)
    n_samples = len(X)
    n_classes = len(fitted_base[next(iter(fitted_base))].classes_)

    base_proba = np.zeros((n_samples, n_base * n_classes))
    for i, (_, learner) in enumerate(fitted_base.items()):
        base_proba[:, i * n_classes:(i + 1) * n_classes] = learner.predict_proba(X)

    y_pred = meta_learner.predict(base_proba)
    y_proba = meta_learner.predict_proba(base_proba)
    return y_pred, y_proba


# ==========================================
# 主流程
# ==========================================
def main(image_dir: Path = None, mask_dir: Path = None, output_csv: Path = None):
    image_dir  = Path(image_dir)  if image_dir  else SUBMIT_DIR / 'image'
    mask_dir   = Path(mask_dir)   if mask_dir   else SUBMIT_DIR / 'mask'
    output_csv = Path(output_csv) if output_csv else SUBMIT_DIR / 'output.csv'

    if not image_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {image_dir}")
    bundle_path = SUBMIT_DIR / 'stacking_pipeline.pkl'
    if not bundle_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {bundle_path}")

    print(f"[submit] 加载权重: {bundle_path}")
    bundle = joblib.load(bundle_path)
    fitted_base  = bundle['fitted_base']
    meta_learner = bundle['meta_learner']
    class_names  = list(bundle['class_names'])
    print(f"  基学习器: {list(fitted_base.keys())}")
    print(f"  类别: {class_names}")

    image_paths = sorted(image_dir.glob('*.jpg'), key=lambda p: p.stem)
    if not image_paths:
        raise FileNotFoundError(f"{image_dir} 下没有 .jpg 图像")

    print(f"[submit] 待处理图像 {len(image_paths)} 张 (来源: {image_dir})")

    rows_X, rows_meta = [], []
    skipped = []
    for img_path in image_paths:
        stem = img_path.stem
        mask_path = mask_dir / f'mask_{stem}.jpg'
        if not mask_path.exists():
            print(f"  [warn] 缺 mask: {mask_path.name}, 跳过 {img_path.name}")
            skipped.append(img_path.name)
            continue

        image = np.array(Image.open(img_path).convert('RGB'))      # (H, W, 3)
        mask  = np.array(Image.open(mask_path).convert('L'))        # (H, W) 0/255

        vec = _extract_one(image, mask)
        if vec.shape[0] != TOTAL_FEATURE_DIMS:
            raise RuntimeError(
                f"特征维度异常 ({vec.shape[0]} != {TOTAL_FEATURE_DIMS}) at {img_path.name}"
            )
        rows_X.append(vec)
        rows_meta.append({'image_id': stem, 'img_path': str(img_path)})

    if not rows_X:
        raise RuntimeError("没有成功提取任何图像的特征。请检查 image/ 和 mask/ 文件。")

    X = np.stack(rows_X, axis=0)
    print(f"[submit] 特征矩阵: {X.shape} (期望 (n, {TOTAL_FEATURE_DIMS}))")

    y_pred, y_proba = _predict_stacking(fitted_base, meta_learner, X)

    # LR 元学习器在 fit 时使用字符串 label, 因此 predict() 直接返回字符串；
    # 但若以后换成数字 label, 这里也兼容 (此时用 class_names 做映射)。
    if len(y_pred) and not isinstance(y_pred[0], str):
        y_pred = [class_names[i] for i in y_pred]

    # 输出 CSV 严格对齐 data/original_data/label.csv 的格式: image_id, dx
    out_df = pd.DataFrame({
        'image_id': [m['image_id'] for m in rows_meta],
        'dx':       y_pred,
    })
    out_df.to_csv(output_csv, index=False)

    # 调试信息: 显示置信度分布, 不写入 CSV
    confidence = y_proba.max(axis=1)
    print(f"  预测置信度: mean={confidence.mean():.3f}, "
          f"min={confidence.min():.3f}, max={confidence.max():.3f}")

    print(f"\n[submit] 预测完成 {len(out_df)} 张")
    print(f"  各类别计数: {out_df['dx'].value_counts().to_dict()}")
    if skipped:
        print(f"  跳过 {len(skipped)} 张 (缺 mask): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    print(f"  输出文件: {output_csv}")
    return out_df


if __name__ == "__main__":
    main()
