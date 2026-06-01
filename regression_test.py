"""
回归测试：验证
  1. texture_features.extract() 输出能正确按 FEATURE_NAMES 命名
  2. classifier.load_features() 能正确读取并拼接 features + color_shape
  3. classifier.group_stratified_split() 保证 image_id 不跨集合
"""
import sys
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path('f:/BIPproject')
sys.path.insert(0, str(ROOT / 'code' / 'src'))

from preprocess import load_data
from texture_features import extract, FEATURE_NAMES
from classifier import load_features, group_stratified_split, META_COLS, RESULT_DIR

# --- 1) 用前 2 张图产生两个匹配的测试 CSV（不污染真实 features.csv）---
DATA_DIR = ROOT / 'data' / 'original_data'
images, masks, info = load_data(DATA_DIR)

real_features = RESULT_DIR / 'features.csv'
real_cs = RESULT_DIR / 'color_shape_features.csv'
backup_tex = RESULT_DIR / 'features.csv.bak2'
backup_cs = RESULT_DIR / 'color_shape_features.csv.bak2'

# 备份真实文件
shutil.copy(real_features, backup_tex)
shutil.copy(real_cs, backup_cs)

try:
    # 用前 2 张图构造 features_test.csv (新格式)
    rows_tex = []
    for i in range(2):
        img, mask, inf = images[i], masks[i], info[i]
        fv = extract(img, mask)
        row = {'image_id': inf['image_id'], 'is_augmented': inf['is_augmented'], 'label_dx': inf['dx']}
        for name, val in zip(FEATURE_NAMES, fv):
            row[name] = val
        rows_tex.append(row)
    test_tex = RESULT_DIR / 'features.csv'
    pd.DataFrame(rows_tex, columns=META_COLS + FEATURE_NAMES).to_csv(test_tex, index=False)
    df_test = pd.read_csv(test_tex)
    print('=== texture_features 格式验证 ===')
    print(f'  shape: {df_test.shape}  (期望 2 × 271)')
    print(f'  前 5 列: {list(df_test.columns[:5])}')
    print(f'  后 5 列: {list(df_test.columns[-5:])}')

    # 用前 2 张图构造 color_shape_test.csv（按 color_shape 的格式）
    rows_cs = []
    for i in range(2):
        img, mask, inf = images[i], masks[i], info[i]
        # 与 feature_extraction.py 中 color+shape 提取方式保持一致
        from feature_extraction import extract_color_features, extract_shape_features
        c = extract_color_features(img, mask)
        s = extract_shape_features(mask)
        row = {'image_id': inf['image_id'], 'is_augmented': inf['is_augmented'], 'label_dx': inf['dx']}
        row.update(c)
        row.update(s)
        rows_cs.append(row)
    test_cs = RESULT_DIR / 'color_shape_features.csv'
    # 列顺序与 feature_extraction.py 输出一致
    cs_cols = list(rows_cs[0].keys())
    pd.DataFrame(rows_cs, columns=cs_cols).to_csv(test_cs, index=False)
    df_cs_test = pd.read_csv(test_cs)
    print()
    print('=== color_shape_features 格式验证 ===')
    print(f'  shape: {df_cs_test.shape}  (期望 2 × 26)')
    print(f'  前 5 列: {list(df_cs_test.columns[:5])}')
    print(f'  后 5 列: {list(df_cs_test.columns[-5:])}')

    # --- 2) 验证 load_features ---
    X, y, groups, meta = load_features()
    print()
    print('=== load_features 验证 ===')
    print(f'  X.shape = {X.shape}  (期望 2 × 291 = 268 纹理 + 23 颜色形状)')
    print(f'  y = {y.tolist()}')
    print(f'  groups (image_id) = {groups.tolist()}')

    # --- 3) 验证分组分层划分 (2 张图，30% 测试) ---
    train_mask, test_mask, train_ids, test_ids = group_stratified_split(
        meta, test_size=0.3, random_state=42
    )
    print()
    print('=== group_stratified_split 验证 (2 张图) ===')
    print(f'  train_ids={sorted(train_ids)} test_ids={sorted(test_ids)}')
    print(f'  集合互不相交: {train_ids.isdisjoint(test_ids)}')
    print(f'  全 2 行被唯一分配: {train_mask.sum() + test_mask.sum() == 2}')

    # --- 4) 在伪造的元数据上验证分组约束（混合原图+增广）---
    # 构造 9 个 image_id × 3 类，每类 3 个 image_id，每个 image_id 有 1 原 + 2 增广 = 27 行
    fake_rows = []
    label_cycle = ['nv', 'nv', 'nv', 'mel', 'mel', 'mel', 'vasc', 'vasc', 'vasc']
    for iid, label in enumerate(label_cycle, start=1):
        for aug in [False, True, True]:
            fake_rows.append({'image_id': iid, 'is_augmented': aug, 'label_dx': label})
    fake_meta = pd.DataFrame(fake_rows)
    tr_m, te_m, tr_ids, te_ids = group_stratified_split(fake_meta, test_size=0.33, random_state=0)
    print()
    print('=== 分组约束验证（伪造 27 行：9 个 image_id，每类 3 个）===')
    print(f'  唯一 image_id: {sorted(fake_meta.image_id.unique())}')
    print(f'  各类 group 数: {fake_meta.drop_duplicates("image_id")["label_dx"].value_counts().to_dict()}')
    print(f'  train_ids={sorted(tr_ids)} ({len(tr_ids)} 个)')
    print(f'  test_ids ={sorted(te_ids)} ({len(te_ids)} 个)')
    all_ok = True
    for iid in range(1, 10):
        in_train = iid in tr_ids
        in_test = iid in te_ids
        rows_for_iid = fake_meta[fake_meta.image_id == iid]
        idx = rows_for_iid.index.tolist()
        # 该 image_id 的所有 3 行必须全部落在 train 或 test 中
        if in_train and not in_test:
            ok = tr_m[idx].all()
        elif in_test and not in_train:
            ok = te_m[idx].all()
        else:
            ok = False
        all_ok = all_ok and ok
        print(f'  image_id={iid} (n={len(rows_for_iid)}): train={in_train} test={in_test} all-same-side={ok}')
    # 验证整体 train + test = 27
    all_assigned = tr_m.sum() + te_m.sum() == len(fake_meta)
    all_ok = all_ok and all_assigned
    print(f'  全部 27 行都被唯一分配: {all_assigned}')

    print()
    print('全部验证通过' if all_ok else '验证失败')
finally:
    shutil.copy(backup_tex, real_features)
    shutil.copy(backup_cs, real_cs)
    backup_tex.unlink()
    backup_cs.unlink()
    print('已恢复真实 features.csv / color_shape_features.csv')
