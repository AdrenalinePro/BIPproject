"""
回归测试：用合成数据（9 个 image_id × 3 类）验证 stacking_classifier.py
端到端跑通：分支工厂 -> fit_stacking_with_groups -> predict_stacking -> save_pipeline -> load

不会触碰真实 features.csv / color_shape_features.csv。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path('f:/BIPproject')
sys.path.insert(0, str(ROOT / 'code' / 'src'))

from stacking_classifier import (
    make_branch_rf, make_branch_svm, make_branch_knn, make_branch_gb, make_branch_lr,
    make_base_learners, make_meta_learner,
    fit_stacking_with_groups, predict_stacking, save_pipeline,
    FEATURE_GROUPS, TOTAL_FEATURE_DIMS, RESULT_DIR,
)

# --- 1) 验证 5 个分支工厂 ---
print('=== 5 个分支工厂 ===')
for name, factory in [
    ('rf',  make_branch_rf),
    ('svm', make_branch_svm),
    ('knn', make_branch_knn),
    ('gb',  make_branch_gb),
    ('lr',  make_branch_lr),
]:
    p = factory()
    print(f'  {name}: {type(p).__name__}, steps={[s[0] for s in p.steps]}')
    assert isinstance(p, type(make_branch_rf())), f'{name} 不是 Pipeline'
assert list(FEATURE_GROUPS.keys()) == ['rf', 'svm', 'knn', 'gb', 'lr']
print(f'  FEATURE_GROUPS OK, TOTAL_FEATURE_DIMS={TOTAL_FEATURE_DIMS}')

# --- 2) 构造合成数据 ---
# 9 个 image_id, 每类 3 个, 每个 image_id 3 行（1 原图 + 2 增广）= 27 行 × 365 维
print('\n=== 构造合成数据 ===')
n_classes, n_per_class, n_per_image = 3, 3, 3
total = n_classes * n_per_class * n_per_image
np.random.seed(42)
X = np.random.randn(total, TOTAL_FEATURE_DIMS)
y = np.repeat(['mel', 'nv', 'vasc'], n_per_class * n_per_image)
groups = np.repeat(np.arange(1, 10), n_per_image)  # 9 个 unique image_id
print(f'  X: {X.shape}, y: {y.shape}, groups: {groups.shape}')
print(f'  唯一 image_id: {len(np.unique(groups))} (期望 9)')
print(f'  各类样本数: {pd.Series(y).value_counts().to_dict()}')

# --- 3) 训练 (n_splits=3 for 9 image_ids) ---
print('\n=== fit_stacking_with_groups (n_splits=3) ===')
fitted_base, meta_learner, oof = fit_stacking_with_groups(X, y, groups, n_splits=3)
print(f'  fitted_base: {list(fitted_base.keys())}')
print(f'  oof shape: {oof.shape} (期望 ({total}, {5*3}))')
print(f'  meta_learner classes: {list(meta_learner.classes_)}')

# --- 4) 预测 ---
y_pred, y_proba = predict_stacking(fitted_base, meta_learner, X)
print(f'\n=== predict_stacking ===')
print(f'  y_pred[:5] = {y_pred[:5].tolist()}')
print(f'  y_proba shape = {y_proba.shape} (期望 ({total}, 3))')
print(f'  y_proba[0] = {y_proba[0].round(3).tolist()}')

# --- 5) 保存与加载 ---
print('\n=== save_pipeline / load 验证 ===')
out_path = save_pipeline(fitted_base, meta_learner)
print(f'  保存到: {out_path}')
print(f'  存在: {out_path.exists()}, 大小: {out_path.stat().st_size} bytes')

bundle = joblib.load(out_path)
expected_keys = {'fitted_base', 'meta_learner', 'feature_groups', 'total_feature_dims',
                 'meta_cols', 'base_learner_names', 'class_names'}
assert set(bundle.keys()) == expected_keys, f'bundle keys 不匹配: {set(bundle.keys())}'
print(f'  bundle keys 完整: {sorted(bundle.keys())}')
print(f'  class_names: {bundle["class_names"]}')
print(f'  base_learner_names: {bundle["base_learner_names"]}')

# 用 bundle 里的模型再次预测，验证一致
y_pred2, _ = predict_stacking(bundle['fitted_base'], bundle['meta_learner'], X)
assert np.array_equal(y_pred, y_pred2), 'load 后预测结果不一致'
print('  [OK] load 后预测结果与训练时一致')

# 清理
out_path.unlink()
print('\n=== 全部验证通过 [OK] ===')
