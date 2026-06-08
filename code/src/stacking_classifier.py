"""
Stacking 融合分类器
    5 个基学习器（RF / SVM / KNN / GB / LR）在不同特征子集上做"分支降维"
    1 个元学习器（LR）在 OOF 概率上做最终融合
    训练完成后保存到 result/stacking_pipeline.pkl

约束：
  - 不使用深度学习方法
  - 5 个分类器固定为 {RF, SVM, KNN, GB, LR}
  - 降维发生在 Stacking 内部（Pipeline 的 select / scaler / L1 penalty 步骤）

特征列布局（与 classifier.load_features 输出顺序一致）：
  [   0 : 268 ]  = 纹理 (LBP 256 + GLCM 12)
  [ 268 : 340 ]  = 颜色 (RGB+HSV+Lab+YCbCr, 4 空间 × 3 通道 × 6 统计)
  [ 340 : 358 ]  = 形状 (regionprops 5 + 新增 6 + Hu 矩 7)
  [ 358 : 365 ]  = ABCD (A 1 + B 1 + C 2 + D 1 + 2 辅助)
  总计 365 维
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 路径 & 共享配置
# ==========================================
project_root = Path(__file__).parent.parent.parent
RESULT_DIR = project_root / "result"
sys.path.insert(0, str(project_root / "code" / "src"))
from classifier import load_features, group_stratified_split, META_COLS, RANDOM_SEED


# ==========================================
# 特征列分组（与 load_features 输出顺序严格一致）
# ==========================================
FEATURE_GROUPS: Dict[str, list] = {
    'rf':  list(range(0, 268)),       # 纹理 (LBP 256 + GLCM 12)
    'svm': list(range(268, 340)),     # 颜色 (72)
    'knn': list(range(340, 365)),     # 形状(18) + ABCD(7) = 25
    'gb':  list(range(0, 365)),       # 全部
    'lr':  list(range(0, 365)),       # 全部
}
TOTAL_FEATURE_DIMS = 365


# ==========================================
# 1. 5 个分支（基学习器工厂）
# ==========================================
def make_branch_rf() -> Pipeline:
    """Branch 1: RandomForest 在纹理子集 (LBP+GLCM) 上。
    降维：SelectKBest(f_classif, k=120)。
    """
    return Pipeline([
        ('select', SelectKBest(score_func=f_classif, k=120)),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


def make_branch_svm() -> Pipeline:
    """Branch 2: SVC(rbf) 在颜色子集上。
    降维：StandardScaler + SelectKBest(f_classif, k=40)。
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=40)),
        ('clf', SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=RANDOM_SEED,
        )),
    ])


def make_branch_knn() -> Pipeline:
    """Branch 3: KNN 在形状 + ABCD 子集上。
    降维：仅 StandardScaler（25 维已足够小，无需选择）。
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski')),
    ])


def make_branch_gb() -> Pipeline:
    """Branch 4: GradientBoosting 在全特征上。
    降维：SelectFromModel(GB, threshold='median') — 树 importance 嵌入式选择。
    """
    return Pipeline([
        ('select', SelectFromModel(
            estimator=GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=3, random_state=RANDOM_SEED,
            ),
            threshold='median',
        )),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED,
        )),
    ])


def make_branch_lr() -> Pipeline:
    """Branch 5: LogisticRegression(L1) 在全特征上。
    降维：StandardScaler + L1 penalty（内置稀疏化）。
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty='l1', solver='saga', C=0.3,
            class_weight='balanced', max_iter=5000, random_state=RANDOM_SEED,
        )),
    ])


def make_base_learners() -> Dict[str, Pipeline]:
    """5 个基学习器，按固定顺序返回 (rf / svm / knn / gb / lr)。"""
    return {
        'rf':  make_branch_rf(),
        'svm': make_branch_svm(),
        'knn': make_branch_knn(),
        'gb':  make_branch_gb(),
        'lr':  make_branch_lr(),
    }


def make_meta_learner() -> LogisticRegression:
    """元学习器：在 5×3=15 维 OOF 概率上训练的 LogisticRegression。
    经典 stacking 配方（Wolpert 1992）；class_weight='balanced' 处理不平衡。
    """
    return LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=2000, random_state=RANDOM_SEED,
    )


# ==========================================
# 2. 手动 Stacking（5 折 StratifiedGroupKFold）
#    原因：sklearn.ensemble.StackingClassifier.fit 不支持 groups 参数，
#    用手动实现保证同 image_id 的所有行（原图+增强）落在同一折。
# ==========================================
def fit_stacking_with_groups(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int = 5,
) -> Tuple[Dict[str, Pipeline], LogisticRegression, np.ndarray]:
    """
    手动 Stacking 训练流程。

    步骤：
      1) StratifiedGroupKFold(n_splits) 划分训练集
      2) 对每个基学习器做 OOF 预测（predict_proba），得到 (n_train, n_classes) 矩阵
      3) 在完整训练集上重训每个基学习器（用于最终推理）
      4) 把 5 个基学习器的 OOF 概率横向拼成 (n_train, 5×n_classes) 喂给元学习器
      5) 训练元学习器

    返回:
        fitted_base:     训练好的基学习器 dict {name: Pipeline}
        meta_learner:    训练好的元学习器
        oof_predictions: OOF 概率矩阵 shape=(n_train, n_base × n_classes)
    """
    base_learners = make_base_learners()
    meta_learner = make_meta_learner()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    n_train = len(X_train)
    n_base = len(base_learners)
    n_classes = len(np.unique(y_train))

    oof_predictions = np.zeros((n_train, n_base * n_classes))
    fitted_base: Dict[str, Pipeline] = {}

    for i, (name, learner) in enumerate(base_learners.items()):
        print(f"  [OOF] base learner {i+1}/{n_base}: {name}")
        oof = np.zeros((n_train, n_classes))

        for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y_train, groups_train)):
            learner_clone = clone(learner)
            learner_clone.fit(X_train[tr_idx], y_train[tr_idx])
            oof[va_idx] = learner_clone.predict_proba(X_train[va_idx])

        oof_predictions[:, i * n_classes:(i + 1) * n_classes] = oof

        # 在完整训练集上重训（用于最终推理时的基学习器）
        final_learner = clone(learner)
        final_learner.fit(X_train, y_train)
        fitted_base[name] = final_learner

    # 训练元学习器
    meta_learner.fit(oof_predictions, y_train)
    print(f"  [META] OOF shape: {oof_predictions.shape}, classes: {list(meta_learner.classes_)}")

    return fitted_base, meta_learner, oof_predictions


def predict_stacking(
    fitted_base: Dict[str, Pipeline],
    meta_learner: LogisticRegression,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    推理：每个基学习器先 predict_proba，再把 5×n_classes 维概率拼起来喂给元学习器。
    返回 (y_pred, y_proba)。
    """
    n_base = len(fitted_base)
    n_samples = len(X)
    n_classes = len(fitted_base[list(fitted_base.keys())[0]].classes_)

    base_proba = np.zeros((n_samples, n_base * n_classes))
    for i, (name, learner) in enumerate(fitted_base.items()):
        base_proba[:, i * n_classes:(i + 1) * n_classes] = learner.predict_proba(X)

    y_pred = meta_learner.predict(base_proba)
    y_proba = meta_learner.predict_proba(base_proba)
    return y_pred, y_proba


# ==========================================
# 3. 保存权重
# ==========================================
def save_pipeline(
    fitted_base: Dict[str, Pipeline],
    meta_learner: LogisticRegression,
    save_dir: Path = RESULT_DIR,
    filename: str = 'stacking_pipeline.pkl',
) -> Path:
    """
    把训练好的 stacking pipeline 整体保存到 result/<filename>.pkl

    bundle 包含：
        - fitted_base:        训练好的 5 个基学习器（Pipeline）
        - meta_learner:       训练好的元学习器（LogisticRegression）
        - feature_groups:     各分支对应的特征列索引（仅文档用）
        - total_feature_dims: 365
        - meta_cols:          ['image_id', 'is_augmented', 'label_dx']
        - base_learner_names: 基学习器名称列表
        - class_names:        元学习器训练时见过的类别

    推理时的 load / infer.py 待后续写。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        'fitted_base': fitted_base,
        'meta_learner': meta_learner,
        'feature_groups': FEATURE_GROUPS,
        'total_feature_dims': TOTAL_FEATURE_DIMS,
        'meta_cols': META_COLS,
        'base_learner_names': list(fitted_base.keys()),
        'class_names': list(meta_learner.classes_),
    }
    out_path = save_dir / filename
    joblib.dump(bundle, out_path)
    return out_path


# ==========================================
# 4. 主程序
# ==========================================
def main(holdout: bool = False):
    """
    主入口。

    参数:
        holdout: 若 True, 80/20 划分后训练, 并在 holdout 上评估 (调试/开发用)。
                 若 False (默认), 在**全量数据**上训练, 只保存权重到 result/。
                 全量训练用于外部评测场景 —— 既然测试集与训练集完全独立,
                 应当把所有标注样本都用上以获得最强的最终模型。
    """
    # 1. 加载特征（要求 features.csv 和 color_shape_features.csv 都已是新格式）
    X, y, groups, meta_df = load_features()
    print(f"加载特征矩阵: {X.shape} (纹理 268 + 颜色 72 + 形状 18 + ABCD 7)")
    print(f"唯一 image_id: {len(np.unique(groups))}, 标签分布: {pd.Series(y).value_counts().to_dict()}")

    n_classes = len(np.unique(y))

    if holdout:
        # ==========================================
        # 调试模式: 80/20 划分, 在 holdout 上评估
        # ==========================================
        train_mask, test_mask, train_ids, test_ids = group_stratified_split(
            meta_df, test_size=0.2, random_state=RANDOM_SEED
        )
        assert train_ids.isdisjoint(test_ids), "image_id 出现泄漏"

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]

        print(f"\n[holdout] 训练集: {X_train.shape[0]} 样本, {len(train_ids)} 唯一 image_id, "
              f"标签分布 {pd.Series(y_train).value_counts().to_dict()}")
        print(f"[holdout] 测试集: {X_test.shape[0]} 样本, {len(test_ids)} 唯一 image_id, "
              f"标签分布 {pd.Series(y_test).value_counts().to_dict()}")

        fitted_base, meta_learner, oof = fit_stacking_with_groups(
            X_train, y_train, groups_train, n_splits=5
        )

        y_pred, y_proba = predict_stacking(fitted_base, meta_learner, X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n[holdout] Stacking Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        classes = meta_learner.classes_
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f"Stacking Confusion Matrix (acc={acc:.3f})")
        plt.tight_layout()
        cm_path = RESULT_DIR / 'stacking_confusion_matrix.png'
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[holdout] 混淆矩阵: {cm_path}")
    else:
        # ==========================================
        # 生产模式: 在**全量数据**上训练, 只保存权重
        # ==========================================
        # fit_stacking_with_groups 内部仍做 5 折 StratifiedGroupKFold OOF,
        # 这样元学习器 LR 训练时拿到的是非泄漏的 OOF 概率。
        # 最终每个基学习器在完整 X 上重训一次, 用于推理。
        print(f"\n[full-data] 在全量 {X.shape[0]} 样本 (image_id={len(np.unique(groups))}) 上训练 Stacking ...")
        fitted_base, meta_learner, oof = fit_stacking_with_groups(
            X, y, groups, n_splits=5
        )

        # 报告每个基学习器的 5 折 OOF 准确率 (诊断参考)
        # OOF 列是按 learner.classes_ 顺序排列的, argmax 给出局部索引,
        # 需用 classes_ 反查回真实标签, 再与 y 比较。
        print(f"\n[full-data] 各基学习器 OOF (5-fold) 准确率:")
        for i, (name, fitted_learner) in enumerate(fitted_base.items()):
            base_oof = oof[:, i * n_classes:(i + 1) * n_classes]
            classes = fitted_learner.classes_
            base_pred = classes[np.argmax(base_oof, axis=1)]
            base_acc = accuracy_score(y, base_pred)
            print(f"  - {name:5s}: {base_acc:.4f}")

    # 6. 保存权重 (无论哪种模式都做)
    out_path = save_pipeline(fitted_base, meta_learner)
    print(f"\n权重已保存到: {out_path}")
    print(f"  bundle keys: {list(joblib.load(out_path).keys())}")
    print(f"  class_names: {list(meta_learner.classes_)}")

    return fitted_base, meta_learner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 Stacking 融合分类器")
    parser.add_argument(
        '--holdout', action='store_true',
        help='使用 80/20 划分并评估 (默认: 在全量数据上训练, 只保存权重)',
    )
    args = parser.parse_args()
    main(holdout=args.holdout)
