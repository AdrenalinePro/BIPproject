"""
分类器训练
    从 result/features.csv (纹理) 和 result/color_shape_features.csv (颜色+形状) 读取
    拼接两组特征，划分训练/测试集（按 image_id 分组，原图与增强图必须落在同一集合）
    训练分类器 - 随机森林、SVM、KNN、梯度提升、逻辑回归
    为每个分类器生成
        混淆矩阵图   confusion_matrix.png
        分类报告   classification_report.txt
        训练好的模型文件   model.pkl
    对当前准确率最高的分类器生成
        特征重要性图-前30个特征 feature_importance_GradientBoosting.png
        CSV数据                feature_importance_GradientBoosting.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

project_root = Path(__file__).parent.parent.parent
RESULT_DIR = project_root / "result"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

META_COLS = ['image_id', 'is_augmented', 'label_dx']


def load_features():
    """
    读取纹理特征和颜色/形状特征并按行拼接。

    两个 CSV 的前 3 列均为元数据 (image_id, is_augmented, label_dx)，
    且由 load_data 的确定性顺序保证行序一致。读取时校验元数据匹配，
    拼接策略：保留一份元数据 + 两份特征列。

    返回:
        X:        np.ndarray, 特征矩阵
        y:        np.ndarray, 标签
        groups:   np.ndarray, 分组键 (image_id)，用于分组划分
        meta_df:  pd.DataFrame, 元数据子表（包含 image_id, is_augmented, label_dx）
    """
    texture_path = RESULT_DIR / "features.csv"
    color_shape_path = RESULT_DIR / "color_shape_features.csv"

    for p in [texture_path, color_shape_path]:
        if not p.exists():
            raise FileNotFoundError(f"未找到特征文件: {p}")

    df_tex = pd.read_csv(texture_path)
    df_cs = pd.read_csv(color_shape_path)

    # 校验两个文件的元数据列名都存在
    for name, df in [('features.csv', df_tex), ('color_shape_features.csv', df_cs)]:
        missing = [c for c in META_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 缺少元数据列: {missing}")

    # 校验行数一致 + 元数据按行对齐（两文件均来自 load_data 的确定性顺序）
    if len(df_tex) != len(df_cs):
        raise ValueError(
            f"两个 CSV 行数不一致: features={len(df_tex)}, color_shape={len(df_cs)}"
        )
    if not df_tex[META_COLS].equals(df_cs[META_COLS]):
        raise ValueError("两个 CSV 的元数据列(image_id/is_augmented/label_dx)按行不一致")

    tex_feat_cols = [c for c in df_tex.columns if c not in META_COLS]
    cs_feat_cols = [c for c in df_cs.columns if c not in META_COLS]

    X_df = pd.concat(
        [df_tex[tex_feat_cols].reset_index(drop=True),
         df_cs[cs_feat_cols].reset_index(drop=True)],
        axis=1,
    )
    meta_df = df_cs[META_COLS].reset_index(drop=True)
    X = X_df.values
    y = meta_df['label_dx'].values
    groups = meta_df['image_id'].values
    return X, y, groups, meta_df


def group_stratified_split(meta_df, test_size=0.2, random_state=42):
    """
    分组分层划分：同一 image_id 的所有行（原图+增强）必须落在同一集合。
    分层使用各组的"主标签"（优先取非增强的行的标签；缺省时退回首个出现的标签）。

    返回:
        train_mask, test_mask: 与 meta_df 等长的布尔数组
        train_ids, test_ids:   划入训练/测试的 image_id 集合
    """
    # 主标签：每个 image_id 优先选 is_augmented=False 的行的 label_dx
    non_aug = meta_df[~meta_df['is_augmented']]
    primary = non_aug.drop_duplicates('image_id').set_index('image_id')['label_dx']
    fallback = meta_df.groupby('image_id')['label_dx'].first()
    group_label = primary.combine_first(fallback)

    unique_ids = np.array(sorted(meta_df['image_id'].unique()))
    group_labels = group_label.reindex(unique_ids).values

    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels,
    )

    train_mask = meta_df['image_id'].isin(train_ids).values
    test_mask = meta_df['image_id'].isin(test_ids).values
    return train_mask, test_mask, set(train_ids.tolist()), set(test_ids.tolist())


def main():
    X, y, groups, meta_df = load_features()
    print(f"加载特征矩阵: {X.shape} (含纹理 + 颜色 + 形状)")
    print(f"  标签分布: {pd.Series(y).value_counts().to_dict()}")
    print(f"  唯一 image_id: {len(np.unique(groups))}")

    # 分组分层划分
    train_mask, test_mask, train_ids, test_ids = group_stratified_split(
        meta_df, test_size=0.2, random_state=RANDOM_SEED
    )
    assert train_ids.isdisjoint(test_ids), "image_id 出现泄漏：训练集和测试集有交集"

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"训练集: {X_train.shape[0]} 样本, {len(train_ids)} 唯一 image_id, 标签分布 {pd.Series(y_train).value_counts().to_dict()}")
    print(f"测试集: {X_test.shape[0]} 样本, {len(test_ids)} 唯一 image_id, 标签分布 {pd.Series(y_test).value_counts().to_dict()}")
    print("✓ 验证：image_id 在训练集和测试集无重叠（增强图与原图同侧）")

    # 特征归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    (RESULT_DIR / "common").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "classifiers").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, RESULT_DIR / "common" / "scaler_texture.pkl")

    # 分类器
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    }

    results = []

    for name, clf in classifiers.items():
        clf_dir = RESULT_DIR / "classifiers" / name
        clf_dir.mkdir(exist_ok=True)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        # 保存混淆矩阵图
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=clf.classes_, yticklabels=clf.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(clf_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # 保存分类报告文本
        with open(clf_dir / "classification_report.txt", "w") as f:
            f.write(f"Classifier: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(y_test, y_pred))

        # 保存模型
        joblib.dump(clf, clf_dir / "model.pkl")

        # 记录结果
        results.append({"Classifier": name, "Accuracy": acc})

        # 准确率最高的分类器（当前GradientBoosting）保存特征重要性
        if name == "GradientBoosting":
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({"feature_index": np.arange(len(importances)), "importance": importances})
            imp_df.to_csv(RESULT_DIR / "common" / "feature_importance_GradientBoosting.csv", index=False)

            n_display = min(30, len(importances))
            indices = np.argsort(importances)[::-1][:n_display]
            plt.figure(figsize=(10, 6))
            plt.barh(range(n_display), importances[indices][::-1])
            plt.yticks(range(n_display), [f"Feat_{i}" for i in indices][::-1])
            plt.xlabel("Importance")
            plt.title("Top 30 Feature Importances - GradientBoosting")
            plt.tight_layout()
            plt.savefig(RESULT_DIR / "common" / "feature_importance_GradientBoosting.png", dpi=150)
            plt.close()

if __name__ == "__main__":
    main()
