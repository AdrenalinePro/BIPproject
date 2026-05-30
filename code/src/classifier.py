"""
分类器训练
    从 result/features.csv 读取预先提取的纹理特征和标签
    特征标准化  scaler_texture.pkl 
    训练分类器-随机森林、SVM、KNN、梯度提升、逻辑回归
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

# 纹理特征
def load_features():
    csv_path = RESULT_DIR / "features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到纹理特征文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def main():

    X, y = load_features()

    # 划分训练/测试集（训练70%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    
    # 特征归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    (RESULT_DIR / "common").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "classifiers").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, RESULT_DIR/"common" / "scaler_texture.pkl")

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
        clf_dir = RESULT_DIR/"classifiers" / name
        clf_dir.mkdir(exist_ok=True)   
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
   
        # 保存混淆矩阵图
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        plt.figure(figsize=(6,5))
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
            imp_df.to_csv(RESULT_DIR/"common" / "feature_importance_GradientBoosting.csv", index=False)

            n_display = min(30, len(importances))
            indices = np.argsort(importances)[::-1][:n_display]
            plt.figure(figsize=(10,6))
            plt.barh(range(n_display), importances[indices][::-1])
            plt.yticks(range(n_display), [f"Feat_{i}" for i in indices][::-1])
            plt.xlabel("Importance")
            plt.title("Top 30 Feature Importances - GradientBoosting")
            plt.tight_layout()
            plt.savefig(RESULT_DIR/"common" / "feature_importance_GradientBoosting.png", dpi=150)
            plt.close()

if __name__ == "__main__":
    main()