# preprocess.py 函数说明

本模块所有函数均针对**二维图像**处理。

## load_data(data_dir)

将所有图像（包括原始和增强图像）及其对应的mask加载到内存，并记录每张图像的信息。

**参数：**
- `data_dir`: Path，指向 `original_data` 文件夹的路径

**返回：**
- `images`: np.ndarray，形状 `(N, H, W, C)`，所有二维RGB图像
- `masks`: np.ndarray，形状 `(N, H, W)`，所有二维mask
- `info`: list of dict，每元素包含 `image_id`、`dx`（病变类型）、`is_augmented`（是否为增强图像）

**示例：**
```python
images, masks, info = load_data(data_dir)
print(info[0])  # {'image_id': 1, 'dx': 'nv', 'is_augmented': False}
```

---

## apply_mask(image, mask)

将图像与mask相乘，只保留mask覆盖区域的内容。支持RGB图像和灰度图。

**参数：**
- `image`: np.ndarray，形状 `(H, W, C)` 的RGB图像或 `(H, W)` 的灰度图
- `mask`: np.ndarray，形状 `(H, W)`，值为0-255的灰度图

**返回：**
- np.ndarray，形状与输入image一致，与mask相乘后的图像

**示例：**
```python
masked_img = apply_mask(images[0], masks[0])
gray_masked = apply_mask(gray_img, masks[0])
```

---

## rgb_to_gray(image)

将RGB图像转换为灰度图。

**参数：**
- `image`: np.ndarray，形状 `(H, W, C)` 的RGB图像

**返回：**
- np.ndarray，形状 `(H, W)` 的灰度图像

**示例：**
```python
gray_img = rgb_to_gray(images[0])
```

---

## remove_hair(image)

去除图像中的毛发，采用Blackhat形态学和inpaint技术。

**参数：**
- `image`: np.ndarray，形状 `(H, W, C)` 的RGB图像

**返回：**
- `result`: np.ndarray，去毛发后的图像
- `hair_mask`: np.ndarray，检测到的毛发区域mask

**示例：**
```python
clean_img, hair_mask = remove_hair(images[0])
```

---

# stacking_classifier.py — Stacking 融合分类器

## 概述

5 个基学习器（RF / SVM / KNN / GB / LR）在**不同特征子集**上做"分支降维"，
1 个元学习器（LR）在 OOF 概率上做最终融合。整体不使用任何深度学习方法，
分类器限定在 `{RF, SVM, KNN, GB, LR}` 这 5 类中。

## 5 个分支（基学习器）

| Branch | 分类器             | 特征子集                    | 维度 | 降维方式                                      |
|--------|-------------------|-----------------------------|------|-----------------------------------------------|
| `rf`   | RandomForest      | 纹理 (LBP 256 + GLCM 12)    | 268  | `SelectKBest(f_classif, k=120)`              |
| `svm`  | SVC(rbf)          | 颜色 (RGB+HSV+Lab+YCbCr)    | 72   | `StandardScaler` + `SelectKBest(f_classif, k=40)` |
| `knn`  | KNN               | 形状 (18) + ABCD (7)        | 25   | `StandardScaler`（25 维无需选择）              |
| `gb`   | GradientBoosting  | 全部                        | 365  | `SelectFromModel(GB, threshold='median')`     |
| `lr`   | LogisticRegression| 全部                        | 365  | `StandardScaler` + L1 penalty（内置稀疏）      |

所有分支都接 `class_weight='balanced'` 处理类别不平衡。

## 元学习器

`LogisticRegression(C=1.0, class_weight='balanced')`，在 `5 × 3 = 15` 维 OOF 概率上训练。
经典 stacking 配方（Wolpert 1992），小数据上不容易过拟合。

## 训练流程

**手动 Stacking**（不是 `sklearn.ensemble.StackingClassifier`）：
原因是该类不接受 `groups` 参数，无法保证同一 `image_id` 的所有行（原图+增广）落在同侧。
本文件用 `StratifiedGroupKFold(n_splits=5)` 手动实现。

1. `StratifiedGroupKFold(5)` 按 `image_id` 切分训练集
2. 对每个基学习器：5 折 OOF `predict_proba` → `(n_train, n_classes)` 矩阵
3. 在完整训练集上重训每个基学习器（用于最终推理）
4. 把 5 个 OOF 矩阵横向拼成 `(n_train, 15)`，训练元学习器

## 主要函数

| 函数 | 说明 |
|------|------|
| `make_branch_rf()` / `make_branch_svm()` / `make_branch_knn()` / `make_branch_gb()` / `make_branch_lr()` | 5 个分支的工厂函数，返回 `Pipeline` |
| `make_base_learners()` | 返回 `{name: Pipeline}` 字典（5 个基学习器）|
| `make_meta_learner()` | 元学习器 LR |
| `fit_stacking_with_groups(X_train, y_train, groups_train, n_splits=5)` | 训练 Stacking，返回 `(fitted_base, meta_learner, oof)` |
| `predict_stacking(fitted_base, meta_learner, X)` | 推理，返回 `(y_pred, y_proba)` |
| `save_pipeline(fitted_base, meta_learner)` | 保存到 `result/stacking_pipeline.pkl` |

## 特征列布局

`load_features()` 输出矩阵的列布局（与 `FEATURE_GROUPS` 严格对应）：

```
[   0 : 268 ]  = 纹理 (LBP 256 + GLCM 12)
[ 268 : 340 ]  = 颜色 (4 空间 × 3 通道 × 6 统计) = 72
[ 340 : 358 ]  = 形状 (regionprops 5 + 新增 6 + Hu 矩 7) = 18
[ 358 : 365 ]  = ABCD (A 1 + B 1 + C 2 + D 1 + 2 辅助) = 7
总计 365 维
```

## 运行

```bash
python code/src/stacking_classifier.py
```

前置文件：
- `result/features.csv`（纹理，271 列 = 3 meta + 268 features）
- `result/color_shape_features.csv`（颜色+形状+ABCD，100 列 = 3 meta + 97 features）

## 输出

- `result/stacking_pipeline.pkl` — 整个 stacking bundle
  - `fitted_base` 训练好的 5 个基学习器
  - `meta_learner` 训练好的元学习器
  - `feature_groups`, `total_feature_dims`, `meta_cols` 等文档字段
- `result/stacking_confusion_matrix.png` — 测试集混淆矩阵

> 推理脚本 `infer.py`（加载 `stacking_pipeline.pkl` 做预测）待后续写。
