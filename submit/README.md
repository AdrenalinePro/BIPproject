# submit/ — 推理提交包

一个**自包含**的皮肤镜图像三分类 (mel / nv / vasc) 推理脚本。运行前只需要安装几个常用 Python 包，**不需要任何训练数据**。

## 目录结构

```
submit/
├── main.py                # 推理入口 (唯一需要运行的文件)
├── preprocess.py          # apply_mask / rgb_to_gray
├── texture.py             # imageLoader 类 (LBP/GLCM 内部实现)
├── texture_features.py    # 纹理特征 (LBP 256 + GLCM 12 = 268 维)
├── features_cs.py         # 颜色 (72) + 形状 (18) + ABCD (7) = 97 维
├── stacking_pipeline.pkl  # 训练好的 Stacking 融合模型权重 (5.5 MB)
├── README.md              # 本文件
├── image/                 # ← 把待测试图像 (.jpg) 放这里
└── mask/                  # ← 把待测试图像的 mask 放这里 (命名: mask_<stem>.jpg)
```

## 用法

### 1) 准备数据
把待测试的 RGB 图像 (`.jpg`) 放到 `submit/image/` 下，把对应的 mask (灰度图，`.jpg`) 放到 `submit/mask/` 下。**mask 的文件名约定是 `mask_<图像 stem>.jpg`**，例如：

| image/1.jpg       | → mask/mask_1.jpg         |
| image/1_aug1.jpg  | → mask/mask_1_aug1.jpg    |
| image/100_aug2.jpg| → mask/mask_100_aug2.jpg  |

> 训练时使用的数据集采用同一命名约定。

### 2) 安装依赖
```bash
pip install numpy pandas scikit-learn scikit-image opencv-python Pillow joblib scipy
```
（`cv2` = `opencv-python`）

### 3) 运行
```bash
python submit/main.py
```

程序会：
1. 按文件名字典序遍历 `submit/image/*.jpg`
2. 对每张图到 `submit/mask/mask_<stem>.jpg` 找 mask
3. 提取 365 维特征向量 (纹理 268 + 颜色 72 + 形状 18 + ABCD 7)
4. 喂入 Stacking 模型，输出每类的预测概率
5. 写出 `submit/output.csv`，格式与 `data/original_data/label.csv` 严格一致：

```
image_id,dx
1,nv
1_aug1,nv
1_aug2,nv
...
```

`image_id` 保留**完整文件名 stem**（即 `"1"`, `"1_aug1"`, `"100_aug2"` 各自一行），`dx` 是预测类别 (`mel` / `nv` / `vasc`)。

## 特征列布局 (固定, 与训练时一致)

| 区间         | 维度 | 含义                                 |
|--------------|------|--------------------------------------|
| `[0:268]`    | 268  | 纹理: 256 LBP + 12 GLCM              |
| `[268:340]`  | 72   | 颜色: RGB/HSV/Lab/YCbCr × 3 通道 × 6 统计 |
| `[340:358]`  | 18   | 形状: 11 regionprops/几何 + 7 Hu 矩  |
| `[358:365]`  | 7    | ABCD: A/B/C_cluster/C_hsv_std/D + area/perimeter |
| **总计**     | 365  |                                      |

## 模型 (Stacking 融合)

5 个分支 (基学习器) + 1 个元学习器 (LR)，所有模型在 `stacking_pipeline.pkl` 内：

| 分支 | 分类器              | 特征子集     |
|------|--------------------|--------------|
| `rf`  | RandomForest       | 纹理 (LBP+GLCM) |
| `svm` | SVC(rbf)           | 颜色          |
| `knn` | KNN                | 形状 + ABCD  |
| `gb`  | GradientBoosting   | 全部          |
| `lr`  | LogisticRegression(L1) | 全部     |
| 元   | LogisticRegression | 5×3 = 15 维 OOF 概率 |

**不使用任何深度学习方法**。所有分类器限定在 `{RF, SVM, KNN, GB, LR}`。

## 自包含性

`submit/` 整个目录可以独立拷贝到其他机器上运行，**不依赖** `code/`、`data/`、`result/` 任何训练相关路径或文件。

## 自定义输入/输出位置

`main()` 函数接受三个可选参数：

```python
main(image_dir=Path('path/to/images'),
     mask_dir=Path('path/to/masks'),
     output_csv=Path('predictions.csv'))
```

例如：
```bash
python -c "import sys; sys.path.insert(0, 'submit'); from main import main; main(image_dir=__import__('pathlib').Path('data/original_data/image'), mask_dir=__import__('pathlib').Path('data/original_data/mask'), output_csv=__import__('pathlib').Path('submit/output.csv'))"
```

## 输出示例

运行结束后屏幕打印：

```
[submit] 加载权重: submit/stacking_pipeline.pkl
  基学习器: ['rf', 'svm', 'knn', 'gb', 'lr']
  类别: ['mel', 'nv', 'vasc']
[submit] 待处理图像 600 张 (来源: submit/image)
[submit] 特征矩阵: (600, 365) (期望 (n, 365))
  预测置信度: mean=0.842, min=0.331, max=0.998
[submit] 预测完成 600 张
  各类别计数: {'nv': 311, 'mel': 174, 'vasc': 115}
  输出文件: submit/output.csv
```

`submit/output.csv` 内容：

```csv
image_id,dx
1,nv
1_aug1,nv
1_aug2,nv
2,nv
...
```
