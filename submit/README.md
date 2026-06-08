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
### 3) 运行
```bash
python submit/main.py
```


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
