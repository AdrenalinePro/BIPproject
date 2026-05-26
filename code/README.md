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