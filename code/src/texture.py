import numpy as np
from PIL import Image
from preprocess import *


class imageLoader:
    def __init__(self, source, mask=None):
        """
        初始化imageLoader实例，根据输入类型加载图像数据。

        参数:
            source: str 或 np.ndarray, 图像文件路径或numpy数组形式的图像数据
            mask:   可选 np.ndarray, 形状为 (H, W) 的 0/255 灰度 mask；
                    >0 的位置表示 ROI。提供时，LBP/GLCM 统计将只覆盖 mask 区域内的像素。
        """
        self.image_data = None
        if isinstance(source, str):
            self._load_from_file(source)
        elif isinstance(source, np.ndarray):
            self._load_from_array(source)
        else:
            raise TypeError("source must be a file path string or numpy array")

        if mask is not None:
            if mask.shape != self.image_data.shape[:2]:
                raise ValueError(
                    f"mask shape {mask.shape} does not match image shape {self.image_data.shape[:2]}"
                )
            self.mask = mask > 0
        else:
            self.mask = None

    def _load_from_file(self, addr):
        """
        从文件路径加载图像并转换为RGB格式，存储为numpy二维矩阵。

        参数:
            addr: str, 图像文件路径（支持jpg格式）
        """
        img = Image.open(addr).convert("RGB")
        img_array = np.array(img)
        self.image_data = np.array([[(tuple(pixel)) for pixel in row] for row in img_array])

    def _load_from_array(self, np_array):
        """
        从numpy数组加载图像，转换为RGB格式后存储为numpy二维矩阵。

        参数:
            np_array: np.ndarray, 形状为 (H, W) 的灰度图或 (H, W, C) 的图像数组
        """
        if len(np_array.shape) == 2:
            img = Image.fromarray(np_array).convert("RGB")
            np_array = np.array(img)
        elif len(np_array.shape) == 3 and np_array.shape[2] == 1:
            img = Image.fromarray(np_array.squeeze()).convert("RGB")
            np_array = np.array(img)
        elif len(np_array.shape) == 3 and np_array.shape[2] == 3:
            pass
        else:
            raise ValueError("Unsupported array shape")
        self.image_data = np.array([[(tuple(pixel)) for pixel in row] for row in np_array])

    def _get_gray(self):
        """计算灰度图（uint8），与原 LBP/GLCM 中的灰度转换保持一致。"""
        height, width = self.image_data.shape[0], self.image_data.shape[1]
        gray = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                r, g, b = self.image_data[i, j]
                gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
        return gray

    def LBP_process(self):
        """
        对加载的图像进行3×3邻域的LBP（局部二值模式）运算。

        若初始化时传入了 mask，则仅对 mask 内的中心像素统计到直方图；
        mask 外的中心像素被跳过（其 8 邻居即便落在 mask 内也仍参与该中心像素的 LBP 编码）。

        返回:
            np.ndarray, 长度为256的LBP值直方图向量
        """
        if self.image_data is None:
            raise ValueError("No image data loaded")

        height, width = self.image_data.shape[0], self.image_data.shape[1]
        gray = self._get_gray()

        lbp_hist = np.zeros(256, dtype=np.int32)

        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, 1), (1, 1), (1, 0),
                   (1, -1), (0, -1)]

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # 仅当中心像素位于 mask 内时才把它的 LBP 码累加到直方图
                if self.mask is not None and not self.mask[i, j]:
                    continue
                center = gray[i, j]
                binary = 0
                for k, (di, dj) in enumerate(offsets):
                    neighbor = gray[i + di, j + dj]
                    if neighbor >= center:
                        binary |= (1 << k)
                lbp_hist[binary] += 1

        return lbp_hist

    def GLCM_process(self):
        """
        对加载的图像进行GLCM（灰度共生矩阵）纹理特征提取。

        若初始化时传入了 mask，则仅统计两个像素同时位于 mask 内的邻接对，
        避免 (0,0) 邻接计数对结果的淹没。

        返回:
            list of tuple, 包含两个元素，分别对应距离为1和距离为2的特征元组
            每个特征元组包含：(对比度, 角二阶矩, 逆差矩, 相关性, 熵, 和方差)
        """
        if self.image_data is None:
            raise ValueError("No image data loaded")

        height, width = self.image_data.shape[0], self.image_data.shape[1]
        gray = self._get_gray().astype(np.float64)

        gray_16 = (gray / 16).astype(np.int32)
        gray_16 = np.clip(gray_16, 0, 15)

        # 把 mask 外的像素标记为 -1，后续累加时会被跳过
        if self.mask is not None:
            gray_16 = np.where(self.mask, gray_16, -1)

        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        results = []

        for d in [1, 2]:
            glcm_sum = np.zeros((16, 16), dtype=np.float64)
            for dx, dy in directions:
                glcm = np.zeros((16, 16), dtype=np.float64)
                for i in range(height):
                    for j in range(width):
                        g1 = gray_16[i, j]
                        if g1 < 0:
                            continue
                        ni = i + d * dx
                        nj = j + d * dy
                        if 0 <= ni < height and 0 <= nj < width:
                            g2 = gray_16[ni, nj]
                            if g2 < 0:
                                continue
                            glcm[g1, g2] += 1
                glcm_sum += glcm

            glcm_avg = glcm_sum / 4.0
            glcm_symmetric = glcm_avg + glcm_avg.T

            total = glcm_symmetric.sum()
            if total > 0:
                p = glcm_symmetric / total
            else:
                p = glcm_symmetric

            contrast = 0.0
            asm = 0.0
            idm = 0.0
            entropy = 0.0

            for i in range(16):
                for j in range(16):
                    if p[i, j] > 0:
                        diff_sq = (i - j) ** 2
                        contrast += diff_sq * p[i, j]
                        asm += p[i, j] ** 2
                        idm += p[i, j] / (1 + diff_sq)
                        entropy -= p[i, j] * np.log(p[i, j])

            mu_i = 0.0
            mu_j = 0.0
            for i in range(16):
                for j in range(16):
                    mu_i += i * p[i, j]
                    mu_j += j * p[i, j]

            sigma_i = 0.0
            sigma_j = 0.0
            for i in range(16):
                for j in range(16):
                    sigma_i += ((i - mu_i) ** 2) * p[i, j]
                    sigma_j += ((j - mu_j) ** 2) * p[i, j]
            sigma_i = np.sqrt(sigma_i)
            sigma_j = np.sqrt(sigma_j)

            correlation = 0.0
            if sigma_i > 0 and sigma_j > 0:
                for i in range(16):
                    for j in range(16):
                        correlation += (i - mu_i) * (j - mu_j) * p[i, j]
                correlation /= (sigma_i * sigma_j)

            mu_s = 0.0
            for i in range(16):
                for j in range(16):
                    mu_s += (i + j) * p[i, j]

            sum_variance = 0.0
            for i in range(16):
                for j in range(16):
                    sum_variance += ((i + j) - mu_s) ** 2 * p[i, j]

            results.append((contrast, asm, idm, correlation, entropy, sum_variance))

        return results


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    if len(sys.argv) > 1:
        addr = sys.argv[1]
    else:
        addr = input("请输入图像文件路径: ")

    try:
        loader = imageLoader(addr)
        lbp_hist = loader.LBP_process()

        plt.figure(figsize=(12, 6))
        plt.bar(range(256), lbp_hist, width=1.0)
        plt.xlabel('LBP值')
        plt.ylabel('频数')
        plt.title('LBP直方图')
        plt.xlim([0, 255])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"错误: {e}")
