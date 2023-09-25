import cv2
import numpy as np
import pylab as plt
from AMPD import AMPD

image = cv2.imread('1.jpg')

# BGR三通道
B, G, R = cv2.split(image)

# 计算灰度图像数据
denominator = B + G + R
non_zero_denominator = np.where(denominator != 0, denominator, 1)
b = B / non_zero_denominator
g = G / non_zero_denominator
r = R / non_zero_denominator
gray = (2 * g - b - r) * ((0 < 2 * g - b - r) & (2 * g - b - r < 1)) + 1 * (2 * g - b - r >= 1)

# 将 gray 数组转换为灰度图像
gray_image = (gray * 255).astype(np.uint8)

# 使用Otsu二值化
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 形态学降噪
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(binary_image, kernel, iterations=2)
denoised_image = cv2.dilate(eroded, kernel, iterations=2)

# 均值滤波
mean_blur = cv2.blur(denoised_image, (3, 3))  # 使用5x5的卷积核
# 计算均值滤波后图像的垂直投影
vertical_projection = np.sum(mean_blur, axis=0)
# 创建垂直投影的x坐标
x = np.arange(len(vertical_projection))

# 找到每一段的最大值的索引
all_max_x = AMPD(vertical_projection)
print("像数值最高点的横坐标:", all_max_x)

# 创建一个与原始图像相同大小的空白图像
line_image = np.zeros_like(image)
for max_x in all_max_x:
    cv2.line(line_image, (max_x, 0), (max_x, line_image.shape[0]), (0, 0, 255), 5)

# 将绘制的线添加到原始图像上
marked_image = cv2.addWeighted(image, 1, line_image, 1, 0)

# 绘制垂直投影图
plt.figure(figsize=(8, 4))
plt.rc('font', family='SimHei')
plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.title('标记了 max_x 的图像')
plt.axis('off')
plt.tight_layout()
plt.show()
