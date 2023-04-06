import numpy as np
import matplotlib.pyplot as plt

# 定义常量
c = 3e8  # 光速
f = 1e9  # 频率
lambda_ = c / f  # 波长
k = 2 * np.pi / lambda_  # 波数
d = lambda_ / 2  # 元素间距

# 定义角度范围（弧度）
theta = np.linspace(-np.pi, np.pi, 1000)


# 定义阵列因子函数
def array_factor(N, d, k, theta):
    # N: 元素个数
    # d: 元素间距
    # k: 波数
    # theta: 角度（弧度）
    af = 0  # 初始化阵列因子为0
    for n in range(N):  # 对每个元素求和
        af += np.exp(1j * n * d * k * np.cos(theta))  # 加上每个元素对应的复数指数项
    return af


# 计算四个元素的阵列因子并取绝对值平方（归一化）
af4 = array_factor(4, d, k, theta)
af4_norm = np.abs(af4) ** 2 / np.max(np.abs(af4) ** 2)

# 绘制方向图（极坐标）
plt.figure()
plt.polar(theta, af4_norm)
plt.title('Directional pattern of a four-element array')
plt.show()
