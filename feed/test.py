import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def shrink(arr):
    edge_min = 0
    edge_max = 360 + edge_min
    arr[arr > edge_max] -= 360
    arr[arr < edge_min] += 360
    if np.any(arr > edge_max) or np.any(arr < edge_min):
        return shrink(arr)
    else:
        return arr


if __name__ == "__main__":
    f = 32e9  # 工作频率
    c = 3e8  # 光速
    wl = c / f
    print("wave length:{:.2f}mm".format(wl * 1e3))
    k = 2 * np.pi / wl  # 波数
    # h =
    # feed = [-91.88 / 1e3, 0, 342.9 / 1e3]
    feed = [0, 0, 0.17]
    N = 40  # 反射面共有N*N个单元

    dx = dy = 4.7 / 1e3
    x_arr = np.arange(-N / 2 * dx, N / 2 * dx, dx)
    y_arr = np.arange(-N / 2 * dy, N / 2 * dy, dy)
    X, Y = np.meshgrid(x_arr, y_arr)

    # 创建一个空的二维数组表示反射面上的相位分布
    phi_arr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x = X[i, j]  # 获取当前单元的x坐标值
            y = Y[i, j]  # 获取当前单元的y坐标值
            phi_arr[i, j] = - k * np.sqrt((x - feed[0]) ** 2 + (y - feed[1]) ** 2 + feed[2] ** 2) *180 / np.pi

    phi_arr = shrink(phi_arr)
    plt.figure()
    plt.imshow(phi_arr, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
