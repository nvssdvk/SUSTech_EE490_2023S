import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns


def phase_distribution():
    def shrink(arr):
        edge_min = 0
        edge_max = 360 + edge_min
        arr[arr > edge_max] -= 360
        arr[arr < edge_min] += 360
        if np.any(arr > edge_max) or np.any(arr < edge_min):
            return shrink(arr)
        else:
            return arr

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
            phi_arr[i, j] = - k * np.sqrt((x - feed[0]) ** 2 + (y - feed[1]) ** 2 + feed[2] ** 2) * 180 / np.pi

    phi_arr = shrink(phi_arr)
    plt.figure()
    plt.imshow(phi_arr, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

    return phi_arr


def aperture_efficiency(wave_len, x, y, q):
    def cal_spillover(x, y, h, q):
        out = 0
        for i in range(len(x)):
            for j in range(len(y)):
                r = np.sqrt(np.power(x[i], 2) + np.power(y[j], 2) + np.power(h, 2))
                temp = np.power(h / r, q * 2)
                out += (0.015 * 0.015 * temp * h / np.power(r, 3))
        out = out / (2 * np.pi / (2 * q + 1))
        return out

    def cal_illumination(x, y, h, q):
        # aperture_size = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
        aperture_size = (np.max(x) - np.min(x)) ** 2
        out1, out2 = 0, 0
        amp = np.zeros([len(x), len(y)])
        for i in range(len(x)):
            for j in range(len(y)):
                r = np.sqrt(np.power(x[i], 2) + np.power(y[j], 2) + np.power(h, 2))
                temp = np.power(h / r, q + 0) / r
                amp[i, j] = temp
                out1 += 0.015 * 0.015 * temp
                out2 += 0.015 * 0.015 * np.power(np.abs(temp), 2)
        out = 1 / aperture_size * np.power(np.abs(out1), 2) / out2
        return out, amp

    h_list = np.arange(start=wave_len * 10, stop=wave_len * 100, step=wave_len)
    list_num = len(h_list)
    e_spil = np.zeros([list_num, 1])
    e_illu = np.zeros([list_num, 1])
    amp_dist = np.zeros([list_num, len(x), len(y)])

    for h in h_list:
        id_h = np.where(h_list == h)[0].item()
        e_spil[id_h] = cal_spillover(x, y, h, q)
        e_illu[id_h], amp_dist[id_h] = cal_illumination(x, y, h, q)
    e_antenna = e_spil * e_illu

    plt.figure()
    plt.plot(h_list / wave_len, e_spil, color="r", label="Spillover")
    plt.plot(h_list / wave_len, e_illu, color="g", label="Illumination")
    plt.plot(h_list / wave_len, e_antenna, color="b", label="Antenna")
    plt.xlabel("Height(m)/$lambda$")
    plt.ylabel("")
    plt.grid()
    plt.legend()
    plt.title("Efficiency")
    plt.show()

    id_best = np.where(e_antenna == np.max(e_antenna))[0].item()
    h_best = h_list[id_best]
    print("Best Height of Feed: {:.3f} m\nEfficiency:\n\tSpillover: {:.3f}\n\tIllumination: {:.3f}\n\tAntenna: {:.3f}\n"
          .format(h_best, e_spil[id_best].item(), e_illu[id_best].item(), e_antenna[id_best].item()))
    return h_best


if __name__ == "__main__":
    wave_len = 0.3 / 10

    x = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    y = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    q = 6.5
    h = aperture_efficiency(wave_len, x, y, q)

    # h = 100 * wave_len
    # phase_array = phase_distribution()
