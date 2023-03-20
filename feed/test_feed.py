import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns


def plot_phase_distribution(phase_array):
    phase_data = pd.DataFrame(phase_array)
    array_id = np.asarray(range(len(phase_array)))
    df = pd.DataFrame(phase_data, index=array_id[::-1], columns=array_id)

    plt.figure()
    sns.heatmap(data=df)
    plt.title("Phase Distribution")
    plt.show()


def phase_distribution(wave_len, x, y, h):
    def phase_normal(phase_array):
        mask = phase_array > 180  # 创建一个布尔数组，表示哪些元素大于180
        phase_array[mask] = np.subtract(phase_array[mask], 360)  # 对大于180的元素减去360
        mask = phase_array < -180  # 创建一个布尔数组，表示哪些元素大于180
        phase_array[mask] = np.add(phase_array[mask], 360)  # 对大于180的元素减去360
        mask = (phase_array > 180) | (phase_array < -180)
        cnt = np.count_nonzero(mask)
        if cnt == 0:
            return phase_array
        else:  # 创建一个布尔数组，表示哪些元素大于180或小于-180
            return phase_normal(phase_array)

    wave_num = 2 * np.pi / wave_len
    # wave_num = 1 / wave_len
    r = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            r[i, j] = np.sqrt(np.power(x[i], 2) + np.power(y[j], 2) + np.power(h, 2))
    phase_array = wave_num * r
    phase_array = phase_normal(phase_array)
    return phase_array


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
    f = 10e9  # 工作频率为10GHz
    c = 3e8  # 光速
    k = 2 * np.pi * f / c  # 波数
    x0 = 0  # 馈电天线的相位中心x坐标
    y0 = 0  # 馈电天线的相位中心y坐标
    z0 = 100  # 馈电天线的相位中心z坐标
    N = 20  # 反射面共有20*20个单元
    # 假设反射面边长为1米，每个单元边长为0.05米
    dx = dy = 0.05
    # 创建一个二维数组表示反射面上的x坐标
    x_arr = np.arange(-N / 2 * dx, N / 2 * dx, dx)
    # 创建一个二维数组表示反射面上的y坐标
    y_arr = np.arange(-N / 2 * dy, N / 2 * dy, dy)
    # 使用np.meshgrid函数将两个一维数组转换为二维网格数组
    X, Y = np.meshgrid(x_arr, y_arr)

    wave_len = 0.3 / 10
    # x = np.arange(start=(-150 + 7.5) / 1e3, stop=150 / 1e3, step=15 / 1e3)
    # y = np.arange(start=(-150 + 7.5) / 1e3, stop=150 / 1e3, step=15 / 1e3)
    x = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    y = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    q = 6.5
    # h = aperture_efficiency(wave_len, x, y, q)
    # h = 0.27
    # h = 1.59
    h = 100 * wave_len
    phase_array = phase_distribution(wave_len, x, y, h)
    # %%
    plot_phase_distribution(phase_array)
