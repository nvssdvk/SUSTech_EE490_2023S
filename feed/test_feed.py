import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# from scipy.integrate import tplquad, dblquad, quad

def phase_distribution(wave_len, x, y, h):
    wave_num = 2 * np.pi / wave_len
    r = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            r[i, j] = np.sqrt(np.power(x[i], 2) + np.power(y[j], 2) + np.power(h, 2))
    return wave_num * r


def spillover(wave_len, x, y, q):
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
        aperture_size = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
        out1, out2 = 0, 0
        for i in range(len(x)):
            for j in range(len(y)):
                r = np.sqrt(np.power(x[i], 2) + np.power(y[j], 2) + np.power(h, 2))
                temp = np.power(h / r, q) / r * np.power(h / r, 0)
                out1 += 0.015 * 0.015 * temp
                out2 += 0.015 * 0.015 * np.power(np.abs(temp), 2)
        out = 1 / aperture_size * np.power(np.abs(out1), 2) / out2
        return out

    h_list = np.arange(start=wave_len * 1, stop=wave_len * 50, step=wave_len)
    list_num = len(h_list)
    e_spil = np.zeros([list_num, 1])
    e_illu = np.zeros([list_num, 1])

    for h in h_list:
        e_spil[np.where(h_list == h)[0].item()] = cal_spillover(x, y, h, q)
        e_illu[np.where(h_list == h)[0].item()] = cal_illumination(x, y, h, q)
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
    x = np.arange(start=(-150 + 7.5) / 1e3, stop=150 / 1e3, step=15 / 1e3)
    y = np.arange(start=(-150 + 7.5) / 1e3, stop=150 / 1e3, step=15 / 1e3)
    # x = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    # y = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    q = 6.5
    # spillover(wave_len, x, y, q)
    h = 0.27
    phase_array = phase_distribution(wave_len, x, y, h)
