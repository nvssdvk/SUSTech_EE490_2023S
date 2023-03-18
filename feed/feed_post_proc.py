import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import tplquad, dblquad, quad


def spillover(wave_len):
    def cal_spillover(x, y, h, q):
        r2 = np.power(x, 2) + np.power(y, 2)
        r = np.sqrt(r2)
        s2 = r2 + np.power(h, 2)
        out = np.power((h ** 2 + r2 - s2) / (2 * r * h), q * 2)
        out = np.sum(0.015 * 0.015 * out * h / (r2 * r))
        return out

    # def cal_spillover(q, h):
    #     f = lambda y, x: h / (np.sqrt(x ** 2 + y ** 2 + h ** 2).item() ** 3) * (
    #             (h ** 2 + np.sqrt(x ** 2 + y ** 2 + h ** 2).item() ** 2 - (x ** 2 + y ** 2)) / (
    #             2 * h * np.sqrt(x ** 2 + y ** 2 + h ** 2).item())) ** (2 * q)
    #
    #     return dblquad(f, -150/1000, 150/1000, -150/1000, 150/1000)
    x = np.arange(start=-150 + 7.5, stop=150, step=15)
    y = np.arange(start=-150 + 7.5, stop=150, step=15)
    q = 6.5
    h_list = np.arange(start=wave_len * 1, stop=wave_len * 25, step=wave_len)
    list_num = len(h_list)
    value = np.zeros([list_num, 1])
    for h in h_list:
        value[np.where(h_list == h)[0].item()] = cal_spillover(x, y, h, q)

    plt.figure()
    plt.plot(h_list, value, color="r", label="CST")
    plt.show()


def illumination():
    pass


def efficiency():
    pass


def find_best_q():
    data = pd.read_table(r"../data/feed/horn_pattern.txt", sep="\s+").values
    pattern = data[:, 2] - np.max(data[:, 2])
    ang = np.linspace(-180, 180, 361, dtype=int).reshape([361, 1])
    mag = np.concatenate((pattern[::-1], pattern[1::])).reshape([361, 1])

    plt.figure()
    plt.plot(ang, mag, color="r", label="CST")

    q_range = np.arange(start=1, stop=20, step=0.1)
    loss = np.zeros([len(q_range), 1])
    cos_model = np.zeros([361, len(q_range)])
    with np.errstate(divide='ignore'):
        for q in q_range:
            cos_x = np.cos(ang * np.pi / 180)
            cos_value = 10 * np.log10(np.power(cos_x, q * 2))
            index = np.where(q_range == q)[0].item()
            cos_model[:, index] = cos_value.flat
            # plt.plot(ang, cos_model, label="q={:.1f}".format(q))
            loss[index] = 1 / 61 * np.sum(np.abs(cos_value[150:211] - mag[150:211]))

    # plt.xlim(-90, 90)
    # plt.ylim(-30, 10)
    # plt.xlabel("Theta")
    # plt.ylabel("Mag(dB)")
    # plt.title("E-Pattern")
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(q_range, loss)
    # plt.xlabel("Theta")
    # plt.ylabel("Mag(dB)")
    # plt.xlim(0, 20)
    # plt.title("Loss of cos-q Model")
    # plt.grid()
    # plt.show()

    q_best_id = np.where(loss == np.min(loss))[0].item()
    q_best = q_range[q_best_id]
    print("Best q is {:.1f}".format(q_best))
    return q_best


if __name__ == "__main__":
    config = {
        'feed': {
            'hf': 20,
            'q': 6.5,
        },
        'aperture': {
            'num': 20,
            'len': 15,  # unit:mm
            'x_range': [-150, 150],
            'y_range': [-150, 150],
        },

    }
    # units_center = np.arange(start=-150 + 7.5, stop=150, step=15)
    # units_center = units_center.reshape([len(units_center), 1])
    spillover(wave_len=300 / 11)
    # df = pd.DataFrame(columns=tt_name, data=tt_set)
    # df.to_csv(f'../data/dataset/tt_set.csv', encoding='utf-8', index=False)
