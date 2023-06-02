import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_best_h(wl, q, unit_num, qe=8.5, unit_len=None, theta=0):
    if unit_len is None:
        unit_len = wl / 2
    dx = dy = unit_len
    x = np.arange(-unit_num / 2 * dx + unit_len / 2, unit_num / 2 * dx + unit_len / 2, dx)
    y = np.arange(-unit_num / 2 * dy + unit_len / 2, unit_num / 2 * dy + unit_len / 2, dy)
    aperture_size = (unit_num * unit_len) ** 2

    def cal_spillover(x0):
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt((xx - x0) ** 2 + yy ** 2 + h ** 2)
        temp = np.power(h, q * 2 + 1) / r ** (3 + 2 * q)
        out = np.sum(unit_len ** 2 * temp)
        out /= (2 * np.pi / (2 * q + 1))
        return out

    def cal_illumination(x0):
        xx, yy = np.meshgrid(x, y, indexing='ij')
        r = np.sqrt((xx - x0) ** 2 + yy ** 2 + h ** 2)
        amp = h ** (q + qe) / r ** (1 + q + qe)
        out1 = np.sum(amp)
        out2 = np.sum(np.abs(amp) ** 2)
        out = (wl / 2) ** 2 * (out1 ** 2 / out2) / aperture_size
        return out, amp

    h_list = np.arange(start=wl * 3, stop=wl * 60, step=wl / 2)
    # h_list = np.asarray([wl * 8.0])
    list_num = len(h_list)
    e_spil = np.zeros_like(h_list)
    e_illu = np.zeros_like(h_list)
    amp_dist = np.zeros([list_num, len(x), len(y)])
    for h in h_list:
        x0 = -h * np.tan(np.deg2rad(theta))
        id_h = np.where(h_list == h)[0].item()
        e_spil[id_h] = cal_spillover(x0)
        e_illu[id_h], amp_dist[id_h] = cal_illumination(x0)
    e_antenna = e_spil * e_illu

    plt.figure()
    plt.plot(h_list / wl, e_spil, color="r", label="Spillover")
    plt.plot(h_list / wl, e_illu, color="g", label="Illumination")
    plt.plot(h_list / wl, e_antenna, color="b", label="Antenna")
    plt.xlabel("Height(m)/$\lambda$")
    plt.ylabel("")
    plt.grid()
    plt.legend()
    plt.title("Efficiency")
    plt.savefig(r'../../img/feed/efficiency.png')
    plt.show()

    id_best = np.where(e_antenna == np.max(e_antenna))[0].item()
    h_best = h_list[id_best]
    print(
        "Best Height of Feed: {:.1f} wave length\nEfficiency:\n\tSpillover: {:.3f}\n\tIllumination: {:.3f}\n\tAntenna: {:.3f}\n"
        .format(h_best / wl, e_spil[id_best].item(), e_illu[id_best].item(), e_antenna[id_best].item()))
    return h_best


def find_best_q(file_path):
    df_data = pd.read_table(file_path, sep="\s+").values
    data = df_data[1:]
    pattern = data[:, 2] - np.max(data[:, 2])
    ang = np.linspace(-180, 180, 361, dtype=int).reshape(-1, 1)
    mag = np.concatenate((pattern[::-1], pattern[1::])).reshape(-1, 1)

    plt.figure(figsize=(19.2, 7.2))
    plt.subplot(121)
    plt.plot(ang, mag, color="r", label="CST")

    q_range = np.arange(start=1, stop=20, step=0.5)
    loss = np.zeros([len(q_range), 1])
    cos_model = np.zeros([361, len(q_range)])
    with np.errstate(divide='ignore'):
        for q in q_range:
            cos_x = np.cos(ang * np.pi / 180)
            cos_value = 10 * np.log10(np.power(np.clip(cos_x, 1e-10, None), q * 2))
            index = np.where(q_range == q)[0].item()
            cos_model[:, index] = cos_value.flat
            loss[index] = 1 / 61 * np.sum(np.abs(cos_value[150:211] - mag[150:211]))
        for i in range(len(q_range)):
            if i == np.where(loss == np.min(loss))[0].item():
                q_best = q_range[i]
                plt.plot(ang, cos_model[:, i], label="q={:.1f}".format(q_range[i]))

    plt.xlim(-90, 90)
    plt.ylim(-30, 5)
    plt.xlabel("Theta")
    plt.ylabel("Mag(dB)")
    plt.title("Power")
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(q_range, loss)
    plt.xlabel("q")
    plt.ylabel("Loss(dB)")
    plt.xlim(0, 20)
    plt.title("Loss of cos-q Model")
    plt.grid()
    plt.savefig("../../img/feed/q.png")
    plt.show()

    print("Best q is {:.1f}".format(q_best))
    return q_best


if __name__ == "__main__":
    q = find_best_q(r"../../data/dataset/feed_horn_pattern_30.txt")

    # h = find_best_h(wl=3e8 / 10e9, q=q, qe=0, unit_num=21, theta=10)
    h = find_best_h(wl=3e8 / 10e9, q=q, qe=0, unit_num=21, theta=10)
    '''
    Phase center (0, 0, 17.0448)
    
    theta = 0
    Best q is 8.5
    Best Height of Feed: 12.5 wave length
    Efficiency:
        Spillover: 0.829
        Illumination: 0.799
        Antenna: 0.663
       
    theta = 30
    Best q is 7
    Best Height of Feed: 6.0 wave length
    Efficiency:
        Spillover: 0.888
        Illumination: 0.236
        Antenna: 0.210

    '''
