import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def edge_taper(theta_fed, hf, qf):
    def cal_illumination(q, theta, r):
        molecule = np.sign(np.cos(theta)) * (np.abs(np.cos(theta))) ** q
        denominator = r
        illumination = molecule / denominator
        return illumination

    d_ra = unit_len * unit_num
    delta_b = 0
    r_fp = hf / np.cos(theta_fed)
    r_c = np.sqrt(delta_b ** 2 + r_fp ** 2
                  - 2 * r_fp * delta_b * np.cos(theta_fed + np.pi / 2))
    r_u = np.sqrt((d_ra / 2 + delta_b) ** 2 + r_fp ** 2
                  - 2 * r_fp * (d_ra / 2 + delta_b) * np.cos(theta_fed + np.pi / 2))
    r_l = np.sqrt((d_ra / 2 - delta_b) ** 2 + r_fp ** 2
                  - 2 * r_fp * (d_ra / 2 - delta_b) * np.cos(theta_fed + np.pi / 2))
    r_s = np.sqrt((d_ra / 2) ** 2 + r_c ** 2)

    theta_u = np.arccos(hf / r_u) - theta_fed
    theta_l = np.arccos(hf / r_l) + theta_fed
    theta_s = np.arctan(d_ra / (2 * r_c))

    i_max = cal_illumination(q=qf, theta=theta_fed, r=r_fp)
    i_ue = cal_illumination(q=qf, theta=theta_u, r=r_u)
    i_le = cal_illumination(q=qf, theta=theta_l, r=r_l)
    i_se = cal_illumination(q=qf, theta=theta_s, r=r_s)

    et_u = 20 * np.log10(i_ue / i_max + 1e-15)
    et_l = 20 * np.log10(i_le / i_max + 1e-15)
    et_s = 20 * np.log10(i_se / i_max + 1e-15)
    return et_u, et_l, et_s


if __name__ == "__main__":
    wl = 3e8 / 10e9
    k = np.pi * 2 / wl
    unit_len = wl / 2
    unit_num = 21
    qf = 8.5
    theta_fed = np.deg2rad(0)
    hf = np.arange(wl, wl * 30, wl/2).reshape(-1)
    theta_beam = np.deg2rad(0)

    et_u_arr = np.zeros_like(hf)
    et_l_arr = np.zeros_like(hf)
    et_s_arr = np.zeros_like(hf)
    for i in range(len(hf)):
        et_u_arr[i], et_l_arr[i], et_s_arr[i] = edge_taper(theta_fed, hf[i], qf)

    # plt.figure(figsize=(19.2, 10.8))
    x = (hf / wl)
    plt.plot(x, et_u_arr, label="$ET_{UE}$")
    plt.plot(x, et_l_arr, label="$ET_{LE}$")
    plt.plot(x, et_s_arr, label="$ET_{SE}$")
    plt.xlabel(r"$H_F$ / $\lambda$")
    plt.ylabel("EdgeTaper(dB)")
    plt.title(r"Edge Taper with qf={:.1f}, $\theta$={:.1f}".format(qf, np.rad2deg(theta_fed)))
    plt.legend()

    x_value = 12.5
    y_value = et_u_arr[np.where(x == x_value)]
    plt.annotate(f"y={y_value}", xy=(x_value, y_value), xytext=(x_value + 0.5, y_value + 1),
                 arrowprops=dict(arrowstyle='->'))

    plt.axvline(x=x_value, color='r', linestyle='--')
    threshold = -10
    plt.fill_between(x, et_u_arr, threshold, where=(et_u_arr < threshold), color='gray', alpha=0.5)

    plt.show()
