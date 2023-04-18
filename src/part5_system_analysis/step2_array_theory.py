import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def shrink(arr):
    edge_min = 0
    edge_max = 360
    arr = np.mod(arr - edge_min, edge_max - edge_min) + edge_min
    return arr


def cal_element_pattern(qe, uv, bv, theta=None):
    if theta is None:
        theta = np.arange(-179, 180, 1)

    element_pattern = np.power(np.clip(np.cos(theta), 1e-10, None), qe) * np.exp(1j * k * np.dot(uv, bv))
    return element_pattern


def cal_excitation(qf, qe, uv, fv, phase):
    r_f = np.sqrt(fv[0] ** 2 + fv[1] ** 2 + fv[2] ** 2)
    theta_f = np.arccos(fv[2] / r_f)
    r_e = np.sqrt(uv[0] ** 2 + uv[1] ** 2 + uv[2] ** 2)
    if r_e == 0:
        theta_e = 0
    else:
        theta_e = np.arccos(uv[2] / r_e)

    # theta_e = np.where(r_e == 0, 0, np.arccos(uv[2] / r_e))

    temp_1 = np.power(np.clip(np.cos(theta_f), 1e-10, None), qf) / np.linalg.norm(uv - fv)
    temp_2 = np.exp(-1j * k * np.linalg.norm(uv - fv))
    temp_3 = np.power(np.clip(np.cos(theta_e), 1e-10, None), qe)
    temp_4 = np.exp(1j * phase)

    illumination = temp_1 * temp_2 * temp_3 * temp_4
    return illumination


def plot_pattern(theta, pattern):
    plt.figure()
    plt.plot(theta, np.abs(pattern))
    plt.show()


if __name__ == "__main__":
    data_aperture = pd.read_csv(r"../../data/dataset/aperture_dist.csv", header=0, engine="c").values

    wl = 3e8 / 10e9
    k = np.pi * 2 / wl
    d = wl / 2
    num = int(np.sqrt(data_aperture.shape[0]))
    qe = 7
    qf = 8.5

    aperture_phase = np.zeros([num, num])
    for i in range(num):
        for j in range(num):
            temp_data = data_aperture[(data_aperture[:, 0] == i) & (data_aperture[:, 1] == j)]
            aperture_phase[i, j] = temp_data[:, 2]
    aperture_phase = shrink(aperture_phase)

    x, y = np.meshgrid(np.arange(-(num - 1) // 2, (num + 1) // 2), np.arange(-(num - 1) // 2, (num + 1) // 2))
    x = x * d
    y = y * d
    z = np.zeros_like(x)
    uv = np.stack((x, y, z), axis=2)
    bv = np.asarray([0, 0, 1])
    fv = np.asarray([0, 0, 9.5 * wl])

    theta = np.arange(-179, 180, 1)
    aperture_pattern = np.zeros(len(theta), dtype=complex)
    aperture_pattern_array = np.zeros([num, num, len(theta)], dtype=complex)
    for i in range(num):
        for j in range(num):
            pattern = cal_element_pattern(qe=qe, uv=uv[i, j], bv=bv)
            illumination = cal_excitation(qe=qe, qf=qf, uv=uv[i, j], fv=fv, phase=np.deg2rad(aperture_phase[i, j]))
            aperture_pattern_array[i, j] = pattern * illumination
            aperture_pattern += pattern * illumination
    plot_pattern(theta, aperture_pattern)
