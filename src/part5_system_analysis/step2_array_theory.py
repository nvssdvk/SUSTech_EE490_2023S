import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def shrink(arr):
    edge_min = 0
    edge_max = 360
    arr = np.mod(arr - edge_min, edge_max - edge_min) + edge_min
    return arr


def cal_element_pattern(qe, uv, bv, theta):
    element_pattern = np.power(np.clip(np.cos(theta), 1e-10, None), qe) * np.exp(1j * k * np.dot(uv, bv))
    # element_pattern = np.power(np.clip(np.cos(0), 1e-10, None), qe) * np.exp(1j * k * np.dot(uv, bv))
    return element_pattern


def cal_excitation(qf, qe, uv, fv, phase, theta):
    local_v = uv - fv
    r = np.linalg.norm(local_v)
    if r == 0:
        theta_f = 0
    else:
        theta_f = np.arccos(local_v[2] / r)
    local_v = fv - uv
    r = np.linalg.norm(local_v)
    if r == 0:
        theta_e = 0
    else:
        theta_e = np.arccos(local_v[2] / r)
    temp_1 = np.power(np.clip(np.cos(theta_f), 1e-10, None), qf) / np.linalg.norm(uv - fv)
    temp_2 = np.exp(-1j * k * np.linalg.norm(uv - fv))
    temp_3 = np.power(np.clip(np.cos(theta_e), 1e-10, None), qe)
    temp_4 = np.exp(1j * phase)

    illumination = temp_1 * temp_2 * temp_3 * temp_4
    return illumination


def plot_pattern(theta, pattern):
    file_path = r"../../data/dataset/reflectarray_pattern_10GHz.txt"
    data = pd.read_table(file_path, sep="\s+").values
    sub_pattern = data[:, 2] - np.max(data[:, 2])
    simulation_pattern = np.concatenate((sub_pattern[::-1], sub_pattern[1::])).reshape(-1, 1)
    plt.figure()
    plt.plot(theta, pattern, label="Array-Theory Method")
    plt.plot(theta, simulation_pattern, label="Simulation")
    plt.legend()
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

    theta = np.linspace(-180, 180, 361, dtype=int)
    aperture_pattern = np.zeros(len(theta), dtype=complex)
    aperture_pattern_array = np.zeros([num, num, len(theta)], dtype=complex)
    pattern_array = np.zeros([num, num, len(theta)], dtype=complex)
    pattern_array_0 = np.zeros([num, num], dtype=complex)
    illumination_array = np.zeros([num, num], dtype=complex)
    for i in range(num):
        for j in range(num):
            pattern = cal_element_pattern(qe=qe, uv=uv[i, j], bv=bv, theta=np.deg2rad(theta))
            illumination = cal_excitation(qe=qe, qf=qf, uv=uv[i, j], fv=fv, phase=np.deg2rad(aperture_phase[i, j]),
                                          theta=np.deg2rad(theta))
            aperture_pattern_array[i, j] = pattern * illumination
            pattern_array[i, j] = pattern
            pattern_array_0[i, j] = pattern[int(len(theta) / 2)]
            illumination_array[i, j] = illumination
            aperture_pattern += pattern * illumination
    # for i in range(len(aperture_pattern)):
    #     aperture_pattern[i] = np.linalg.norm(aperture_pattern[i])
    # aperture_pattern = aperture_pattern.astype(float)
    aperture_pattern = np.abs(aperture_pattern)
    # aperture_pattern = 10 * np.log10(aperture_pattern)
    plot_pattern(theta, (aperture_pattern))
