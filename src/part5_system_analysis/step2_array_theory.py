import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def shrink(arr):
    edge_min = 0
    edge_max = 360
    arr = np.mod(arr - edge_min, edge_max - edge_min) + edge_min
    return arr


def cal_element_pattern(qe, uv, bv, theta):
    # mask = np.logical_and(theta > -90, theta < 90)
    # element_pattern = np.zeros_like(theta, dtype=complex)
    # element_pattern[mask] = np.power(np.cos(theta[mask]), qe * 2) * np.exp(1j * k * np.dot(uv, bv))
    temp_1 = np.exp(1j * k * np.dot(uv, bv))
    temp_2 = np.power(np.cos(theta), qe)
    element_pattern = temp_1 * temp_2
    # element_pattern = 10 * np.log10(np.abs(element_pattern))
    return element_pattern


def cal_excitation(qf, qe, uv, fv, bv, phase):
    local_v = fv - uv
    r = np.linalg.norm(local_v)
    theta_f = np.arccos(local_v[2] / r)
    theta_e = np.arccos(local_v[2] / r) if r != 0 else 0
    # theta_f = np.arccos(fv[2] / np.linalg.norm(fv)) if r != 0 else 0

    a = np.cos(theta_f)
    temp_1 = np.power(a, qf) / r
    temp_2 = np.exp(-1j * k * np.linalg.norm(uv - fv))
    temp_3 = np.power(np.cos(theta_e), qe * 2)
    temp_4 = np.exp(1j * phase)

    illumination = temp_1 * temp_2 * temp_3 * temp_4
    return illumination


def plot_pattern(theta, pattern):
    file_path = r"../../data/dataset/reflectarray_pattern_10GHz.txt"
    df_data = pd.read_table(file_path, sep="\s+").values
    data = df_data[1:]
    sub_pattern = data[:, 2]
    simulation_pattern = np.concatenate((sub_pattern[::-1], sub_pattern[1::])).reshape(-1, 1)
    simulation_pattern = simulation_pattern[90:271]
    center = int(len(simulation_pattern) / 2)
    three_db_id = 5
    # sub_pattern = data[:, 2] - np.max(data[:, 2])
    # pattern[:center - three_db_id] = -np.Inf
    # pattern[center + three_db_id + 1:] = -np.Inf
    pattern = pattern[90:271]
    print(np.max(pattern))
    plt.figure(1)
    plt.plot(theta[90:271], pattern, label="Array-Theory Method")
    plt.plot(theta[90:271], simulation_pattern, label="Simulation")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(theta[90:271], simulation_pattern, label="Simulation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_aperture = pd.read_csv(r"../../data/dataset/aperture_dist.csv", header=0, engine="c").values

    wl = 3e8 / 10e9
    k = np.pi * 2 / wl
    d = wl / 2
    num = int(np.sqrt(data_aperture.shape[0]))
    qe = 1
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
            illumination = cal_excitation(qe=qe, qf=qf, uv=uv[i, j], fv=fv, bv=bv,
                                          phase=np.deg2rad(aperture_phase[i, j]))
            aperture_pattern_array[i, j] = pattern * illumination
            pattern_array[i, j] = pattern
            pattern_array_0[i, j] = pattern[int(len(theta) / 2)]
            illumination /= np.max(np.abs(illumination))
            illumination_array[i, j] = illumination
            # aperture_pattern += pattern * np.abs(illumination)
            aperture_pattern += pattern * illumination
    # for i in range(len(aperture_pattern)):
    #     aperture_pattern[i] = np.linalg.norm(aperture_pattern[i])
    # aperture_pattern = aperture_pattern.astype(float)
    aperture_pattern = np.abs(aperture_pattern)
    aperture_pattern = 10 * np.log10(aperture_pattern + 1e-15)
    plot_pattern(theta, (aperture_pattern))
