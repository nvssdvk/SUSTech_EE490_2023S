import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def shrink(arr):
    edge_min = 0
    edge_max = 360
    arr = np.mod(arr - edge_min, edge_max - edge_min) + edge_min
    return arr


def vector_normalize(vector):
    out = vector / np.linalg.norm(vector)
    return out


def cal_element_pattern(qe, uv, ov, theta):
    element_pattern = np.zeros_like(theta, dtype=complex)
    for index in range(len(theta)):
        # temp_1 = np.exp(1j * k * vector_normalize(uv * ov[index]))
        temp_1 = np.exp(1j * k * (np.dot(uv, ov[index])))
        temp_2 = np.sign(np.cos(theta[index])) * (np.abs(np.cos(theta[index]))) ** qe
        element_pattern[index] = temp_1 * temp_2
    return element_pattern


def cal_excitation(qf, qe, uv, fv, ov, bv, phase, theta):
    illumination = np.zeros_like(theta, dtype=complex)
    r = np.linalg.norm(fv - uv)
    temp_2 = np.exp(-1j * k * r)
    temp_4 = np.exp(1j * phase)
    for index in range(len(theta)):
        ov_sub = ov[index]
        # if np.linalg.norm(uv) != 0:
        #     theta_e = np.arccos(np.dot(uv, bv_sub) / (np.linalg.norm(uv) * np.linalg.norm(bv_sub)))
        # else:
        #     theta_e = np.arccos(bv_sub[2] / np.linalg.norm(bv_sub))
        # theta_f = np.arccos(bv_sub[2] / np.linalg.norm(bv_sub))
        if np.linalg.norm(uv) != 0:
            theta_e = np.arccos(np.dot(uv, bv) / (np.linalg.norm(uv) * np.linalg.norm(bv)))
        else:
            theta_e = np.arccos(bv[2] / np.linalg.norm(bv))
        theta_f = np.arccos(np.dot(bv, fv) / (np.linalg.norm(fv) * np.linalg.norm(bv)))
        temp_1 = np.sign(np.cos(theta_f)) * (np.abs(np.cos(theta_f))) ** qf
        temp_3 = np.sign(np.cos(theta_e)) * (np.abs(np.cos(theta_e))) ** qe
        illumination[index] = temp_1 * temp_2 * temp_3 * temp_4
    return illumination


def plot_pattern(theta, pattern):
    file_path = r"../../data/dataset/reflectarray_pattern_10GHz.txt"
    df_data = pd.read_table(file_path, sep="\s+").values
    data = df_data[1:]
    sub_pattern = data[:, 2]
    simulation_pattern = np.concatenate((sub_pattern[::-1], sub_pattern[1::])).reshape(-1, 1)
    simulation_pattern = simulation_pattern[90:271]
    simulation_pattern -= np.max(simulation_pattern)
    center = int(len(simulation_pattern) / 2)

    pattern = pattern[90:271]
    print(np.max(pattern))
    plt.figure(1)
    plt.plot(theta[90:271], pattern, label="Array-Theory Method")
    plt.plot(theta[90:271], simulation_pattern, label="Simulation")
    plt.legend()
    plt.ylim(-50, 5)
    plt.title("qe={:.1f}, qf={:.1f}".format(qe, qf))
    plt.show()

    # plt.figure(2)
    # plt.plot(theta[90:271], simulation_pattern, label="Simulation")
    # plt.legend()
    # plt.show()

    # plt.figure(3)
    # plt.plot(theta[90:271], pattern, label="Array-Theory Method")
    # plt.plot(theta[90:271], simulation_pattern, label="Simulation")
    # plt.legend()
    # plt.title("qe={:.1f}, qf={:.1f}".format(qe, qf))
    # plt.show()


if __name__ == "__main__":
    data_aperture = pd.read_csv(r"../../data/dataset/aperture_dist.csv", header=0, engine="c").values

    wl = 3e8 / 10e9
    k = np.pi * 2 / wl
    d = wl / 2
    num = int(np.sqrt(data_aperture.shape[0]))
    qe = 0
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

    phi = 90
    theta = np.linspace(-180, 180, 361, dtype=int)
    uv = np.stack((x, y, z), axis=2)
    bv = np.asarray([0, 0, 1])
    ov = np.zeros([len(theta), 3])
    ov[:, 0] = np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    ov[:, 1] = np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    ov[:, 2] = np.cos(np.deg2rad(theta))
    fv = np.asarray([0, 0, 9.5 * wl])

    aperture_pattern = np.zeros(len(theta), dtype=complex)
    aperture_pattern_array = np.zeros([num, num, len(theta)], dtype=complex)
    pattern_array = np.zeros([num, num, len(theta)], dtype=complex)
    illumination_array = np.zeros([num, num, len(theta)], dtype=complex)
    for i in range(num):
        for j in range(num):
            pattern = cal_element_pattern(qe=qe, uv=uv[i, j], ov=ov, theta=np.deg2rad(theta))
            illumination = cal_excitation(qe=qe, qf=qf, uv=uv[i, j], fv=fv, ov=ov, bv=bv,
                                          phase=np.deg2rad(aperture_phase[i, j]), theta=np.deg2rad(theta))
            # pattern = np.power(10, pattern/10)
            aperture_pattern_array[i, j] = pattern * illumination
            # aperture_pattern_array[i, j] = 10 * np.log10(pattern * illumination)
            aperture_pattern += aperture_pattern_array[i, j]
            pattern_array[i, j] = pattern
            illumination_array[i, j] = illumination

    aperture_pattern = np.abs(aperture_pattern)
    aperture_pattern = 20 * np.log10(aperture_pattern + 1e-15)
    aperture_pattern -= np.max(aperture_pattern)
    plot_pattern(theta, aperture_pattern)
