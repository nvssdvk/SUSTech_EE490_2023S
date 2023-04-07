import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def phase_distribution(wl=3e8 / 32e9, feed_position=None, unit_len=None, unit_num=40, beam_theta=0, beam_phi=0):
    if feed_position is None:
        feed_position = [0, 0, 0.17]
    if unit_len is None:
        unit_len = wl / 2

    def shrink(arr):
        edge_min = 0
        edge_max = 360
        arr = np.mod(arr - edge_min, edge_max - edge_min) + edge_min
        return arr

    print("wave length:{:.2f}mm".format(wl * 1e3))
    k = 2 * np.pi / wl  # 波数
    beam_theta = np.deg2rad(beam_theta)
    beam_phi = np.deg2rad(beam_phi)

    dx = dy = unit_len
    x_arr = np.arange(-unit_num / 2 * dx, unit_num / 2 * dx, dx)
    y_arr = np.arange(-unit_num / 2 * dy, unit_num / 2 * dy, dy)
    xx, yy = np.meshgrid(x_arr, y_arr)

    plt.figure(figsize=(19.2, 10.8))

    plt.subplot(221)
    xspd = xx - feed_position[0]  # vectorize calculation of x and y
    yspd = yy - feed_position[1]
    z = feed_position[2]
    phi_arr_spd = -k * np.sqrt(xspd ** 2 + yspd ** 2 + z ** 2)
    phi_arr_spd = phi_arr_spd * 180 / np.pi
    phi_arr_spd = shrink(phi_arr_spd)
    plt.imshow((phi_arr_spd), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [element number]")
    plt.ylabel("y-axis [element number]")
    plt.title("Spatial Delay")

    plt.subplot(222)
    cos_phi = np.cos(beam_phi)
    sin_phi = np.sin(beam_phi)
    phi_arr_pp = -k * (xx * cos_phi + yy * sin_phi) * np.sin(beam_theta)
    phi_arr_pp = phi_arr_pp * 180 / np.pi
    phi_arr_pp = shrink(phi_arr_pp)
    plt.imshow((phi_arr_pp), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [element number]")
    plt.ylabel("y-axis [element number]")
    plt.title("Progressive Phase")

    plt.subplot(223)
    phi_arr = -phi_arr_spd + phi_arr_pp
    phi_arr = shrink(phi_arr)
    plt.imshow(phi_arr, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [element number]")
    plt.ylabel("y-axis [element number]")
    plt.title("Phase Distribution on the Reflectarray Antenna")

    plt.subplot(224)
    unit_num_con = 1e3
    dx = dy = unit_len
    x_arr = np.arange(-unit_num / 2 * dx, unit_num / 2 * dx, unit_len * unit_num / unit_num_con)
    y_arr = np.arange(-unit_num / 2 * dy, unit_num / 2 * dy, unit_len * unit_num / unit_num_con)
    xx, yy = np.meshgrid(x_arr, y_arr)
    xspd = xx - feed_position[0]  # vectorize calculation of x and y
    yspd = yy - feed_position[1]
    z = feed_position[2]
    cos_phi, sin_phi = np.cos(beam_phi), np.sin(beam_phi)
    phi_arr_con = k * np.sqrt(xspd ** 2 + yspd ** 2 + z ** 2) - k * (xx * cos_phi + yy * sin_phi) * np.sin(beam_theta)
    phi_arr_con = phi_arr_con * 180 / np.pi
    phi_arr_con = shrink(phi_arr_con)
    plt.imshow(phi_arr_con, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [mm]")
    plt.ylabel("y-axis [mm]")
    plt.title("Phase Distribution on the Continuous Aperture")

    plt.show()

    return phi_arr


def aperture_efficiency(wl, x, y, q):
    def cal_spillover(x, y, h, q):
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(np.power(xx, 2) + np.power(yy, 2) + np.power(h, 2))
        temp = np.power(h / rr, q * 2)
        out = np.sum(0.015 * 0.015 * temp * h / np.power(rr, 3))
        out = out / (2 * np.pi / (2 * q + 1))
        return out

    def cal_illumination(x, y, h, q):
        aperture_size = (np.max(x) - np.min(x)) ** 2
        out1, out2 = 0, 0
        xx, yy = np.meshgrid(x, y, indexing='ij')
        r = np.sqrt(xx ** 2 + yy ** 2 + h ** 2)
        temp = (h / r) ** (q + 0) / r
        amp = temp.copy()
        out1 += 0.015 ** 2 * np.sum(temp)
        out2 += 0.015 ** 2 * np.sum(np.abs(temp) ** 2)
        out = 1 / aperture_size * np.abs(out1) ** 2 / out2
        return out, amp

    h_list = np.arange(start=wl * 10, stop=wl * 100, step=wl)
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
    plt.plot(h_list / wl, e_spil, color="r", label="Spillover")
    plt.plot(h_list / wl, e_illu, color="g", label="Illumination")
    plt.plot(h_list / wl, e_antenna, color="b", label="Antenna")
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


def find_best_q(file_path):
    data = pd.read_table(file_path, sep="\s+").values
    pattern = data[:, 2] - np.max(data[:, 2])
    ang = np.linspace(-180, 180, 361, dtype=int).reshape([361, 1])
    mag = np.concatenate((pattern[::-1], pattern[1::])).reshape([361, 1])

    plt.figure()
    plt.plot(ang, mag, color="r", label="CST")

    q_range = np.arange(start=1, stop=20, step=0.5)
    loss = np.zeros([len(q_range), 1])
    cos_model = np.zeros([361, len(q_range)])
    with np.errstate(divide='ignore'):
        for q in q_range:
            cos_x = np.cos(ang * np.pi / 180)
            cos_value = 10 * np.log10(np.power(cos_x, q * 2))
            index = np.where(q_range == q)[0].item()
            cos_model[:, index] = cos_value.flat
            # plt.plot(ang, cos_model)
            loss[index] = 1 / 61 * np.sum(np.abs(cos_value[150:211] - mag[150:211]))

    plt.xlim(-90, 90)
    plt.ylim(-30, 10)
    plt.xlabel("Theta")
    plt.ylabel("Mag(dB)")
    plt.title("Power")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(q_range, loss)
    plt.xlabel("Theta")
    plt.ylabel("Mag(dB)")
    plt.xlim(0, 20)
    plt.title("Loss of cos-q Model")
    plt.grid()
    plt.show()

    q_best_id = np.where(loss == np.min(loss))[0].item()
    q_best = q_range[q_best_id]
    print("Best q is {:.1f}".format(q_best))
    return q_best


if __name__ == "__main__":
    # q_best = find_best_q(r"../../data/feed/horn_pattern.txt")

    # h = aperture_efficiency(wl, x, y, q)

    # h = 100 * wl

    # # P69
    # phase_array = phase_distribution(wl=3e8 / 32e9,
    #                                  feed_position=[0, 0, 170 / 1e3],
    #                                  unit_len=4.7 / 1e3,
    #                                  unit_num=40,
    #                                  beam_theta=0,
    #                                  beam_phi=0)
    #
    # # P71
    # phase_array = phase_distribution(wl=3e8 / 32e9,
    #                                  feed_position=[-85 / 1e3, 0, 147.22 / 1e3],
    #                                  unit_len=4.7 / 1e3,
    #                                  unit_num=40,
    #                                  beam_theta=30,
    #                                  beam_phi=0)
    #
    # P337
    phase_array = phase_distribution(wl=3e8 / 14.25e9,
                                     feed_position=[-91.88 / 1e3, 0, 342.9 / 1e3],
                                     unit_len=10 / 1e3,
                                     unit_num=36,
                                     beam_theta=15,
                                     beam_phi=0)

    # df = pd.DataFrame(columns=tt_name, data=tt_set)
    # df.to_csv(f'../data/dataset/tt_set.csv', encoding='utf-8', index=False)
