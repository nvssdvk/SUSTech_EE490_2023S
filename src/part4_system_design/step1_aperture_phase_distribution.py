import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def phase_unwrap(phi_wrap_arr):
    len_x, len_y = phi_wrap_arr.shape[0], phi_wrap_arr.shape[1]
    row_center, col_center = int(phi_wrap_arr.shape[0] / 2), int(phi_wrap_arr.shape[1] / 2)
    c = phi_wrap_arr[-1, -1]
    phi_unwrap_arr_sub = phi_wrap_arr[row_center:, col_center:]
    for i in range(phi_unwrap_arr_sub.shape[0]):
        phi_unwrap_arr_sub[i, :] -= 360
        for j in range(1, len(phi_unwrap_arr_sub[i])):
            if phi_unwrap_arr_sub[i, j - 1] - phi_unwrap_arr_sub[i, j] > 100:
                phi_unwrap_arr_sub[i, j:] -= 360

    phi_unwrap_arr_sub_lr = np.fliplr(phi_unwrap_arr_sub)
    phi_unwrap_arr_sub_h = np.hstack((phi_unwrap_arr_sub_lr, phi_unwrap_arr_sub))
    phi_unwrap_arr_sub_ud = np.flipud(phi_unwrap_arr_sub_h)
    phi_unwrap_arr = np.vstack((phi_unwrap_arr_sub_ud, phi_unwrap_arr_sub_h))

    print(phi_unwrap_arr.shape)  # 输出(21, 21)
    # row_sub_wrap = phi_wrap_arr_sub[0, :]
    # row_sub_unwrap = row_sub_wrap.copy()
    # row_sub_unwrap -= 360
    # for i in range(1, len(row_sub_unwrap)):
    #     if row_sub_unwrap[i - 1] - row_sub_unwrap[i] > 100:
    #         row_sub_unwrap[i:] -= 360
    cnt = 0

    # cnt = 0
    # phi_last = 0
    # for i in range(int(len_x/2), 1):
    #     phi_last = phi_wrap_arr[i, int(len_y/2)]

    # for i in range(len_x):
    #     for j in range(len_y):

    return phi_wrap_arr


def phase_wrap(phi_arr):
    num = int((np.max(phi_arr) + 360) / 360)
    phi_arr -= num * 360
    phi_arr[phi_arr < -750] += 360 * 3
    phi_arr[phi_arr > -53] -= 360
    return phi_arr


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
    x_arr = np.arange(-unit_num / 2 * dx + unit_len / 2, unit_num / 2 * dx + unit_len / 2, dx)
    y_arr = np.arange(-unit_num / 2 * dy + unit_len / 2, unit_num / 2 * dy + unit_len / 2, dy)
    xx, yy = np.meshgrid(x_arr, y_arr)

    plt.figure(figsize=(19.2, 10.8))

    plt.subplot(221)
    xspd = xx - feed_position[0]  # vectorize calculation of x and y
    yspd = yy - feed_position[1]
    z = feed_position[2]
    phi_arr_spd = -k * np.sqrt(xspd ** 2 + yspd ** 2 + z ** 2)
    phi_arr_spd = phi_arr_spd * 180 / np.pi
    plt.imshow((shrink(phi_arr_spd)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [part1_unit_design number]")
    plt.ylabel("y-axis [part1_unit_design number]")
    plt.title("Spatial Delay")

    plt.subplot(222)
    cos_phi = np.cos(beam_phi)
    sin_phi = np.sin(beam_phi)
    phi_arr_pp = -k * (xx * cos_phi + yy * sin_phi) * np.sin(beam_theta)
    phi_arr_pp = phi_arr_pp * 180 / np.pi
    plt.imshow((shrink(phi_arr_pp)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [part1_unit_design number]")
    plt.ylabel("y-axis [part1_unit_design number]")
    plt.title("Progressive Phase")

    plt.subplot(223)
    phi_arr = -phi_arr_spd + phi_arr_pp
    # phi_arr = shrink(phi_arr)
    phi_arr = phase_wrap(phi_arr)
    plt.imshow(shrink(phi_arr), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("x-axis [part1_unit_design number]")
    plt.ylabel("y-axis [part1_unit_design number]")
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

    plt.savefig(r'../../img/system/aperture_phase_distribution.png')
    plt.show()

    df_name = ['row', 'col', 'phase']
    df_data = np.zeros([unit_num * unit_num, 3])
    df_id = 0
    for i in range(unit_num):
        for j in range(unit_num):
            df_data[df_id, 0] = i
            df_data[df_id, 1] = j
            df_data[df_id, 2] = phi_arr[i, j]
            df_id += 1

    df = pd.DataFrame(columns=df_name, data=df_data)
    df.to_csv(r'../../data/dataset/aperture_dist.csv', encoding='utf-8', index=False)

    return phi_arr


if __name__ == "__main__":
    # P69
    # phase_array = phase_distribution(wl=3e8 / 32e9,
    #                                  feed_position=[0, 0, 170 / 1e3],
    #                                  unit_len=4.7 / 1e3,
    #                                  unit_num=40,
    #                                  beam_theta=0,
    #                                  beam_phi=0)

    # P71
    # phase_array = phase_distribution(wl=3e8 / 32e9,
    #                                  feed_position=[-85 / 1e3, 0, 147.22 / 1e3],
    #                                  unit_len=4.7 / 1e3,
    #                                  unit_num=40,
    #                                  beam_theta=30,
    #                                  beam_phi=0)

    # P337
    # phase_array = phase_distribution(wl=3e8 / 14.25e9,
    #                                  feed_position=[-91.88 / 1e3, 0, 342.9 / 1e3],
    #                                  unit_len=10 / 1e3,
    #                                  unit_num=36,
    #                                  beam_theta=15,
    #                                  beam_phi=0)

    # test1, beam_theta=0
    # wl = 3e8 / 10e9
    # phase_array = phase_distribution(wl=wl,
    #                                  feed_position=[0, 0, 9.5 * wl],
    #                                  unit_num=21,
    #                                  beam_theta=0,
    #                                  beam_phi=0)

    # test2, beam_theta=15
    wl = 3e8 / 10e9
    phase_array = phase_distribution(wl=wl,
                                     feed_position=[-9.5 * wl * np.sin(15), 0, 9.5 * wl * np.sin(15)],
                                     unit_num=21,
                                     beam_theta=15,
                                     beam_phi=0)
