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

    plt.savefig(r'../../img/system/aperture_phase_distribution.png')
    plt.show()


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

    # test
    phase_array = phase_distribution(wl=3e8 / 10e9,
                                     feed_position=[0, 0, 0.255],
                                     unit_num=20,
                                     beam_theta=0,
                                     beam_phi=0)
