import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_best_q(file_path):
    data = pd.read_table(file_path, sep="\s+").values
    pattern = data[:, 2] - np.max(data[:, 2])
    ang = np.linspace(-180, 180, 361, dtype=int).reshape(-1, 1)
    mag = np.concatenate((pattern[::-1], pattern[1::])).reshape(-1, 1)

    plt.figure(figsize=(19.2, 7.2))
    plt.subplot(121)
    plt.plot(ang, mag, color="r", label="CST")

    q_range = np.arange(start=100, stop=300, step=0.5)
    loss = np.zeros([len(q_range), 1])
    cos_model = np.zeros([361, len(q_range)])
    with np.errstate(divide='ignore'):
        for q in q_range:
            cos_x = np.cos(ang * np.pi / 180)
            cos_value = 10 * np.log10(np.power(cos_x, q * 2) + 1e-15)
            index = np.where(q_range == q)[0].item()
            cos_model[:, index] = cos_value.flat
            loss[index] = 1 / 61 * np.sum(np.abs(cos_value[177:183] - mag[177:183]))
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
    plt.title("Loss of cos-q Model")
    plt.grid()
    plt.savefig("../../img/system/q.png")
    plt.show()

    print("Best q is {:.1f}".format(q_best))
    return q_best


if __name__ == "__main__":
    q = find_best_q(r"../../data/dataset/reflectarray_pattern_10GHz.txt")

    '''
    Best q is 185
    '''
