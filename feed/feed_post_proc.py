import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def find_best_q():
    data = pd.read_table(r"../data/feed/e-pattern.txt", sep="\s+").values
    pattern = data[:, 2] - np.max(data[:, 2])
    ang = np.linspace(-180, 180, 361, dtype=int).reshape([361, 1])
    mag = np.concatenate((pattern[::-1], pattern[1::])).reshape([361, 1])

    plt.figure()
    plt.plot(ang, mag, color="r", label="CST")

    q_range = np.arange(start=1, stop=20, step=0.1)
    loss = np.zeros([len(q_range), 1])
    cos_model = np.zeros([361, len(q_range)])
    with np.errstate(divide='ignore'):
        for q in q_range:
            cos_x = np.cos(ang * np.pi / 180)
            cos_value = 10 * np.log10(np.power(cos_x, q * 2))
            index = np.where(q_range == q)[0].item()
            cos_model[:, index] = cos_value.flat
            # plt.plot(ang, cos_model, label="q={:.1f}".format(q))
            loss[index] = 1 / 61 * np.sum(np.abs(cos_value[150:211] - mag[150:211]))

    # plt.xlim(-90, 90)
    # plt.ylim(-30, 10)
    # plt.xlabel("Theta")
    # plt.ylabel("Mag(dB)")
    # plt.title("E-Pattern")
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(q_range, loss)
    # plt.xlabel("Theta")
    # plt.ylabel("Mag(dB)")
    # plt.xlim(0, 20)
    # plt.title("Loss of cos-q Model")
    # plt.grid()
    # plt.show()

    q_best_id = np.where(loss == np.min(loss))[0].item()
    q_best = q_range[q_best_id]
    print("Best q is {:.1f}".format(q_best))
    return q_best


if __name__ == "__main__":
    pass

    # df = pd.DataFrame(columns=tt_name, data=tt_set)
    # df.to_csv(f'../data/dataset/tt_set.csv', encoding='utf-8', index=False)
