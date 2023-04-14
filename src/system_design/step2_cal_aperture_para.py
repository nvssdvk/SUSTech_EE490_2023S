import os
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_score(phase, data):
    w_phase = 0.7
    w_weight = 0.3

    diff = np.abs(data[:, -1] - phase)
    # selected = data[diff < loss]

    closest_indexes = np.argpartition(diff, kth=10)[:10]  # 找到最小的10个误差对应的数组索引
    selected = data[closest_indexes]  # 获取这10个误差最小的数组元素

    phase_loss = np.abs(selected[:, -1] - phase).reshape(-1, 1)
    weight_loss = weight(selected[:, 0], selected[:, 1], selected[:, 2]).reshape(-1, 1)

    scores = w_phase * phase_loss + w_weight * weight_loss

    out = np.concatenate((selected, phase_loss), axis=1)
    out = np.concatenate((out, weight_loss), axis=1)
    out = np.concatenate((out, scores), axis=1)

    out = np.take_along_axis(out, np.argsort(out[:, -1]).reshape(-1, 1), axis=0)
    return out


def weight(a, h, e):
    density = 1.25  # g/cm**3
    b = 15
    vf = np.zeros_like(e)
    vf[e == 1.24] = 0.18
    vf[e == 2.25] = 0.7
    vf[e == 2.53] = 0.73
    vf[e == 2.72] = 1
    volume = 1 / 3 * (a ** 2 + a * b + b ** 2) * h
    weight = volume * density * vf / 1e3
    return weight


if __name__ == "__main__":
    data_pred = pd.read_csv(r"../../data/dataset/pred_set.csv", header=0, engine="c").values
    data_aperture = pd.read_csv(r"../../data/dataset/aperture_dist.csv", header=0, engine="c").values
    data_aperture_para = np.zeros(data_aperture.shape[0])
    unit_num = int(np.sqrt(data_aperture.shape[0]))
    df_data = np.zeros([unit_num * unit_num, 9])
    df_data[:, 0:3] = data_aperture

    phases = data_aperture[:, 2]
    id_data = 0
    for i in range(unit_num):
        for j in range(unit_num):
            temp_data = data_aperture[(data_aperture[:, 0] == i) & (data_aperture[:, 1] == j)]
            temp_score = get_score(temp_data[:, 2], data_pred)
            df_data[id_data, 3:6] = temp_score[0, -3:]
            df_data[id_data, -3:] = temp_score[0, 0:3]
            id_data += 1

    df_name = ['row', 'col', 'phase', 'phase_loss', 'weight', 'score', 'a', 'h', 'e']
    df = pd.DataFrame(columns=df_name, data=df_data)
    df.to_csv(r'../../data/dataset/aperture_para.csv', encoding='utf-8', index=False)
