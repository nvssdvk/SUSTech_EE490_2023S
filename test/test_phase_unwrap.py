import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import ndimage


def phase_unwrap(wrap_path, unwrap_path):
    data = pd.read_csv(wrap_path, header=0, engine="c").values

    para_h = np.unique(data[:, 1])
    para_e = np.unique(data[:, 2])

    for e in para_e:
        data_copy_at_e = data[(data[:, 2] == e)]
        sort_index = np.lexsort((data_copy_at_e[:, 0], data_copy_at_e[:, 1]))
        data_copy_at_e = data_copy_at_e[sort_index]

        # 对每个h内的不同a展开相位
        len_h = np.zeros(len(para_h), dtype=int)
        h_end_id = np.zeros(len(para_h), dtype=int)
        for i in range(len(para_h)):
            h = para_h[i]
            data_copy_at_h = data_copy_at_e[data_copy_at_e[:, 1] == h]
            phase_unwrap_at_h = data_copy_at_h[:, 3]
            len_h[i] = len(phase_unwrap_at_h)
            if i == 0:
                h_end_id[i] = (len(phase_unwrap_at_h) - 1)
            else:
                h_end_id[i] = (h_end_id[i - 1] + len(phase_unwrap_at_h))

            for j in range(data_copy_at_h.shape[0] - 1):
                if (phase_unwrap_at_h[j] < -50) & (phase_unwrap_at_h[j + 1] > 30):
                    phase_unwrap_at_h[j + 1] -= 360
            data_copy_at_h[:, 3] = phase_unwrap_at_h
            data_copy_at_e[data_copy_at_e[:, 1] == h] = data_copy_at_h

        # 对不同的h之间展开相位
        phase_unwrap_at_e = data_copy_at_e[:, 3]
        cnt = 0
        for i in range(len(phase_unwrap_at_e) - 1):
            if i == h_end_id[cnt]:
                if ((phase_unwrap_at_e[i] < -100) & (phase_unwrap_at_e[i + 1] > 50)) or \
                        (phase_unwrap_at_e[i + 1] - phase_unwrap_at_e[i] > 200):
                    if (phase_unwrap_at_e[i + 1] - phase_unwrap_at_e[i] < 540):
                        for j in range(len_h[cnt + 1]):
                            phase_unwrap_at_e[i + 1 + j] -= 360
                    elif (phase_unwrap_at_e[i + 1] - phase_unwrap_at_e[i] < 900):
                        for j in range(len_h[cnt + 1]):
                            phase_unwrap_at_e[i + 1 + j] -= 720
                cnt += 1
        data_copy_at_e = data_copy_at_e[data_copy_at_e[:, 3] == phase_unwrap_at_e]

        data[(data[:, 2] == e)] = data_copy_at_e

    df_name = ["a", "h", "e", "phase"]
    df_data = data
    df = pd.DataFrame(columns=df_name, data=df_data)
    df.to_csv(unwrap_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    phase_unwrap(wrap_path=r'../data/dataset/tr_set.csv',
                 unwrap_path=r'../data/dataset/tr_set_unwrap.csv')
    phase_unwrap(wrap_path=r'../data/dataset/ve_set.csv',
                 unwrap_path=r'../data/dataset/ve_set_unwrap.csv')
