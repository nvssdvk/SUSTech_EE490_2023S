import os
from time import time

import pandas as pd
import numpy as np


def proc_main(s11_path, sample_path, save_path):
    csv_num = len(os.listdir(s11_path))
    tr_name = ["a", "h", "e", "angle"]
    tr_set = np.zeros([csv_num, 4], dtype=float)
    tr_set[:, 0:3] = pd.read_csv(sample_path, header=None, usecols=[1, 2, 3], nrows=csv_num).values
    time_start = time()
    for csv_id in range(csv_num):
        temp_set = pd.read_csv(f'{s11_path}/s11_{csv_id}.csv', header=0, usecols=[2]).values
        tr_set[csv_id, 3] = temp_set[int(len(temp_set) / 2)]
        # tr_set[csv_id, 3] = pd.read_csv(f'../data/s11/s11_{csv_id}.csv', header=0,
        #                                 usecols=[2], skiprows=1749, nrows=1).values
        time_end = time()
        print(f'ID={csv_id}, RunTime={time_end - time_start}, AvgTime ={(time_end - time_start) / (csv_id + 1)}',
              end='\r')

    tr_set_num = int(csv_num)
    tr_set = tr_set[0:tr_set_num]
    df = pd.DataFrame(columns=tr_name, data=tr_set)
    df.to_csv(save_path, encoding='utf-8', index=False)


if __name__ == "__main__":
    proc_main(f'../data/s11_tr', f'../data/dataset/samples_tr.csv', f'../data/dataset/tr_set.csv')
    proc_main(f'../data/s11_ve', f'../data/dataset/samples_ve.csv', f'../data/dataset/ve_set.csv')

    tt_name = ["a", "h", "e"]
    tt_set = pd.read_csv(r"../data/dataset/space.csv", header=0, engine="c").values
    tt_set = tt_set[:, 1:]
    df = pd.DataFrame(columns=tt_name, data=tt_set)
    df.to_csv(f'../data/dataset/tt_set.csv', encoding='utf-8', index=False)
