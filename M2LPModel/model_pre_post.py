import os
from time import time

import pandas as pd
import numpy as np

if __name__ == "__main__":
    csv_num = len(os.listdir('../Data/s11'))
    tr_name = ["a", "h", "e", "angle"]
    tr_set = np.zeros([csv_num, 4], dtype=float)
    tr_set[:, 0:3] = pd.read_csv(f'../Data/dataset/samples.csv', header=None, usecols=[1, 2, 3], nrows=csv_num).values
    time_start = time()
    for csv_id in range(csv_num):
        tr_set[csv_id, 3] = pd.read_csv(f'../Data/s11/s11_{csv_id}.csv', header=0,
                                        usecols=[2], skiprows=625, nrows=1).values
        time_end = time()
        print(f'ID={csv_id}, RunTime={time_end - time_start}, AvgTime ={(time_end - time_start) / (csv_id + 1)}')

    tr_set_num = int(csv_num)
    tr_set = tr_set[0:tr_set_num]
    df = pd.DataFrame(columns=tr_name, data=tr_set)
    df.to_csv(f'../Data/dataset/tr_set.csv', encoding='utf-8', index=False)

    del df
    tt_name = ["a", "h", "e"]
    tt_set = pd.read_csv(r"../Data/dataset/space.csv", header=0, engine="c").values
    tt_set = tt_set[:,1:]
    df = pd.DataFrame(columns=tt_name, data=tt_set)
    df.to_csv(f'../Data/dataset/tt_set.csv', encoding='utf-8', index=False)
