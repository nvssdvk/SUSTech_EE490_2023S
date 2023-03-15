import os
from time import time

import pandas as pd
import numpy as np

if __name__ == "__main__":
    csv_num = len(os.listdir('../data/s11'))
    # print(csv_num)

    df_name = ["a", "h", "e", "angle"]
    df_data = np.zeros([csv_num, 4], dtype=float)

    time_start = time()

    for csv_id in range(csv_num):
        if csv_id == 0:
            df_data[0, 0:3] = pd.read_csv(f'../data/lhs/samples.csv',
                                          header=None, engine="c", usecols=[1, 2, 3], nrows=1).values
        else:
            df_data[csv_id, 0:3] = pd.read_csv(f'../data/lhs/samples.csv',
                                               header=None, engine="c", usecols=[1, 2, 3], skiprows=csv_id,
                                               nrows=1).values
        df_data[csv_id, 3] = pd.read_csv(f'../data/s11/s11_{csv_id}.csv',
                                         header=0, engine="c", usecols=[2], skiprows=625, nrows=1).values
        time_end = time()
        print(f'ID={csv_id}, RunTime={time_end - time_start}, AvgTime ={(time_end - time_start) / (csv_id + 1)}')

    df = pd.DataFrame(columns=df_name, data=df_data)
    df.to_csv(f'../data/dataset/dataset.csv', encoding='utf-8', index=False)

    test_num = 500
    tr_set_num = int(csv_num - test_num)

    tr_data = df_data[0:tr_set_num]
    df_tr = pd.DataFrame(columns=df_name, data=tr_data)
    df_tr.to_csv(f'../data/dataset/tr_set.csv', encoding='utf-8', index=False)

    test_data = df_data[(csv_num - test_num):csv_num]
    df_test = pd.DataFrame(columns=df_name, data=test_data)
    df_test.to_csv(f'../data/dataset/tt_set.csv', encoding='utf-8', index=False)
