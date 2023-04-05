import numpy as np
import pandas as pd

if __name__ == "__main__":
    data_dir = f'../data/dataset/tr_set.csv'
    data = pd.read_csv(data_dir, header=0, engine="c").values
    phase = data[:, -1].reshape([len(data), 1])
    unwrapped_phase = np.unwrap(np.deg2rad(phase))
    unwrapped_phase = np.rad2deg(unwrapped_phase)