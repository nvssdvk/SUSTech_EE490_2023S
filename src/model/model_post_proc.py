import os
from time import time

import pandas as pd
import numpy as np

if __name__ == "__main__":
    tt_name = ["a", "h", "e"]
    tt_set = pd.read_csv(r"../../data/dataset/space.csv", header=0, engine="c").values
    tt_set = tt_set[:, 1:]
    df = pd.DataFrame(columns=tt_name, data=tt_set)
    df.to_csv(r'../../data/dataset/tt_set.csv', encoding='utf-8', index=False)
