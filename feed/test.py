import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



if __name__ == "__main__":
    wave_len = 0.3 / 10
    # x = np.arange(start=(-150 + 7.5) / 1e3, stop=150 / 1e3, step=15 / 1e3)
    # y = np.arange(start=(-150 + 7.5) / 1e3, stop=150 / 1e3, step=15 / 1e3)
    x = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)
    y = np.arange(start=(-900 + 7.5) / 1e3, stop=900 / 1e3, step=15 / 1e3)


    plt.figure()
    plt.plot(h_list / wave_len, value, color="r", label="CST")
    plt.show()
