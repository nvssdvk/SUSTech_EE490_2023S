import numpy as np
import pandas as pd

if __name__ == "__main__":
    a = np.random.randint(20, size=(15, 1, 1))
    b = np.zeros([15, 2])
    c = a[:,0,0]
    b[:, -1] = a[:,0,0]
