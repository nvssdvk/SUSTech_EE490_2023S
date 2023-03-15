
import numpy as np

if __name__ == "__main__":
    a = np.random.randint(0 ,9 ,[4 ,3])
    a_min = np.min(a, axis=0)
    a_max = np.max(a, axis=0)

    b = (a - a_min) / (a_max - a_min)
