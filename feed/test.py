import os
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gradient_descent(f, x0, e):
    # f: 目标函数
    # x0: 初始值
    # e: 收敛条件
    alpha = 0.01 # 学习率
    x = x0 # 当前值
    while True:
        grad = f(x) * 2 * x # 计算梯度
        x_new = x - alpha * grad # 更新当前值
        if abs(x_new - x) < e: # 满足收敛条件
            break # 结束循环
        else:
            x = x_new # 更新当前值
    return x_new # 返回最优值

# 测试函数：f(x) = x**2 - 10*x + 25，最小值点为x=5



if __name__ == "__main__":
    a = 2
    b = 8/2**3