import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_hotmap(arr):
    x = arr[:, 0]
    y = arr[:, 1]

    matrix = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            col = x[j]
            row = y[i]
            temp_arr = arr[(arr[:, 0] == x[j]) & (arr[:, 1] == y[i])]
            matrix[i, j] = temp_arr[0, 2]

    # 绘制热图
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='coolwarm')

    # 添加网格
    ax.grid(True, which='both', color='gray', linewidth=1)

    # 设置坐标轴和标题
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('2D Heatmap')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 显示图形
    plt.show()


def plot_hot_map(data):
    # 从数据中提取行、列和数值信息
    rows = np.arange(0, len(data[:, 0]), 1)
    cols = np.arange(0, len(data[:, 1]), 1)
    values = data[:, 2]

    # 创建一个空的矩阵用于填充热图
    heatmap = np.zeros((np.max(rows) + 1, np.max(cols) + 1))

    # 使用数值信息填充热图矩阵
    heatmap[rows, cols] = values

    # 绘制热图
    plt.figure(figsize=(12.8, 7.2))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    wrap_path = r'../data/dataset/tr_set.csv'
    unwrap_path = r'../data/dataset/ve_set_unwrap.csv'

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
                if ((phase_unwrap_at_e[i] < -100) & (phase_unwrap_at_e[i + 1] > 0)) or \
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

    # df_name = ["a", "h", "e", "phase"]
    # df_data = data
    # df = pd.DataFrame(columns=df_name, data=df_data)
    # df.to_csv(unwrap_path, encoding='utf-8', index=False)

#
# for h in para_h:
#     plt.figure(figsize=(12.8, 7.2))
#     for e in para_e[0:5]:
#         data_copy_at_h = data[(data[:, 2] == e) & (data[:, 1] == h)]
#         a = data_copy_at_h[:, 0]
#         pha = data_copy_at_h[:, 3]
#         plt.scatter(a, pha, label='e={:.2f}'.format(e))
#     plt.xlim(2, 14)
#     plt.ylim(-180, 180)
#     plt.xlabel('Upper Surface Side Length')
#     plt.ylabel('Phase')
#     plt.title('h={:.1f}'.format(h))
#     plt.legend()
#     plt.savefig('../img/h={:.1f}.png'.format(h))
#     plt.show()

# for h in para_h[0:5]:
#     for e in para_e[0:5]:
#         data_copy_at_h = data[(data[:, 2] == e) & (data[:, 1] == h)]
#         a = data_copy_at_h[:, 0]
#         pha = data_copy_at_h[:, 3]
#
#         plt.figure(figsize=(12.8,7.2))
#         plt.scatter(a, pha)
#         plt.xlim(2, 14)
#         plt.ylim(-180, 180)
#         plt.xlabel('Upper Surface Side Length')
#         plt.ylabel('Phase')
#         plt.title('h={:.1f},e={:.2f}'.format(h,e))
#         plt.show()

# for e in para_e:
#     data_copy_at_h = data[(data[:, 2] == e)]
#     x = data_copy_at_h[:, 0].flatten()
#     y = data_copy_at_h[:, 1].flatten()
#     z = data_copy_at_h[:, 3].flatten()
#     c = data_copy_at_h[:, 3].flatten()
#     plt.figure(figsize=(12.8, 7.2))
#     plt.tricontourf(x, y, z, levels=len(z), cmap='coolwarm', vmin=-180, vmax=180)
#     plt.colorbar()
#     plt.xlabel('Upper Surface Side Length')
#     plt.ylabel('Height')
#     plt.title('Wrap Phase with Dielectric Constant of {:.2f}'.format(e))
#     plt.show()
