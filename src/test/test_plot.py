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


if __name__ == "__main__":
    pass

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
