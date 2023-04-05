import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_map(data):
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 3],
                         c=data[:, 3], cmap='viridis', s=20)

    # 添加轴标签和标题
    ax.set_xlabel('Upper Surface Side Length')
    ax.set_ylabel('Height')
    ax.set_zlabel('Phase')
    ax.set_title(f'Dielectric Constant')

    # 添加颜色图例
    cbar = plt.colorbar(scatter)
    cbar.ax.set_ylabel('Phase')

    # 显示图形
    plt.show()


def plot_hot_map(data):
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection='3d')
    # 从数据中提取行、列和数值信息
    rows = np.arange(0, len(data[:, 0]), 1)
    cols = np.arange(0, len(data[:, 1]), 1)
    values = data[:, 3]

    # 创建一个空的矩阵用于填充热图
    heatmap = np.zeros((np.max(rows) + 1, np.max(cols) + 1))

    # 使用数值信息填充热图矩阵
    heatmap[rows, cols] = values

    # 绘制热图
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.show()


def plot_3d_map(data):
    # 将数据分为三个变量表示 x、y、z 坐标和颜色值
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    c = data[:, 3]

    # 创建三维坐标系
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制颜色表示数值强度的散点图
    img = ax.scatter(x, y, z, c=c, cmap=plt.hot(), s=20)
    fig.colorbar(img)

    # 设置坐标轴标签
    ax.set_xlabel('Upper Surface Side Length')
    ax.set_ylabel('Height')
    ax.set_zlabel('Dielectric Constant')

    # 显示图像
    plt.show()


if __name__ == "__main__":
    data_dir = f'../data/dataset/tr_set.csv'
    data = pd.read_csv(data_dir, header=0, engine="c").values

    # temp_data = data[(data[:, 2] == 2.72) & (data[:, 1] == 20.0) & (data[:, 0] == 12.3)]
    # print(temp_data[:,3])

    # plot_3d_map(data)

    para_h = np.unique(data[:, 1])
    para_e = np.unique(data[:, 2])


    #
    # for h in para_h:
    #     plt.figure(figsize=(12.8, 7.2))
    #     for e in para_e[0:5]:
    #         temp_data = data[(data[:, 2] == e) & (data[:, 1] == h)]
    #         a = temp_data[:, 0]
    #         pha = temp_data[:, 3]
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
    #         temp_data = data[(data[:, 2] == e) & (data[:, 1] == h)]
    #         a = temp_data[:, 0]
    #         pha = temp_data[:, 3]
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
    #     temp_data = data[(data[:, 2] == e)]
    #     x = temp_data[:, 0].flatten()
    #     y = temp_data[:, 1].flatten()
    #     z = temp_data[:, 3].flatten()
    #     c = temp_data[:, 3].flatten()
    #     plt.figure(figsize=(12.8, 7.2))
    #     plt.tricontourf(x, y, z, levels=len(z), cmap='coolwarm', vmin=-180, vmax=180)
    #     plt.colorbar()
    #     plt.xlabel('Upper Surface Side Length')
    #     plt.ylabel('Height')
    #     plt.title('Wrap Phase with Dielectric Constant of {:.2f}'.format(e))
    #     plt.show()
    def height(x, y):
        out = np.zeros([len(x), len(y)])
        for i in range(len(x)):
            for j in range(len(y)):
                temp_data = data[(data[:, 0] == x[i]) & (data[:, 1] == y[j])]
                out[i, j] = temp_data[:, 3].flatten()
        return


    for e in para_e:
        temp_data = data[(data[:, 2] == e)]
        x = temp_data[:, 0].flatten()
        y = temp_data[:, 1].flatten()
        X, Y = np.meshgrid(x, y)
        Z = height(x, y)
        # z = temp_data[:, 3].flatten()
        # c = temp_data[:, 3].flatten()
        plt.figure(figsize=(12.8, 7.2))
        plt.contourf(X, Y, Z, levels=20, cmap='coolwarm', vmin=-180, vmax=180)
        plt.colorbar()
        plt.xlabel('Upper Surface Side Length')
        plt.ylabel('Height')
        plt.title('Wrap Phase with Dielectric Constant of {:.2f}'.format(e))
        plt.show()
