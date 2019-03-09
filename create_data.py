import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
import random
import math


def gauss_data():
    data1 = np.random.randn(200, 2)/3
    data2 = np.random.randn(200, 2)/3 + 2.
    return np.append(data1, data2, axis=0)


def point_cycle(point_num, radius, x_offset=0., y_offset=0.):
    arr = np.empty((point_num, 2))
    for i in range(point_num):
        theta = random.random() * 2 * np.pi
        r = random.uniform(0, radius)
        x = math.sin(theta) * (r ** 0.5) + x_offset
        y = math.cos(theta) * (r ** 0.5) + y_offset
        # plt.plot(x, y, '.', color="black")
        arr[i, 0] = x
        arr[i, 1] = y
    return arr


def point_line(point_num, height, wide, x_offset, y_offset):
    arr = np.empty((point_num, 2))
    for i in range(0, point_num):
        x = random.uniform(0, wide)+x_offset
        y = random.uniform(0, height)+y_offset
        # plt.plot(x, y, '.', color="red")
        arr[i, 0] = x
        arr[i, 1] = y
    return arr


def circles_data():
    data1, _ = make_circles(n_samples=200, shuffle=True, noise=0.05, random_state=None, factor=0.9)
    data2 = point_cycle(100, 0.2,)
    data3 = np.concatenate((data1, data2), axis=0)
    return data3


def moon_data(N=100, d=2, r=10, w=2):
    N1 = 10 * N
    w2 = w / 2
    done = True
    data = np.empty(0)
    while done:
        # generate Rectangular data
        tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y = (r + w2) * np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
        # generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    # print(data)
    db_moon = data[0:N, :]
    # print(db_moon)
    # generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon


def get_iris():
    # read the first 4 columns
    data = np.genfromtxt(".\\iris.csv", delimiter=',', usecols=(0, 1, 2, 3))
    label = np.zeros((150,), dtype=int)
    label[0:50] = 1
    label[50:100] = 2
    label[100:150] = 3
    # print(label)
    return data, label


def get_wine():
    data = np.genfromtxt(".\\wine.csv", delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    label = np.zeros((178,), dtype=int)
    label[0:59] = 1
    label[59:130] = 2
    label[130:178] = 3
    # print(label)
    return data, label


def get_seeds():
    data = np.genfromtxt(".\\seed.csv", delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6))
    label = np.zeros((210,), dtype=int)
    label[0:70] = 1
    label[70:140] = 2
    label[140:210] = 3
    # print(label)
    return data, label


def boon_data():
    R = 1
    arr1 = point_cycle(150, R, -2, 0)
    arr2 = point_cycle(150, R, 2, 0)
    arr3 = point_line(5, 0.05, 1.1, -1, 0)
    arr4 = point_line(5, 0.05, 1, -0.1, 0)
    arr = np.concatenate((arr1, arr2, arr3, arr4), axis=0)
    label = np.zeros((310,), dtype=int)
    label[0:150] = 1
    label[150:300] = 2
    label[300:305] = 1
    label[305:310] = 2
    return arr, label


def hand_data():
    noise = np.random.random((200, 2))
    data = np.zeros((200, 2))
    label = np.ones((200,))
    label[100:200] = 2
    theta1 = np.linspace(0.3* np.pi, 2 * np.pi, 100)
    r = np.linspace(0.03, 2, 100)
    x1 = r * np.sin(theta1)
    y1 = r * np.cos(theta1)
    theta2 = np.linspace(1.3 * np.pi, 3 * np.pi, 100)
    x2 = r * np.sin(theta2)
    y2 = r * np.cos(theta2)
    # for i in range(100):
    #     data[i, 0] = x[i]
    #     data[i, 1] = y[i]
    data[0:100, 0] = x1
    data[0:100, 1] = y1 + 0.18
    data[100:200, 0] = x2
    data[100:200, 1] = y2 - 0.18
    data = data + noise*0.1
    return data, label


if __name__ == '__main__':
    np.random.seed(10)
    # a = gauss_data()
    # print(a.shape)
    # n, d, r, w = 150, -2, 10, 5
    data, _ = get_seeds()
    #
    # print(data.shape, type(data))
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], '.', color="black")
    plt.show()
    # c = np.arange(0.1, 0.5, 0.1)
    # a=np.ones((2,2))
    # b=np.zeros((2,2))
    # print(np.concatenate((a,b,a),axis=0))
    pass
