import numpy as np
from cvxopt import solvers, matrix
import create_data as cd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method_sv_sort as ss
import method_snn as snn
import accuracy as ac


def svc(samples, c, q):
    num, _ = samples.shape
    # calculate kernel matrix
    k_matrix = kernel_matrix(samples, q)
    # solve lagrangian multipliers
    beta = solve_lagrangian(num, k_matrix, c)
    # find support vector
    sv, num_sv, bsv, num_bsv, un_sv = find_sv_bsv(samples, beta, c)
    # calculate radius of sphere
    quad, r = cal_sphere_r(samples, sv, beta, k_matrix, q)
    return sv, bsv, beta, quad, r, un_sv


def kernel_matrix(samples, q):
    dist = np.sum(np.square(samples), axis=1)
    dist = np.reshape(dist, [-1, 1])
    sq_dists = np.add(np.subtract(dist, np.multiply(2., np.matmul(samples, np.transpose(samples)))), np.transpose(dist))
    k_matrix = np.exp(np.multiply(-q, np.abs(sq_dists)))
    # print(type(k_matrix),k_matrix.shape,type(k_matrix[0,0]))
    return k_matrix


def r_ker_matrix(samples, pre_data, q):
    dist1 = np.sum(np.square(samples), axis=1)
    dist1 = np.reshape(dist1, [-1, 1])
    dist2 = np.sum(np.square(pre_data), axis=1)
    dist2 = np.reshape(dist2, [-1, 1])
    sq_dists = np.add(np.subtract(dist2, np.multiply(2., np.matmul(pre_data, np.transpose(samples)))), np.transpose(dist1))
    rk_matrix = np.exp(np.multiply(-q, np.abs(sq_dists)))
    return rk_matrix


def solve_lagrangian(num, k_matrix, c):
    k = np.multiply(2, k_matrix)
    P = matrix(k)
    q = matrix(-np.diag(k_matrix))
    G = matrix(np.reshape(np.append(np.eye(num), -np.eye(num), axis=0), [2 * num, num]))
    h = matrix(np.reshape(np.append(np.ones((num, 1)) * c, np.zeros((num, 1)), axis=0), [2 * num, 1]))
    A = matrix(np.ones((1, num)))
    b = matrix(np.ones((1, 1)))
    sol = solvers.qp(P, q, G, h, A, b)
    x = sol["x"]
    beta = np.reshape(x, [-1, 1])
    return beta


def find_sv_bsv(samples, beta, c):
    num, attr = samples.shape
    num_bsv, num_sv = 0, 0
    bsv, sv, un_sv = [], [], []
    for i in range(num):
        if beta[i] > c - 0.001:
            num_bsv += 1
            bsv.append(samples[i, :])
        elif beta[i] > 0.001:
            num_sv += 1
            sv.append(samples[i, :])
        else:
            un_sv.append(samples[i, :])
    return np.array(sv), num_sv, np.array(bsv), num_bsv, np.array(un_sv)


def cal_sphere_r(samples, sv, beta, k_matrix, q):
    quad = np.sum(np.multiply(np.multiply(beta, k_matrix), np.transpose(beta)))
    rk_matrix = r_ker_matrix(samples, sv, q)
    r_matrix = kernel_matrix(sv, q)
    distance = np.reshape(np.diag(r_matrix), [-1, 1]) - np.sum(np.multiply(2, np.multiply(np.transpose(beta), rk_matrix)), axis=1) + quad
    # print(distance)
    r = np.max(distance, axis=0)
    return quad, r


if __name__ == '__main__':
    # 获取数据
    np.random.seed(10)
    # 1.two gauss data
    # c, q = 1, 1.5
    # sample = cd.gauss_data()

    # 2.two clusters circle data
    # c, q = 1, 3.5
    # sample = cd.circles_data()

    # 3.moon
    c, q = 1, 5
    sample = cd.moon_data(150, -2, 10, 5)

    # 4.iris
    # c, q = 0.01, 10
    # # c, q = 0.01, 20
    # sample, true_label = cd.get_iris()
    # true_cluster_num = int(np.max(true_label))

    # 5.boon
    # c, q = 0.01, 3
    # sample, _ = cd.boon_data()

    # 6.hand
    # c, q = 0.05, 3
    # sample, _ = cd.hand_data()

    sample = preprocessing.scale(sample)
    num, attr = sample.shape  # num：数据数量。 attr：属性维度
    sv, bsv, beta, quad, r, un_sv = svc(sample, c, q)  # solve the first step of algorithm

    sv_matrix = kernel_matrix(sv, q)  # the matrix using for sort sv
    sv_unsv_matrix = r_ker_matrix(un_sv, sv, q)  # num_sv x num_un_sv kernel matrix
    sv_sample_matrix = r_ker_matrix(sample, sv, q)  # num_sv x num_sample kernel matrix

    # 支持向量点排序
    sv_order, distance_list = ss.point_sort(sv_matrix, sv)

    plt.figure()
    plt.plot(sample[:, 0], sample[:, 1], ".", color="green")
    plt.plot(sv[:, 0], sv[:, 1], "o", color="red")
    plt.plot(sv_order[:, 0], sv_order[:, 1],  color="black")
    plt.show()

    num_sv, _ = sv.shape
    # 1.
    k, e = int(np.floor(num / num_sv * 3)), 0
    # 2.
    # k, e =5, 0
    # 3.
    # k, e = int(np.floor(num / num_sv*3)), 2
    # k, e = 13, 12
    # 共享近邻
    sv_label, num_cluster = snn.cluster_sv(sv_unsv_matrix, k, e)
    sample_label = snn.cluster_sample(sample, sv_label, sv_sample_matrix)

    # accuracy = ac.accuracy_cluster(sample_label, num_cluster, true_label, true_cluster_num)
    # print(accuracy)
    snn.show_result(sample, sample_label, num_cluster)
