import numpy as np
import matplotlib.pyplot as plt
import copy


def cluster_sv(sv_unsv_matrix, k, ee):
    adj_matrix = connect_matrix(sv_unsv_matrix, k, ee)
    # print(ee, np.sum(adj_matrix))
    num_cluster_sv = 0  # count of clusters
    num, attr = adj_matrix.shape
    sv_label = np.zeros((num,), dtype=int)
    done = 0  # 判断是否完成
    while done != 1:
        root = 0  # 深度优先搜索的根节点
        while sv_label[root] != 0:
            root += 1
            if root == num:
                done = 1
                break
        if done != 1:
            num_cluster_sv += 1  # 发现新簇
            stack = [root]  # 栈
            while len(stack) != 0:
                node = stack.pop()  # 出栈
                sv_label[node] = num_cluster_sv
                for i in range(num):
                    if adj_matrix[node, i] == 1 and sv_label[i] == 0 and i != node:
                        stack.append(i)
    return sv_label, num_cluster_sv


def cluster_sample(sample, sv_label, sv_sample_matrix):
    num, _ = sample.shape
    sample_label = np.zeros((num,), dtype=int)
    for i in range(num):
        index = np.where(sv_sample_matrix[:, i] == np.max(sv_sample_matrix[:, i]))
        index1 = index[0]
        sample_label[i] = sv_label[index1[0]]
    return sample_label


def connect_matrix(sv_unsv_matrix, k, e):
    num, _ = sv_unsv_matrix.shape
    adj_matrix = np.zeros((num, num), dtype=int)
    top_k_label = top_k(sv_unsv_matrix, k)
    for i in range(num):
        for j in range(i + 1, num):
            shared_num = equal_num(top_k_label[i, :], top_k_label[j, :])
            if shared_num > e:
                adj_matrix[i, j], adj_matrix[j, i] = 1, 1
    return adj_matrix


def equal_num(array1, array2):
    array1 = np.reshape(array1, [-1, 1])
    array2 = np.reshape(array2, [-1, 1])
    array_result = np.add(array1, array2)
    num, _ = array_result.shape
    count = 0
    for i in range(num):
        if array_result[i] == 2:
            count += 1
    return count


def top_k(matrix, k):  # 求每个sv最近的k个值 返回num_sv x nun_unsv 的matrix。1代表是sv的最近k个点
    sv_unsv_matrix = copy.deepcopy(matrix)
    num1, num2 = sv_unsv_matrix.shape
    top_k_label = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(k):
            sv_unsv_matrix[i, np.where(sv_unsv_matrix[i, :] == np.max(sv_unsv_matrix[i, :]))] = -1
    top_k_label[sv_unsv_matrix == -1] = 1
    return top_k_label


def show_result(samples, label, num_cluster):
    color_list = ["red", "green", "blue", "yellow", "black"]
    plt.figure()
    for i in range(num_cluster):
        cluster = samples[label == i + 1, :]
        color = color_list[i % len(color_list)]
        plt.plot(cluster[:, 0], cluster[:, 1], ".", color=color)
    plt.show()


if __name__ == '__main__':
    # i = [4,1,2]
    # a = np.zeros((5,5))
    # a[1,i] = 1
    # b = np.arange(5)
    # c = np.zeros((5,5))
    # c[1, a[1] == 0] = 1
    # print(np.where(np.reshape(c, [-1,1])==0))
    # a = np.reshape(np.arange(4), [2,2])
    # print(np.where(a == [0,3]))
    # a = np.reshape(np.arange(16), [4,4])
    # print(top_k(a,2))
    # print(c[1,:].shape)
    list1 = [1,2,3,4,5,6]
    a = list1.pop()
    print(list1,a)
    pass
