import numpy as np
from itertools import combinations, permutations
from sklearn import metrics


def accuracy_cluster(sample_label, cluster_num, true_label, true_num):
    sample_num, = sample_label.shape
    # first step : calculate matrix which [i,j] means how many same data in the two labels
    same_matrix = np.zeros((true_num, cluster_num))
    for i in range(true_num):
        for j in range(cluster_num):
            true_index, = np.where(true_label == i+1)
            sample_index, = np.where(sample_label == j+1)
            same_matrix[i, j] = len(set(true_index).intersection(set(sample_index)))
    # second step : accord to the matrix calculate in the last step , find the max match situation
    situation_list = list(permutations(range(cluster_num), true_num))
    # situation_list = list(permutations(range(3), 3))
    accuracy = 0
    for s in situation_list:
        correct_num = 0
        for i in range(true_num):
            correct_num += same_matrix[i, s[i]]
        accuracy = correct_num if correct_num>accuracy else accuracy
    return accuracy/sample_num


#  Adjusted Rand index 调整兰德系数
def accuracy_ari(sample_label, true_label):
    return metrics.adjusted_rand_score(sample_label, true_label)


#  Adjusted Mutual Information
def accuracy_ami(sample_label, true_label):
    return metrics.adjusted_mutual_info_score(sample_label, true_label)


#  Fowlkes-Mallows score FMI
def accuracy_mfi(sample_label, true_label):
    return metrics.fowlkes_mallows_score(sample_label, true_label)


if __name__ == '__main__':
    a = np.arange(0, 5)
    b = np.arange(3, 6)
    c = np.zeros((3, 3))
    d, = np.where(c[1, :] == 0)
    e = list(permutations(range(2), 2))
    m, n = 1, 2
    m = n if m < n else m
    l1 = np.array([1, 1, 2, 2])
    l2 = np.array([1, 1, 1, 2])
    print(accuracy_cluster(l1, 2, l2))
