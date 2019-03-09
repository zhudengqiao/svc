import numpy as np
import matplotlib.pyplot as plt


def point_sort(sv_matrix, sv):
    num, attr = sv.shape
    finish_label = np.zeros((num, 1), dtype=int)
    sv_matrix = np.negative(sv_matrix)
    sv_matrix = sv_matrix + np.eye(num) * 1000000.0  # 与自己距离设置为无穷大
    sv_order = np.zeros((2, attr))  # list to store the order of sv
    distance_list = []  # 存放支持向量排序后的相关度
    row, col = np.where(sv_matrix == np.min(sv_matrix))
    index_right, index_left = row[0], col[0]
    # sv_order.insert(0, list(sv[index_right, :]))
    sv_order[0, :] = sv[index_left, :]
    # sv_order.insert(0, list(sv[index_left, :]))
    sv_order[1, :] = sv[index_right, :]
    distance_list.append(np.min(sv_matrix))
    sv_matrix[:, index_right] = float("inf")  # 已排序的点到其他数据距离设置为无穷大
    sv_matrix[:, index_left] = float("inf")
    list_right = sv_matrix[index_right, :]  # 分别找到sv_list两端最近的点
    list_left = sv_matrix[index_left, :]
    # print("l:", index_left)
    # print("l:", index_right)
    min_point_right = np.min(list_right)
    min_point_left = np.min(list_left)
    while num > len(sv_order):
        if min_point_left < min_point_right:  # 如果左边更小，加入sv_list，更新左边节点
            index_left, = np.where(list_left == np.min(list_left))  # "," make return type array
            # print("l:", index_left)
            # index_left = index_left[0]  # tuple change to array
            index_left = index_left[0]  # array change to int
            # sv_order.insert(0, list(sv[index_left, :]))
            sv_order = np.concatenate((np.reshape(sv[index_left, :], [1, attr]), sv_order))
            distance_list.insert(0, np.min(list_left))
            sv_matrix[:, index_left] = float("inf")
            list_left = np.reshape(sv_matrix[index_left, :], num)
            min_point_left = np.min(list_left)  # 两边都要重新算避免两边最近点是同一个点
            min_point_right = np.min(list_right)
        else:
            index_right = np.where(list_right == np.min(list_right))
            # print("r:", index_right)
            index_right = index_right[0]  # tuple change to array
            index_right = index_right[0]  # array change to int
            # sv_order.append(list(sv[index_right, :]))
            sv_order = np.concatenate((sv_order, np.reshape(sv[index_right, :], [1, attr])))
            distance_list.append(np.min(list_right))
            sv_matrix[:, index_right] = float("inf")
            list_right = np.reshape(sv_matrix[index_right, :], num)
            min_point_right = np.min(list_right)
            min_point_left = np.min(list_left)
        # print
    # print(sv_order)
    return sv_order, distance_list


def show_line(sv, sv_order):
    plt.figure()
    plt.plot(sv[:, 0], sv[:, 1], ".", color="red")
    plt.plot(sv_order[:, 0], sv_order[:, 1], color="black")
    plt.show()
    pass


if __name__ == '__main__':
    a = np.array([[3,4],[5,6]])
    b = np.array([[1,2]])
    print(type(a[0,:]))
    c = np.concatenate((b,a))
    print(c)
    # a = []
    # b = np.array([1, 2])
    # a.insert(0,b)
    # print(np.array(a))
    pass
