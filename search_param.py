from MainSVC import *
import os


def save_result(samples, label, num_cluster, accuracy, k, e):
    color_list = ["red", "green", "blue", "yellow", "black"]
    plt.figure()
    for i in range(num_cluster):
        cluster = samples[label == i + 1, :]
        color = color_list[i % len(color_list)]
        plt.plot(cluster[:, 0], cluster[:, 1], ".", color=color)
    plt.savefig("G:\pycharm\pywork\svmwork\primal_svc\\wine_result\\%.4f_c%.2f_q%.2f_k%d_e%d.jpg" % (accuracy, c, q, k, e))  # \%.2f_%.2fdir c, q,
    plt.close()


if __name__ == '__main__':
    # 获取数据
    np.random.seed(10)
    # 4.iris
    # c, q = 0.04, 3
    c_list = np.arange(0.01, 0.5, 0.01)
    q_list = np.arange(1.0, 100.0, 0.1)
    sample, true_label = cd.get_wine()
    true_cluster_num = int(np.max(true_label))
    # flag1 = 0
    # flag2 = 0

    sample = preprocessing.scale(sample)
    num, attr = sample.shape  # num：数据数量。 attr：属性维度
    for c in c_list:
        for q in q_list:
            print(c, q)
            sv, bsv, beta, quad, r, un_sv = svc(sample, c, q)  # solve the first step of algorithm
            sv_num, _ = sv.shape

            if sv_num == num:  # 所以点为sv无法执行算法
                continue
            sv_matrix = kernel_matrix(sv, q)  # the matrix using for sort sv
            sv_unsv_matrix = r_ker_matrix(un_sv, sv, q)  # num_sv x num_un_sv kernel matrix
            sv_sample_matrix = r_ker_matrix(sample, sv, q)  # num_sv x num_sample kernel matrix

            sv_order, distance_list = ss.point_sort(sv_matrix, sv)

            # print(sv_order.shape)
            num_sv, _ = sv.shape
            k_top = int(np.floor(num / num_sv * 3))+1
            k_low = 0  # int(np.floor(num / num_sv * 2))
            # print(k)
            # if not os.path.exists("G:\pycharm\pywork\svmwork\primal_svc\\result1\%.2f_%.2fdir" % (c, q)):
            #     os.mkdir("G:\pycharm\pywork\svmwork\primal_svc\\result1\%.2f_%.2fdir" % (c, q))
            for k in range(k_low, k_top):
                for param_e in range(k):
                    sv_label, num_cluster = snn.cluster_sv(sv_unsv_matrix, k, param_e)
                    if num_cluster < true_cluster_num or num_cluster > true_cluster_num+2:  # 如果聚类个数少于真实值，或者过多。不进行进一步操作
                        continue
                    sample_label = snn.cluster_sample(sample, sv_label, sv_sample_matrix)
                    accuracy = ac.accuracy_cluster(sample_label, num_cluster, true_label, true_cluster_num)
                    if accuracy < 0.30:
                        continue
                    print(accuracy)
                    save_result(sample, sample_label, num_cluster, accuracy, k, param_e)
                    # plt.figure()
                    # plt.plot(sample[:, 0], sample[:, 1], ".", color="green")
                    # plt.plot(sv[:, 0], sv[:, 1], "o", color="red")
                    # plt.plot(sv_order[:, 0], sv_order[:, 1], color="black")
                    # plt.savefig("G:\pycharm\pywork\svmwork\primal_svc\\result\%.2f_%.2f.jpg" % (c, q))
                    # plt.close()
