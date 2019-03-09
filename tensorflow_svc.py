from MainSVC import *
from sklearn import preprocessing
import tensorflow as tf


def solve_gd(num, k_matrix, c, m, n):
    # 待输入约束参数 lam1：等式约束 lam2：不等式约束
    lam1 = tf.placeholder(tf.float32)
    lam2 = tf.placeholder(tf.float32)
    # beta
    # b = tf.Variable(tf.multiply(tf.ones(shape=[1, num]), 1/num), dtype=tf.float32)
    b = tf.Variable(tf.truncated_normal(shape=[1, num], mean=1/num, stddev=0.1), dtype=tf.float32)
    # 待输入核函数矩阵
    my_kernel = tf.placeholder(shape=[num, num], dtype=tf.float32)
    # 创建loss
    f_term = tf.reduce_sum(tf.multiply(b, tf.diag_part(my_kernel)))
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    s_term = tf.reduce_sum(tf.multiply(my_kernel, b_vec_cross))
    st1_term = tf.add(tf.reduce_sum(tf.maximum(tf.negative(b), 0)), tf.reduce_sum(tf.maximum(tf.subtract(b, c), 0)))
    st2_term = tf.abs(tf.reduce_sum(b) - 1)
    loss = tf.add(tf.add(tf.subtract(s_term, f_term), tf.multiply(lam1, st1_term)), tf.multiply(lam2, st2_term))

    # 创建优化器
    learning_rate = 0.0003  # 初始学习速率时0.1
    decay_rate = 0.95  # 衰减率
    global_steps = 9000  # 总的迭代次数
    decay_steps = 80  # 衰减次数

    global_ = tf.placeholder(tf.float32)
    adj_learning_rate = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
    my_opt = tf.train.GradientDescentOptimizer(adj_learning_rate)
    train_step = my_opt.minimize(loss)

    # 初始化变量
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    loss_vec = []
    for i in range(global_steps):
        _, temp_loss = sess.run((train_step, loss), feed_dict={my_kernel: k_matrix, lam1: m + i/500, lam2: n+i/500, global_: i})
        # temp_loss = sess.run(loss, feed_dict={my_kernel: k_matrix, lam1: m, lam2: n})
        loss_vec.append(temp_loss)
        if i % 300 == 0:
            print(i, "loss", temp_loss)
        # loss_vec.append(temp_loss)
    beta = sess.run(b)
    sess.close()
    beta = np.reshape(beta, (num, 1))
    index = range(0, 9000, 30)
    loss_vec = np.array(loss_vec)
    plt.figure()
    plt.plot(index, loss_vec[index], color='black')
    plt.show()
    return beta


def tensor_svc(samples, c, q, m, n):
    num, _ = samples.shape
    # calculate kernel matrix
    k_matrix = kernel_matrix(samples, q)
    # solve lagrangian multipliers
    beta = solve_gd(num, k_matrix, c, m, n)
    # find support vector
    sv, num_sv, bsv, num_bsv, un_sv = find_sv_bsv(samples, beta, c)
    # calculate radius of sphere
    quad, r = cal_sphere_r(samples, sv, beta, k_matrix, q)
    return sv, bsv, beta, quad, r, un_sv


if __name__ == '__main__':
    # 获取数据
    np.random.seed(10)
    # 1.two gauss data
    c, q = 1, 1.5
    m, n = 1, 10
    sample = cd.gauss_data()
    # 3.moon
    # c, q = 0.01, 5
    # sample = cd.moon_data(150, -2, 10, 5)
    sample = preprocessing.scale(sample)
    num, attr = sample.shape  # num：数据数量。 attr：属性维度
    # calculate kernel matrix
    k_matrix = kernel_matrix(sample, q)
    # solve lagrangian multipliers
    beta = solve_gd(num, k_matrix, c, m, n)

    num_bsv, num_sv = 0, 0
    bsv, sv, un_sv = [], [], []
    for i in range(num):
        if beta[i] > c - 0.001:
            num_bsv += 1
            bsv.append(sample[i, :])
        elif beta[i] > 0.001:
            num_sv += 1
            sv.append(sample[i, :])
        else:
            un_sv.append(sample[i, :])
    sv, bsv, un_sv = np.array(sv),  np.array(bsv),  np.array(un_sv)

    plt.figure()
    plt.plot(un_sv[:, 0], un_sv[:, 1], ".", color="green")
    plt.plot(sv[:, 0], sv[:, 1], ".", color="red")
    # plt.plot(bsv[:, 0], bsv[:, 1], color="black")
    plt.show()
