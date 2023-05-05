# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


# X为特征矩阵，y为标签向量
X, y = read_data()


def ridge(data):
    # 样本数量和特征维度
    N, D = X.shape
    # alpha强度超参数
    alpha = 0.16
    A = np.eye(D) * alpha
    # 最小二乘法求解权重
    w_ridge = np.linalg.inv(X.T @ X + alpha * A) @ X.T @ y
    return w_ridge @ data


def lasso(data):
    max_iter = 10000
    lr = 0.001
    alpha = 0.01
    # 初始化权重为0向量
    w_lasso = np.zeros(X.shape[1])
    for i in range(max_iter):
        # 计算预测值和误差
        y_pred = np.matmul(X, w_lasso)
        error = y_pred - y
        # 计算梯度
        gradient = np.matmul(X.T, error) / X.shape[0] + alpha * np.sign(w_lasso)
        # 防止梯度爆炸
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1:
            gradient /= grad_norm
        # 更新权重向量
        w_lasso -= lr * gradient
        # 对权重向量进行L1正则化
        w_lasso = np.sign(w_lasso) * np.maximum(np.abs(w_lasso) - alpha * lr, 0)
    return np.matmul(w_lasso, data)
