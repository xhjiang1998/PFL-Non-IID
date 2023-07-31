import numpy as np


def initialize_parameters_deep(layer_dims): # 参数 layer_dims 定义为一个包含网络各层维数的list ，使用随机数和归零操作来初始化权重 W 和偏置 b 。
    np.random.seed(3)
    parameters = {}
    # number of layers in the network
    L = len(layer_dims)



    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_activation_forward(A_prev, W, b, activation): # 前向计算函数
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    # number of layers in the neural network
    L = len(parameters) // 2

    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        # Implement LINEAR -> SIGMOID
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

if __name__=="__main__":
    parameters = initialize_parameters_deep([5,4,3])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))