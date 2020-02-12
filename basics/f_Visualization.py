import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from basics import e_Advanced_Bit_using_Keras as xornet


def show_normal_distibution():

    random_normal = tf.random.normal([1000000], 0, 1)
    plt.hist(random_normal, bins=1000)
    plt.show()
    """
    hist는 x축의 구역을 몇개로 나눠서 보여줄거인지를 의미하는거같음.
    bins=n 을 이용함. n이 몇 개의 구역으로 나눌건지. 지금같은경우는 1000개
    """

def show_network():
    network = xornet.sequential_nn(2, 1, 'tanh', 'tanh', 2)
    network.verify()
    network.training(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]), 2000, 1)
    plt.plot(network.history.history['loss'])
    plt.show()

    """
    history 변수 안의 loss가 기록으로 남나보네. loss를 따질수있게끔...
    
    결과값을 보면 loss가 급격히 줄다가 안정적으로 작아지게끔 됨
    """

if __name__ == "__main__":
    x = range(20)
        # 1부터 20까지의 list
    y = tf.random.normal([20], 0, 1)
        # 표준편차 1, 평균 0인 정규분포에서 값 20개 추출
    # plt.plot(x, y)
        #일반그래프
    # plt.show()
    # plt.plot(x, y, 'r--')
    """
        bo = blue - o (dot)  -  점을 나타냄
        b- = blue - - (line) -  선을 나타냄
        b-- = blue - -- (점선)-  점선을 나타냄
        b를 바꿔서 색깔 다르게 표시 가능
        하나만 넣으면 그걸 y로 간주하고, x는 자동으로 range(len(y))로 해줌. 
    """
    # plt.show()
    # show_normal_distibution()

    show_network()