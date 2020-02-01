import tensorflow as tf
import numpy as np


"""
단층 신경망의 XOR 문제를 막기 위해, 다층 신경망을 설계해보자
일단 3-layer부터 사용하는데, Keras를 사용한다
"""

if __name__ == "__main__":
    input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output = np.array([[0], [1], [1], [0]])
    """ 초기화의 차이인거같은데, numpy를 이용해서 배열을 만들고, output도 2차배열로 선언해줬다. 어차피 신경망의 weight가 2차배열인걸 생각하면, 옳은 결정이라 생각함.
        따라서 input은 [4,2], output은 [4,1]이다. """

    