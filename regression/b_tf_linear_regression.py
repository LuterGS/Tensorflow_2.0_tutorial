import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from regression import a_linear_regression as linear_regression


class machine_learn_regression:
    def __init__(self, X, Y):
        self.X_data = X
        self.Y_data = Y
        self.a = tf.Variable(random.random())
        self.b = tf.Variable(random.random())
        self.optimizer = tf.optimizers.Adam(lr=0.05)

    def compute_loss(self):
        y_predicted = self.a * self.X_data + self.b
        loss = tf.reduce_mean((self.Y_data - y_predicted) ** 2)
        return loss

    def training(self):

        print(self.X_data, self.Y_data)
        for i in range(100000):
            # 잔차의 제곱의 평균을 최소하 (minimize)
            self.optimizer.minimize(self.compute_loss, var_list=[self.a, self.b])

            if i % 100 == 99:
                print(i, 'a:', self.a.numpy(), 'b:', self.b.numpy(), 'loss:', self.compute_loss().numpy())

        line_x = np.arange(min(self.X_data), max(self.X_data), 0.1)
        line_y = self.a * line_x + self.b

        return line_x, line_y

def show_graph(line_x, line_y, x, y):
    plt.plot(line_x, line_y, 'r-')
    plt.plot(x, y, 'bo')
    plt.show()

if __name__ == "__main__":

    data = linear_regression.get_weather_data()

    test = machine_learn_regression(data[0], data[1])
    x, y = test.training()
    show_graph(x, y, data[0], data[1])
    """
    텐서플로우 함수를 이용한 선형회귀, 10만번 트레이닝, 학습률 0.1로 맞춤으로써 시작시 99만 loss를 보였던 것을 15까지 줄였음
    Training의 횟수를 늘림으로써 해결
    """



