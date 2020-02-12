import tensorflow as tf
import numpy as np
import random
from regression import a_linear_regression as linear_susic
from regression import b_tf_linear_regression as tf_linear_reg

"""
파이썬 공부를 위해 만든 상속을 이용한 2차 회귀함수.
"""
class nonlinear2_regression(tf_linear_reg.machine_learn_regression):

    def __init__(self, X, Y):
        super().__init__(X, Y)
        self.c = tf.Variable(random.random())

    def compute_loss(self):
        y_predicted = self.a * self.X_data ** 2 + self.b * self.X_data + self.c
        loss = tf.reduce_mean((self.Y_data - y_predicted) ** 2)
        return loss

    def set_varlist(self):
        return [self.a, self.b, self.c]

    def return_value(self):
        line_x = np.arange(min(self.X_data), max(self.X_data), 0.1)
        line_y = self.a * line_x * line_x + self.b * line_x + self.c

        return line_x, line_y

"""
모든 n차 회귀를 계산할수있게끔 설계한 all_regression 함수
degree(차수)에 따라 각 x의 차수항에 대입되는 variable을 만들고, 계산하게끔 설계
(좀 더 컴공적 설계)
"""
class all_regression:

    def __init__(self, X, Y, degree, learning_rate):
        self.X_data = X
        self.Y_data = Y
        self.Degree = degree
        self.optimizer = tf.optimizers.Adam(lr=learning_rate)
        self.variables = self.make_variable(degree)

    def make_variable(self, Degree):
        new_variable = []
        for i in range(Degree + 1):
            new_variable.append(tf.Variable(random.random()))
        return new_variable

    def compute_loss(self):
        y_predicted = 0
        for i in range(self.Degree + 1):
            y_predicted += self.variables[i] * (self.X_data ** (self.Degree-i))
        loss = tf.reduce_mean((self.Y_data - y_predicted) ** 2)
        return loss

    def print_mid_result(self, num):
        print("train : ", num, end='   ')
        for i in range(self.Degree + 1):
            print(" ", i, ": ", self.variables[i].numpy(), end='')
        print("         loss : ", self.compute_loss().numpy())

    def training(self, training_cycle, estimate_error):
        for i in range(training_cycle):
            self.optimizer.minimize(self.compute_loss, var_list=self.variables)

            if i % 1000 == 999:
                self.print_mid_result(i)
                if self.compute_loss().numpy() < estimate_error:
                    print("total training cycle : ", i)
                    break

    def return_value(self):
        line_x = np.arange(min(self.X_data), max(self.X_data), 0.1)
        line_y = 0
        for i in range(self.Degree + 1):
            x_value = line_x ** (self.Degree - i)
            line_y += self.variables[i] * x_value

        return line_x, line_y





if __name__ == "__main__":
    data = linear_susic.get_weather_data()
    """
    test = nonlinear2_regression(data[0], data[1])
    test.training(100000, 16)
    x, y = test.return_value()
    linear_susic.show_graph(x, y, data[0], data[1])
    """

    test = all_regression(data[0], data[1], 4, 0.01)
    test.training(100000, 16)
    x, y = test.return_value()
    linear_susic.show_graph(x, y, data[0], data[1])


    """
    4차함수인 경우
    a, b, c, d, e 는 각각 variables에서 0, 1, 2, 3, 4
    ax4 + bx3 + cx2 + dx + e
    """



