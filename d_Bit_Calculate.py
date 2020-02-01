import tensorflow as tf
import numpy as np
import b_Neuron as neuron

"""
Trune는 당연히 1, False는 당연히 0인건 모두가 아는 사실이잖아?

And 연산을 한다 생각했을때, Supervised Learning이라 생각하면
Input   Answer
0,0     0
0,1     1
1,0     1
1,1     1
인 데이터셋을 트레이닝시킨다고 보면 되겠네.

그럼 입력-출력계층만 설계하면
input_node가 2개, output_node가 1개일테니... 한번 세워볼까?
"""
def reLU(x):
    return max(0, x)

class and_func:
    def __init__(self, input_dataset, answer_dataset, learning_rate, act_func, Bias=False):
        if Bias==True:
            self.weight = tf.random.normal([3, 1], 0, 1)
            self.input_nodenum = 3
        else:
            self.weight = tf.random.normal([2, 1], 0, 1)
            self.input_nodenum = 2
        self.input_dataset = input_dataset
        self.answer_dataset = answer_dataset
        self.learning_rate = learning_rate
        self.act_func = act_func
        self.output_node = tf.zeros([1])
        self.answer = tf.zeros([1])
        self.datanum = tf.rank(self.input_dataset)

    def train_onestep(self):
        error_value = 0
        for i in range(4):
            input = np.reshape(self.input_dataset[i-1], (self.input_nodenum,1))
            self.answer = self.answer_dataset[i-1]
            self.output_node = self.act_func(tf.matmul(self.weight, input, True))
            error = self.answer - self.output_node
            self.weight = tf.add_n([self.weight, input * self.learning_rate * error])
            error_value += error
        return error_value/4

    def training(self):

        for i in range(2000):
            error = self.train_onestep()
            if i % 100 == 99:
                print(self.answer, error)

    def test(self):
        for i in range(4):
            input = np.reshape(self.input_dataset[i-1], (self.input_nodenum, 1))
            answer = self.act_func(tf.matmul(self.weight, input, True))
            print("input: ", self.input_dataset[i-1], "\nanswer: ", answer)
        print("weight : ", self.weight)


if __name__ == "__main__":
    input_nonbias = tf.constant([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
    input_bias = tf.constant([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    output = tf.constant([0.0,1.0,1.0,1.0])
    test = and_func(input_bias, output, 0.01, neuron.sigmoid, True)
    test.training()
    test.test()
    """
    reLU, no Bias : [1,1] → 1.3 ,  [1,0],[0,1] → 0.6 ,  [0,0] → 0      사실상 0.6*2정도
    sigmoid, no Bias : [1,1] → 1 ,  [1,0],[0,1] → 1 ,  [0,0] → 0.5 (정확히 이값으로)
        ㄴ 0,0인 경우는 트레이닝이 안되는것때문에 0.0 트레이닝 결과값이 저럼. Bias를 넣고 시도
    reLU, with Bias : [1,1] → 1.25 ,  [1,0],[0,1] → 0.75 ,  [0,0] → 0.25
    sigmoid, with Bias : [1,1] → 1 ,  [1,0],[0,1] → 1 ,  [0,0] → 0.2
        ㄴ 신기하게 sigmoid일 때 편향을 넣었을때 더 잘 먹음. 함수 특성인가?
    """

