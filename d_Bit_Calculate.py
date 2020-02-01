import tensorflow as tf
import numpy as np
import b_Neuron as neuron

"""
Trune는 당연히 1, False는 당연히 0인건 모두가 아는 사실이잖아?

Supervised Learning이라 생각하면
AND 연산              OR 연산               XOR 연산
Input   Answer      Input   Answer      Input   Answer      
0,0     0           0,0     0           0,0     0
0,1     0           0,1     1           0,1     1
1,0     0           1,0     1           1,0     1
1,1     1           1,1     1           1,1     0
인 데이터셋을 트레이닝시킨다고 보면 되겠네.

그럼 입력-출력계층만 설계하면
input_node가 2개, output_node가 1개일테니... 한번 세워볼까? (bias인 경우에는 좀 다르게)
"""
def reLU(x):
    return max(0, x)

class bit_neuralnetwork:
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
                # input_dataset이 그냥 tf.constant 배열이기 떄문에, numpy의 배열로 변환하는 작업.
            self.answer = self.answer_dataset[i-1]
            self.output_node = self.act_func(tf.matmul(self.weight, input, True))
                # 계산시 weight * layer이고, weight가 transpose 되어있어야 계산이 되므로 transpose값을 true로 지정
            error = self.answer - self.output_node
            self.weight = tf.add_n([self.weight, input * self.learning_rate * error])
                # 이젠 이렇게 써줘야 더해지더라...
            error_value += error
        return error_value/4

    def training(self):

        for i in range(2000):
            error = self.train_onestep()
            if i % 100 == 99:
                print(i, error)

    def test(self):
        for i in range(4):
            input = np.reshape(self.input_dataset[i-1], (self.input_nodenum, 1))
            answer = self.act_func(tf.matmul(self.weight, input, True))
            print("input: ", self.input_dataset[i-1], "\nanswer: ", answer)
        print("weight : ", self.weight)


if __name__ == "__main__":
    input_nonbias = tf.constant([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
    input_bias = tf.constant([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    output = tf.constant([0.0, 1.0, 1.0, 0.0])  #여기 정답만 수정해서 AND, OR, XOR 다 실험 가능
    test = bit_neuralnetwork(input_bias, output, 0.01, neuron.sigmoid, True)
    test.training()
    test.test()
    """
    Result
                AND 연산             AND 연산 (Bias)     OR 연산              OR 연산 (Bias)       XOR 연산             XOR 연산 (Bias)
                Input   Output      Input   Output      Input   Output      Input   Output      Input   Output      Input   Output 
                     
    Sigmoid     0,0     0.5(p)      0,0     0.0         0,0     0.5(p)      0,0     0.21        0,0     0.5(p)      0,0     0.5
                0,1     0.5         0,1     0.16        0,1     0.95        0,1     0.91        0,1     0.49        0,1     0.5
                1,0     0.5         1,0     0.17        1,0     0.95        1,0     0.92        1,0     0.5         1,0     0.5
                1,1     0.5         1,1     0.75        1,1     1           1,1     1           1,1     0.5         1,1     0.49
                
    reLU        0,0     0(p)        0,0     0(p)        0,0     0(p)        0,0     0.25        0,0     0(p)        0,0     0.51
                0,1     0.3         0,1     0.01        0,1     0.66        0,1     0.75        0,1     0.33        0,1     0.51
                1,0     0.3         1,0     0.01        1,0     0.66        1,0     0.75        1,0     0.33        1,0     0.5
                1,1     0.6         1,1     0.98        1,1     1.33        1,1     1.25        1,1     0.6         1,1     0.5
                
        * (p)라고 표시되어있는 것들은 정확하게 그 값이 나왔다는 것을 의미함
        * XOR은 전체적으로 잘 구분하지 못하는 모습을 보임. 단층 신경망의 한계인듯 (XOR Problem)
        * 나머지는 편향을 넣었을 때, 보통 잘 판별하며, sigmoid(0) = 0.5, reLU(0) = 0 의 특성 때문에 (p)들이 나온 것이라 생각됨
    """

