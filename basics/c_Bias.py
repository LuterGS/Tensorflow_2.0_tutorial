from basics import b_Neuron as NN
import tensorflow as tf

"""
편향에 대해 "본격적으로" 알아보자!

입력이 0인 경우, 경사하강법을 적용하면 절대 학습되지 않는다. 입력이 0이기 떄문이지.
이걸 대비해, 편향을 넣는다는데, 고정된 값을 받는다고 하네;;

보통의 식 계산은 
    hidden = activation_func ( input * weight ) 
이고, 여기서 input = 0이면 결과도 0이지

편향을 대입한다면
    hidden = activation_func ( input * weight + fixed_value * bias )
    여기서, fixed_value는 input과 비슷하고, bias는 weight과 비슷하다.
    즉, 0으로 빠지는걸 방지한다는 거...인데, 이게 전체 결과값과 차이가 없나?
"""


# 자 기본으로 가보고, 하나 세워보자

def bias_gradient_descent(cycle, input, weight, bias, bias_weight, answer, learning_rate, activation_func):
    for i in range(cycle):
        output = activation_func(input * weight + bias * bias_weight)
        error = answer - output
        weight = weight + (input * learning_rate * error)
        bias_weight = bias_weight + (bias * learning_rate * error)

        if i % 100 == 99:
            print(i, error, output)

    print("this is bias_gradient_descents")

if __name__ == "__main__":
    print("Start b_g")
    bias_gradient_descent(1000, 0, tf.random.normal([1], 0, 1), 1, tf.random.normal([1], 0, 1), 1, 0.01, NN.sigmoid)
    print("End b_g")

    NN.gradient_descent_advanced(1000, 1, tf.random.normal([1], 0, 1), 0, 0.01, NN.sigmoid)
