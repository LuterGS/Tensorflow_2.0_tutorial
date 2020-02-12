import tensorflow as tf
import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))
    #기본적인 시그모이드 함수. tensorflow 모듈 내 구현식이 있지 않을까?'

"""
경사하강법.. Gradient Descent 법으로 하면
w = w + (input * a * error), 이걸 for문으로 구현해보자
"""
def gradient_descent(cycle, input, weight, answer, learning_rate):
    for i in range(cycle):
        output = sigmoid(input * weight)
        error = answer - output
        weight = weight + input*learning_rate*error

        if i % 100 == 99:
            print(i, error, output)

# gradient_descent(10000, 1, tf.random.normal([1],0,1), 0, 0.1) #경사하강법의 함수화 시킨건데, activation function도 함수화시킨다면?

def gradient_descent_advanced(cycle, input, weight, answer, learning_rate, activation_func):
    for i in range(cycle):
        output = activation_func(input * weight)
        error = answer - output
        weight = weight + (input * learning_rate * error)

        if i % 100 == 99:
            print(i, error, output)


if __name__ == "__main__":

    input = 1
    b = sigmoid(input) #이건 단순한 시그모이드함수 결과값 적용한거고'
    W = tf.random.normal([1], 0, 1) #가중치 init시 초기화시키는 느낌으로 설정하면...
    output = sigmoid(input*W) #이게 출력노드값이라 할 수 있겠네
    answer = 0 #신경망의 answer. Supervised Learning시 사용하는거네
    error = answer - output #에러값은 당연히 answer - output으로 표현하는게 맞지
    """
    뉴런 노드들 자체는 전 노드값 * 가중치값을 활성홤수에 넣으니' 
    활성함수를 f, 전 노드값을 p, 가중치를 W라 하면' 
    그 노드의 출력값이 y일 때 y=f(p*W) 라고 할 수 있겠네'
    """
    print(input, b, output)

    gradient_descent_advanced(1000, 1, tf.random.normal([1],0,1), 0, 0.01, sigmoid) #이게 되네;;
