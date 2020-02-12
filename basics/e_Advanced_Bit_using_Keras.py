from typing import Optional, Any

import tensorflow as tf
import numpy as np

#다른곳에서 쓰기 위한 함수화. 제대로된 설명은 밑에

class sequential_nn:

    def __init__(self, first_layer_node, second_layer_node, first_layer_activation, second_layer_activation, input_data_num):
        self.layer_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=first_layer_node, activation=first_layer_activation, input_shape=(input_data_num,)),
            tf.keras.layers.Dense(units=second_layer_node, activation=second_layer_activation)
        ])
        self.layer_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
        self.history = 0

    def verify(self):
        self.layer_model.summary()

    def training(self, input, output, epoch, batch_size):
        self.history = self.layer_model.fit(input, output, epochs=epoch, batch_size=batch_size)
        print(self.layer_model.predict(input))


"""
단층 신경망의 XOR 문제를 막기 위해, 다층 신경망을 설계해보자
일단 3-layer부터 사용하는데, Keras를 사용한다
"""

if __name__ == "__main__":
    input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output = np.array([[0], [1], [1], [0]])
    """ 초기화의 차이인거같은데, numpy를 이용해서 배열을 만들고, output도 2차배열로 선언해줬다. 어차피 신경망의 weight가 2차배열인걸 생각하면, 옳은 결정이라 생각함.
        따라서 input은 [4,2], output은 [4,1]이다. """

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    """
    tf.keras.Sequential : 순차적 layer로 이루어진 신경망을 뜻한다. 순차적이라는 뜻의 Sequential
        인수로 layer가 차례대로 정의된 리스트를 전달 ( Seqneutial([]) 같은 구조로 되어있으니 리스트로 받는게 맞고 )
    tf.keras.Dense : 완전연결신경망을 의미하는구나. 이렇게 연결하면 옆의 레이어에 있는 모든 노드가 서로 완전연결된다고 보면 되겠다.
        units : 노드 수 (각각 2, 1개로 지정)
        activation : 활성함수 (sigmoid로 지정)
        input_shape : Sequential 모델에서 첫 번째 layer에만 지정 (input_layer니까), XOR 문제 해결에서는 1차원으로 들어오고, 원소수가 2개이기 떄문에 (2,)로 명시
                      그럼 만약에 2차원일 경우에는 (3,3,) 이런식으로 명시하나??
                      
    => Sequential 모델같은 경우, input배열과 완전히 동등한 hidden_layer가 생성되네, 즉, input값 - 가중치1 - 첫번째 Dense - 가중치 - 두번째 Dense - output으로 연결되는거같은데
       그럼 두번째 Dense에서 활성함수를 적용한 값이 output으로 인식되게 되어있나보네
       
    => Param은 일종의 가중치를 의미함.
        첫 번째 Dense에서 Param은 (input_shape+1) * units 임. 기본적으로 편향이 적용되어있기 떄문에, input_layer에서 원래자료값 + 편향값 1로 보는게 맞음
        두 번째 Dense에서 Param은 (이전 Dense의 unit + 1) * units임. 편향이 적용되어있기 떄문에, 이전 노드들 + 편향 1개가 현재 units과 완전연결되어있기 떄문.
    """

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
    """
    compile은 일종의 동작 준비 명령같은거네. 1버전대의 Sess.run할때 구성되는것들인가?
    optimizer는 최적화 함수. 딥러닝의 학습식을 정의한다고 보면 됨. 여기서는 SGC(Single Gradient Descent)방식의 optimizer를 사용.
        다른 optimizer - batch, mini batch 등등 - 을 사용할수도 있을까? 아니면 다르게 이전의 1버전대처럼 adamoptimizer를 쓴다거나 그럴수 있나?
        lr=0.1은 학습률을 나타냄.
        loss는 error와 비슷한 개념. 손실을 줄이는 방향으로 활용하는데, 보통은 평균제곱오차 방식을 사용. 이전의 처음 배우는 신경망에서 나왔던 제곱오차 방식을 적용했다고 보면 되겠네
    """

    model.summary()
    """
    평가할수있는, 잘 만들어졌나 확인하는 함수. 구성을 보여주는거같네.
    """

    history = model.fit(input, output, epochs=2500, batch_size=1)
    """
    trainig함수. fit이 training 시키는 구문인가보네. epoch는 한 Data cycle을 의미함. batch_size가 의미하는건, 역전파할때 드러남.
        예를 들어, batch_size가 5면, 5번 모두 순전파하고 그만큼의 error를 쌓은 후 역전파함. 1이면, 한번 순전파될때 바로바로 역전파함
        epoch는 그 batch_size가 돌아가는 횟수
        batch_size가 총 data수와 동일하면 batch방식, 어느정도 텀을 두고 쪼개면 mini_batch, 1이면 SGD라고 할 수 있음
        
        
        
        
    TRAINING RESULT
        잘 나옴
    """

    print(model.predict(input))

    for weight in model.weights:
        print(weight)
    """
    model 클래스 내에 weight가 있나보네
    """


