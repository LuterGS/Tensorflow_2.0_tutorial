import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

"""
실제 데이터를 가지고 하는 딥러닝 회귀.
전처리는 이전에 했던 김포공항 풍속/기압과 비슷하게 처리했다.

이후, 데이터를 정규화해야 한다. 단위랑 퍼센트 등등 전부 다르기에.
데이터를 정규화하려면, 각 데이터에서 평균괎을 뺀 다음 표준편차로 나눈다.


data 폴더의 파일 설명
검증손실차이는, 트레이닝 데이터셋에서 실제로 학습시킨 데이터셋과 검증한 데이터셋의 loss 차이를 그래프로 표현한 것
    2개가 나눠져있다.
정답테스트는, 학습 완료후 실제 데이터셋을 학습시킨 결과와 정답값을 1대1 매칭해 점으로 표시한 그래프. y=x직선에 가까워야 잘 학습된것임

콜백적용은, 네트워크에 콜백을 적용한 결과임.

"""


class seoul_weather_deeplearning:

    def __init__(self, layer, layer_node_num, layer_activation, learning_rate):
        t_input, t_output = self.get_weather_data()
        std_input, std_output = self.standardization_data(t_input, t_output)
        t_input = t_input.transpose()
        std_input = std_input.transpose()
        std_output = std_output.transpose()
        self.train_data, self.train_answer, self.test_data, self.test_answer = self.set_data(std_input, std_output)
        self.nn_model = tf.keras.Sequential(self.set_nn(layer, layer_node_num, layer_activation))
        self.nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='mse')
        self.nn_model.summary()

    @staticmethod
    def get_weather_data():
        file = open('../data/서울시_기상관측.csv', 'r', encoding='utf8')
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        input, output, = [], []
        for line in csv_reader:
            date = line[2]
            line = line[3:]
            output.append(float(line[0]))
            input.append([float(line[i+1]) for i in range(12)])
        file.close()
        np_input = np.asarray(input).transpose()
        np_output = np.asarray(output).transpose()
        return np_input, np_output

    @staticmethod
    def standardization_array(data):
        mean = data.mean()
        std = data.std()
        data -= mean
        data /= std
        return data

    def standardization_data(self, input, answer):
        for i in range(len(input)):
            input[i] = self.standardization_array(input[i])
        answer = self.standardization_array(answer)
        return input, answer

    def set_data(self, input, answer):
        a = 0
        training_input, training_output = [], []
        test_input, test_output = [], []
        for i in range(int(len(answer)/5)):
            for j in range(4):
                training_input.append(input[a])
                training_output.append(answer[a])
                a += 1
            test_input.append(input[a])
            test_output.append(answer[a])
            a += 1
        return training_input, training_output, test_input, test_output

    def set_nn(self, layer, layer_node_num, layer_activation):
        return_list = [
            tf.keras.layers.Dense(units=layer_node_num[0], activation=layer_activation[0], input_shape=(12,))]
        for i in range(layer - 2):
            return_list.append(tf.keras.layers.Dense(units=layer_node_num[i+1], activation=layer_activation[i+1]))
        return_list.append(tf.keras.layers.Dense(units=layer_node_num[layer-1]))

        return return_list

    def training(self):
        self.train_data = np.asarray(self.train_data)
        self.train_answer = np.asarray(self.train_answer)
        self.history = self.nn_model.fit(
            self.train_data, self.train_answer, epochs=50, batch_size=32, validation_split=0.25,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])
        """
        새로 배우는 콜백함수. 러닝 중간에 중단할수있게끔 함. 마치 c_nonlinear_regression에서 if로 중단을 줬던거와 비슷한가?
        콜백을 적용하니 Epoch 7을 하는 중간에 멈췄다.
            - patience 는 몇 회를 지켜볼것인지, monitor는 val_loss를 의미함
                즉, 현재 함수는 val_loss를 3회 epoch로 지켜보며, 3회동안 최고기록을 갱신하지 못한다면 학습을 종료함
                과적합이나 진동을 막기 위한 방법?
        """


    def evaluate(self):
        self.test_data = np.asarray(self.test_data)
        self.test_answer = np.asarray(self.test_answer)
        self.nn_model.evaluate(self.test_data, self.test_answer)
        """
        간단히 evaluate는 loss만을 평가하는것 같음. 따로 시각화하려면 matplotlib.pyplot을 써야 하고, 그 전에 텍스트로 볼 수 있는건 무조건 이거인듯
        넣은 input, 정답을 가지고 단순히 loss만을 계산함
        """

    def show_matching(self):
        predict_answer = self.nn_model.predict(self.test_data)
        '정답에 해당하는 학습데이터를 판별시켜높음. 일종의 실습. 실제 데이터를 트레이닝시킨 결과라 할 수 있음'

        plt.figure(figsize=(5, 5))
        '맵 사이즈를 5x5로 한다는거같은데, 정규분포를 토대로 그려지니 맞는 선택이라 할 수 있음'
        plt.plot(self.test_answer, predict_answer, 'b.')
        '똑같은 데이터셋인데 하나는 학습셋을 학습시킨것, 하나는 진짜 정답값을 1대1 매칭으로 점으로 표기. 즉, y=x 직선에 가까울수록 성능이 좋음'
        plt.axis([min(self.test_answer), max(self.test_answer), min(self.test_answer), max(self.test_answer)])

        'y=x 그리는 함수.'
        plt.plot([min(self.test_answer), max(self.test_answer)], [min(self.test_answer), max(self.test_answer)], ls="--", c=".3")
        plt.xlabel('test_answer')
        plt.ylabel('predicted_answer')
        plt.show()


if __name__ == "__main__":

    test = seoul_weather_deeplearning(4, [52, 39, 26, 1], ['relu', 'relu', 'relu'], 0.05)
    test.training()
    test.evaluate()
    plt.plot(test.history.history['loss'], 'b-', label='loss')
    plt.plot(test.history.history['val_loss'], 'r--', label='val_loss')
    'history는 트레이닝의 과정을 가지고있는 변수 느낌이라, 데이터셋을 트레이닝한 loss, 검증셋을 트레이닝한 val_loss가 있는거같음'
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    test.show_matching()