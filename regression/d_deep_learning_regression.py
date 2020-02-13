import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from regression import a_linear_regression as weather_dataset






if __name__ == "__main__":

    data = weather_dataset.get_weather_data()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=3, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

    model.summary()

    model.fit(data[0], data[1], epochs=10)

    line_x = np.arange(min(data[0]), max(data[0]), 0.01)
    line_y = model.predict(line_x)

    weather_dataset.show_graph(line_x, line_y, data[0], data[1])

    """
    결과값은 김포공항_풍속_기압_딥러닝_3layer회귀.png 로 저장됨
    basics/e_Advanced_Bit_using_Keras.py 에서 나왔던 것을 비슷하게 사용. 다른점은 이 layer에서 output쪽에 해당되는 데이터는
    활성함수를 적용하지 않음.
    """