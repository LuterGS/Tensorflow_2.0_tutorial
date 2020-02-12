#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv


"""
선형 회귀는, 2차원 데이터 - 두 종류의 데이터를 x,y라 할 때, x, y의 방향성을 가장 잘 보여주는 직선을 그리는 것이다.
그린 직선과 각 데이터의 거리가 최소화되는 선, 그게 바로 선형회귀의 회귀선이다.

data폴더의 김포공항_풍속_기압_선형회귀.png를 보면, 선형회귀선은 각 데이터 (점)와 선 사이의 거리를 모두 더했을때, 다른 선들보다 그 값이 작은 선이다.
이런 잔차를 최소화하는 알고리즘을 최소제곱법이라고 한다.

이 선은 직선이므로 2차원 평면상에서 y = ax + b로 표현할 수 있다 (a가 기울기, b가 절편)

a = sum(( yi - y평균) * (xi - x평균)) / (xi - x평균)^2      (i는 1부터 n까지)
b = y평균 - a * x평균

해당 식은 함수에 구현되어있다.

"""




"""
def get_weather_data는 김포공항_풍속_기압.csv에서 데이터를 읽어오는 함수이다. transpose - 전치를 사용해 각 행이 각각의 x,y축이 되도록 했다.
"""
def get_weather_data():
    file = open('../data/김포공항_풍속_기압.csv', 'r', encoding='utf8')
    csv_reader = csv.reader(file)
    csv_reader.__next__()
    data = []
    for line in csv_reader:
        line = line[3:]
        line[0] = float(line[0])
        line[1] = float(line[1])
        data.append(line)
    file.close()
    np_data = np.array(data).transpose()
    #print(np_data)
    return np_data

def delete_abnormal_data(data):
    print("HELLO")


def linear_regression(data):
    x_avg = sum(data[0]/len(data[0]))
    y_avg = sum(data[1]/len(data[1]))
    #print(x_avg, y_avg)

    a = sum([(y - y_avg) * (x - x_avg) for y, x in list(zip(data[1], data[0]))])
    a /= sum([(x - x_avg) ** 2 for x in data[0]])
    b = y_avg - a * x_avg
    print('a:', a, 'b:', b)
    """
    최소제곱법 공식을 적용한다.
    """

    line_x = np.arange(min(x), max(y), 0.01)
    line_y = a * line_x + b
    """
    x, y를 지정해줘야 하는데, x를 풍속의 최소 - 최대값으로 잡아준다. 
    """

    return line_x, line_y

def show_graph(X, Y, x, y):
    #X,Y는 선형회귀선, x, y는 데이터셋
    plt.plot(x, y, 'bo')
    plt.plot(X, Y, 'r-')
    plt.xlabel("Wind speed")
    plt.ylabel("Atmospheric Pressure")
    plt.show()


if __name__ == "__main__":

    data = get_weather_data()
    #첫 번째 데이터는 풍속, 두 번째 데이터는 기압을 나타내게끔 함

    plt.plot(data[0], data[1], 'ro')
    plt.xlabel("Wind Speed")
    plt.ylabel("Atmospheric Pressure")
    plt.show()

    a, b = linear_regression(data)
    show_graph(data[0], data[1], a, b)