import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
이항분류 (Binary Classification)에 들어가기 앞서, 분률를 먼저 정리하자.

회귀는, 아웃풋이 하나이다.
즉, 어느 특수한 값 하나를 "예상"하는 것이다.
분류는, 아웃풋이 하나이다.
즉, 여러 개 중 하나를 골라내는 것이다.

원리는 비슷하다.
신경망에서 회귀나 분류나 출력값이 존재한다는 사실은 분명하지만, 회귀는 하나의 값에 대한 예상밖에 못한다.

그럼 예시를 하나 들어보자.
숫자 판별 MNIST를 회귀로 동작시킬 수 있는가?
예를 들어, 출력값이 0이면 0, 1이면 1... 9이면 9로?
데이터를 normal distribution을 이용해 전처리할 정도로 데이터셋 전체에 대한 선입견을 막으려고 하는 딥러닝/신경망인데, 이렇게 되면 값이 영향을 많이 받는다.
출력값은 말그대로 "세기"를 반영할 뿐, 특정 값을 분류한다고 말할 수는 없다.

따라서, 여러 개 중 하나를 골라내는 분류는 여러 출력node들이 있어야 한다.
그래서 분류가 존재하게 된다.

분류는, 회귀와 다르게 각 출력값의 출력 정도에 따라 정답을 판별한다.
MINST를 예로 들면 0~9까지 총 10개의 노드가 있고, 노드의 출력 수치가 가장 높은 것이 신경망이 추론한 정답이다.
똑똑하게 판별해야한다. 회귀가 좋은지, 분류가 좋은지.





리그 오브 레전드 E-Sports 경기들의 데이터셋을 이용해 블루팀, 퍼플팀의 승리를 판별하는 이항분류를 만들어보았다.
쓰인 데이터는 다음과 같다.
 - 게임 진행 시간
 - 게임 진행 시간의 반절일 때의 각 포지션별 플레이어들의 골드 수
 - 게임 진행 시간의 발절일 때의 골드 격차
정답 평가시 트레이닝 변수를 그대로 뒀을 때 약 75%의 정답률을 보인다. 
 - 15분으로 지정시에도 정답률은 75%이나, 테스트시 72% 정도의 정답률을 보임
 - 10분으로 지정시에는 약 67%의 정답률을 보임

"""


def read_file(location):
    result = pd.read_csv(location, sep=',')
    """
    pandas의 csv읽어들이는 기능을 이용해 csv 파일을 읽어들여온다. sep는 구분자를 의미하는 것 같다.
    """
    # print(result.describe())
    # print(result.info())
    # print(result.head())

    print(len(result))
    print(type(result['golddiff'][2]))

    need_refine = ['golddiff', 'goldblueTop', 'goldblueMiddle', 'goldblueJungle', 'goldblueADC', 'goldblueSupport', 'goldredTop', 'goldredMiddle', 'goldredJungle', 'goldredADC', 'goldredSupport']
    refine_data(result, need_refine)
    result.drop(['goldblue', 'goldred'], axis='columns', inplace=True)

    print(result.info())
    print(result.head())
    result.to_csv("../data/LeagueofLegends_정제.csv", sep=',')
    return result


def refine_data(data, value):

    """
    데이터를 정제시키는 함수이다. 데이터가 원래는 문자열로 표기된 리스트여서, 그걸 쪼갠 다음 중간값만 취했다.
    toList[int(len(toList)/2)])) 에서 []안의 값을 변경하면 되나, 17분 게임도 있었던걸 봐서 15분이 MAX일 것이다.
    또한, 게임을 생각했을 때 앞으로 가면 갈수록 신경망의 정답률도 떨어질 것이다.
    """
    print(len(value))
    for a in range(len(value)):
        list = []
        for b in range(len(data)):
            toList = data[value[a]][b].split(sep=', ')
            list.append(int(toList[10]))
        data[value[a]] = list

    return data


def standardization_data(data):

    """
    데이터를 정제하는 함수이다. 한 번 섞은 뒤, 0.8비율로 테스트셋과 정답셋을 나눈다
    """
    train_length = int(len(data)*0.8)
    data_norm = (data - data.min()) / (data.max() - data.min())
    print(data_norm.head())
    print(data_norm.describe())
    data_shuffle = data_norm.sample(frac=1)
    data_np = data_shuffle.to_numpy()
    train_X, train_Y = data_np[:train_length, 2:], data_np[:train_length, :2]
    test_X, test_Y = data_np[train_length:, 2:], data_np[train_length:, :2]
    return train_X, train_Y, test_X, test_Y



if __name__=="__main__":

    data = read_file("../data/LeagueofLegends.csv")
    train_X, train_Y, test_X, test_Y = standardization_data(data)

    print(len(train_X), len(train_Y), len(test_X), len(test_Y))
    print(train_Y)
    """
    혹시 몰라 정답 데이터 타입을 int로 바꿔줬다.
    """
    train_Y = train_Y.astype(int)
    test_Y = test_Y.astype(int)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
        tf.keras.layers.Dense(units=24, activation='relu'),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    """
    마지막 활성함수만 각 노드가 모든 값을 참조하도록 softmax로 잡아주고, 나머지는 relu로 고정한다.
    출력값은 정답이 2행 array이므로 2로 잡아준다. 이항분류가 아닐 경우 여러개로 늘려주면 될 듯 하다.
    
    loss함수를 sparse_categorical_crossentropy로 설정하면 error가 나서, binary_crossentropy로 사용했다. binary이니 이항분류로 했을 때도 오류 안날듯
    """
    history = model.fit(train_X, train_Y, epochs=30, batch_size=32, validation_split=0.25)


    plt.figure(figsize=(12,4))

    """
    서브 그래프 1. loss함수를 보여주는 그래프를 그린다.
    """
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()


    """
    서브 그래프 2. 정확도를 보여주는 그래프를 그린다. history 변수 내에 정확도롤 측정할 수도 있는 것 같다.
    """
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], 'g-', label='accuracy')
    plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0.5 ,0.8)
    plt.legend()

    plt.show()

    model.evaluate(test_X, test_Y)
    """
    정답률은 대충 75%정도 나옴. loss는 0.46정도 (게임 시간의 중간값일 때)
    10분일 때는 대략 정답률 67%, loss는 0.56정도 나오는 것 같다.
    """