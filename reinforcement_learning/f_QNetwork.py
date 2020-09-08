import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

env = gym.make('FrozenLake-v0')

# input shape가 16이고, ouput이 4이기 때문에, 16*4 인 가중치가 만들어지며, 이게 Q Table이라고 할 수 있다.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, input_shape=[16, ]),
    tf.keras.layers.Dense(units=4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.5), loss='mse', metrics=['accuracy'])
model.summary()



discount_value = 0.99
epsilon = 0.8
epsilon_minimum = 0.01
episode_num = 2000
# epsilon은 일종의 랜덤 행동 지수이며, random으로 행동하는 수치를 의마한다. 갈수록 감소한다.


step_list = []
reward_list = []

for i in range(episode_num):

    # epsilon을 줄여나가는 절차
    epsilon *= 0.92
    epsilon = max(epsilon, epsilon_minimum)

    cur_state = env.reset()
    success_episode = 0
    step = 0

    while step < 99:
        step += 1

        # 신경먕에 현재 위치의 Q값을 질의함
        nn_result = model.predict(np.array([np.eye(16)[cur_state]]))

        # 랜덤 행동을 할건지 측정해서
        if random.random() < epsilon:       # 랜덤한 행동을 해야할 경우
            action = random.randint(0, 3)   # 그냥 랜덤한 0~3값을 고르고
            nn_value = nn_result[0][action] # Q값도 랜덤한 값의 Q값을 고름
        else:                               # 만약 제대로된 행동을 해야할 경우
            action = nn_result.argmax()     # 가장 높은 값을 고리고
            nn_value = np.max(nn_result)    # 그 Q값을 가져옴

        # 위에서 정한 행동을 토대로 실제 환경에서 행동 계시
        new_state, reward, done, info = env.step(action)

        # 밸먼방정식으로 미래의 값을 받아와야하기 때문에, new_state를 넣어서 해당 위치에서의 최선의 Q값을 가져옴.
        update_value = np.max(model.predict(
            np.array([np.eye(16)[new_state]]))
        )
        update = reward + discount_value * (update_value - nn_value)
        
        # 1회 신경망에 결과 학습
        model.train_on_batch(
            np.array([np.eye(16)[cur_state]]),
            np.array([np.eye(4)[action] * update]),
        )

        # 기타 처리
        success_episode += reward
        cur_state = new_state

        if done:
            break

    # 성공 여부에 따라 로그 출력
    if success_episode == 1:
        print(str(i), " success!!!!!!!!")
    else:
        print(str(i), " train")

    # 전체 결과값 저장
    step_list.append(step)
    reward_list.append(success_episode)

# 그래프출력 보기좋게 10개씩 그룹화해서 성공한 개수를 합산해서 보여주도록 함
reshaped_step_list = list(np.reshape(step_list, [200, 10]))
for a in range(200):
    reshaped_step_list[a] = sum(reshaped_step_list[a])

reshaped_reward_list = list(np.reshape(reward_list, [200, 10]))
for a in range(200):
    reshaped_reward_list[a] = sum(reshaped_reward_list[a])

# 그래프로 보여줌
plt.plot(reshaped_step_list)
plt.show()
plt.plot(reshaped_reward_list)
plt.show()


print(step_list)
print(reward_list)
