import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
"""
FrozenLake 환경은
[시작점], [안전], [안전], [안전]
[안전], [위험], [안전], [위험]
[안전], [안전], [안전], [위험]
[위험], [안전], [안전], [도착]

타일로 이루어져있으며, 4x4 배열이고 [0][0]은 1 ~ [3][3]은 15의 인덱스를 가진다.
행동은 상하좌우로 이동할 수 있으며, 위험에 도착하거나 도착에 도착하면 중지된다.
도착에 도착했을때만 reward를 1 얻는다. 즉, 한 episode중 총 reward가 1 이면 이걸 클리어한 것이다.
"""
print(env.observation_space, env.action_space)
print(env._max_episode_steps)


# Q테이블 구성
Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    # n을 뒤에 붙여주면 number가 된다.

# 기타 매개변수 구성
learning_rate = 0.85
discount_value = 0.99
episode_num = 2000

# 각 에피소드의 결과를 저장할 리스트 (통과면 1, 아니면 0)
episode_result = []


for i in range(episode_num):
    # 환경 리셋 후 첫 번째 새로운 관찰을 얻음
    cur_state = env.reset()       # 리셋 후 0이 된다. -> 현재의 위치를 나타낸다.
    success_episode = 0
    d = False
    j = 0
    
    # Q 테이블 학습알고리즘
    while j < 99:   # max episode step 이 100이므로, 99번까지 돌음 (100은 왜 안될까)
        j += 1

        # 액션 선택
        action = np.argmax(Q_table[cur_state, :] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
            # Q_table[reseted_env, :] 는, Q_table에서 reset_env가 행인 열을 추출 -> 즉, [reset_env][0] ~ [reset_env][max] 를 추출하는것
            # np.random.randn(1, env_action_space.n)은, 평균0, 표준편차 1인 가우시안 표준정규분포 난수를  [1, env~] 크기만큼 생성
            # np.argmax는 인덱스 중 가장 높은 값을 추출
            # 이 프로세스 전체가 그냥 4개의 액션 중 랜덤한 값 하나를 추출하는 것이다.

        # 행동 (d_Qlearning.py 참고)
        new_state, reward, done, info = env.step(action)
            # new_state는 새로운 위치를 나타낸다 (행동한 이후의 위치)

        # Q테이블 업데이트
        Q_table[cur_state, action] += (learning_rate *
                                       (reward + discount_value * np.max(Q_table[new_state, :] - Q_table[cur_state, action]))
                                       )
            # 밸먼 방정식 (Q(s,a)* = r + lr(max(Q(s', a'))
            # 현재의 보상은, 진짜 현재의 보상값과, 할인된 미래의 최선의 행동의 보상값의 합이다.
            # 따라서 현재의 보상인 reward와, discound_value가 곱해진 미래값 중 최선의 값을 가져온다
            # (np.max는 array중 최대의 값을 return)
            # Q_table[cur_state, action]을 빼는 이유는 아마 과적합을 막기 위해라고 평가된다.

        # 성공한 에피소드는 1점을 얻으므로, 결국 success_episode는, 이 에피소드가 성공하면 1이 된다.
        success_episode += reward

        # 한번 이동한 처리가 끝났으므로 이동한걸 표시해준다.
        cur_state = new_state

        # 만약 에피소드가 끝나면 종료한다.
        if done:
            break

    # 현재 에피소드의 결과를 저장한다.
    episode_result.append(success_episode)


print("Score: ", str(sum(episode_result)/episode_num))
print("Final Q-Table :")
print(Q_table)

reshaped_reward_list = list(np.reshape(episode_result, [400, 5]))
for a in range(400):
    reshaped_reward_list[a] = sum(reshaped_reward_list[a])

plt.plot(episode_result)
plt.show()
plt.plot(reshaped_reward_list)
plt.show()