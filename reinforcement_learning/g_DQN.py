"""
간단한 원리부터!

QN은 그냥 신경망 자체가 Q Network라서 얻어지는 데이터로 즉시 학습했지만
DQN은 별도의 메모리에 데이터를 저장한 뒤에 어느정도 데이터가 쌓이면 랜덤하게 샘플을 뽑아 학습시킨다
    -> 일종의 데이터 오버샘플링?

Q(s, a) = Q(s, a) + a(R + y max Q(s', a') - Q(s, a))
이게 QNetwork 식이다 (밸먼 방정식, 왜 Q(s, a) 를 빼는지 드디어알겠네;;

여기서, Q(s, a)를 구하려고 네트워크에 접근하는 순간 Q(s', a')도 변해서, 목표값이 흔들리게 된다.

따라서 안쪽 네트워크를 타깃 네트워크로 분리해서, maxQ(s', a')만 "고정된" 가중치를 사용한다.
    -> 너무 시류와 동떨어지면 안되니까 일정 주기마다 가중치를 업데이트해준다.

"""

import gym_2048
import gym

env = gym.make('2048-v0')
observation = env.reset()

print(observation)
print(env.observation_space)
print(env.action_space)

observation, _, _, _ = env.step(0)
print(observation)

"""
gym_2048의 환경은
2차원 배열 4x4가 주어지며, 각 값은 2048의 블록 값과 같다.
행동은 0, 1, 2, 3 4개이며, 각각 상하좌우로 미는 것을 의미한다.
"""