import gym
import numpy as np

show_env = gym.make('MountainCarContinuous-v0')

print(show_env.observation_space)           # 형태는 2차원 박스
print(show_env.observation_space.low)       # 이건 기존과 동일
print(show_env.observation_space.high)
print()
print(show_env.action_space)                # 행동의 형태가 1차원 값
print(show_env.action_space.low)            # 최대 최소값이 -1 ~ 1
print(show_env.action_space.high)
print()
print(show_env._max_episode_steps)          # 최대 999 step까지 가능


def random_agent(episode_num):
    show_env.reset()
    score = 0
    step = 0

    for i in range(episode_num):
        action = show_env.action_space.sample()
        observation, reward, done, info = show_env.step(action)
        # 여기서 reward는 행동의 제곱에 0.1을 곱한 값의 음수값을 준다. 즉, 많이 행동할수록 점수가 더 음수가 된다
        # 단, 깃발에 도달하면 100점을 받는다.
        # 최대치인 999까지 하면 보통 -32정도가 나온다
        # MountainCarContinous-v0은 연속된 100회의 에피소드에서 +90이상의 누적 보상을 얻는다면 환경을 풀었다고 판단한다. -> 100회에서 거의 실패하지 않고 깃발에 도달해야함.

        previous_observation = observation
        score += reward
        step += 1

        if done:
            break
    print(score, step)


state_grid_count = 10
action_grid_count = 6

q_table = []
for i in range(state_grid_count):
    q_table.append([])
    for j in range(state_grid_count):
        q_table[i].append([])
        for k in range(action_grid_count):
            q_table[i][j].append(1e-4)

actions = range(action_grid_count)
actions = np.array(actions).astype(float)
actions *= ((show_env.action_space.high - show_env.action_space.low) / (action_grid_count - 1))
actions += show_env.action_space.low

print(actions)



if __name__ == "__main__":
    random_agent(10000)