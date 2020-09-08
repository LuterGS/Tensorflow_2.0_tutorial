import gym
import random
import seaborn as sns
import numpy as np

show_gym = gym.make('MountainCar-v0')

print(show_gym.observation_space)        # 에이전트가 참고할 환경, Box(2,)가 출력되므로, 에이전트는 2개의 실수값만 볼 수 있다.
print(show_gym.observation_space.low)    # 2개 실수값의 최소값, [-1.2, -0.07]이 출력된다
print(show_gym.observation_space.high)   # 2개 실수값의 최대값, [0.6, 0.7] 이 출력된다
                                    # 종합하면, 변수1의 최대-최소값은 0.6 ~ -1.2, 변수 2의 최대-최소값은 0.7 ~ -0.07이다.
                                    # 변수 1은 차의 X좌표, 변수 2는 차의 속도이다.  (궁금증. 2가 차의 속도인데, 속도가 방향 x, y축을 모두 포함한 속도인지, 아니면 x만 관찰했을떄의 속도인지?)
print(show_gym.action_space)             # 할 수 있는 행동은 이산적 3개이다 (0, 1, 2) - 각각 (왼쪽으로 이동, 정지, 오른쪽으로 이동)
print(show_gym._max_episode_steps)       # 한 에피소드가 무한히 계속되는 것을 방지하는 것. 즉, 200step이 넘으면 환경이 자동으로 종료된다.


class mountain_env:

    def __init__(self):
        self.env = gym.make('MountainCar-v0')

    def reset(self):
        self.env.reset()


class random_agent:

    def __init__(self):
        self.step = 0
        self.score = 0
        self.mountain_env = mountain_env()
        self.mountain_env.reset()

    def training(self):

        self.mountain_env.env.render()
        while True:
            action = self.mountain_env.env.action_space.sample()
            print("action : ", action)
            observation, reward, done, info = self.mountain_env.env.step(action)
            # observation은 action 한 뒤의 car의 위치
            # reward는 위치를 기반으로 한 location
            # done은 max_episode_step까지 왔는가
            # info는 기타
            print("Observation : ", observation)
            print("reward : ", reward)
            print("score : ", self.score)
            print("done : ", done)
            self.score += reward
            self.step += 1

            if done:
                print("done, cur_step : ", self.step)
                break

        self.mountain_env.env.close()


class get_answer_agent(random_agent):

    scores = []
    training_data = []
    accepted_scores = []
    required_score = -198

    def get_agents(self, max_episode_num):
        for i in range(max_episode_num):

            if i % 100 == 0:
                print(i)

            self.mountain_env.reset()

            score = 0
            game_memory = []
            previous_observation = []

            while True:
                action = self.mountain_env.env.action_space.sample()
                observation, reward, done, info = self.mountain_env.env.step(action)
                # 1회 행동함

                if len(previous_observation) > 0:
                    game_memory.append([previous_observation, action])
                # 만약 적어도 2회째 행동했다면, 저번 관측결과와 지금의 행동을 game_memory에 넣는다.


                if observation[0] > -0.2:
                    reward = 1
                # 이건 아마... observation 관측 결과 원하는 위치에 있을 때, reward를 1로 한다는 것

                previous_observation = observation
                score += reward

                if done:
                    #정답에 도달한게 하나라도 있으면 -198이상의 값을 가지게 되는 것
                    break

            self.scores.append(score)
            # 점수를 모두 저장한다.
            if score > self.required_score:
                self.accepted_scores.append(score)
                #만약, 통과한 에피소드가 있다면 그건 accepted_scores 에 저장한다.
                for data in game_memory:
                    self.training_data.append(data)
                    # 이건 순차적으로 쌓는건데 왜 순차적으로 쌓지?
                    # -> 되게 애매한데... 이게 결국은 성공한 에피소드에 한해서 결과 -> 행동을 나타낸다. 즉, 이전 결과를 보고 현재 행동을 뭐로 할지 정하는걸 그냥 성공한 에피소드들에서 긁어와서 학습시킨거임.

        self.scores = np.array(self.scores)
        print(self.scores.mean())
        print(self.accepted_scores)
        sns.distplot(self.scores, rug=True)


if __name__ == "__main__":
    agent1 = get_answer_agent()
    agent1.get_agents(3000)
