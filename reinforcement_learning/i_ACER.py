import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CartPole-v1')

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

observation = env.reset()
for i in range(1000):
    action, _states = model.predict(observation)
    observation, rewards, done, info = env.step(action)
    env.render()