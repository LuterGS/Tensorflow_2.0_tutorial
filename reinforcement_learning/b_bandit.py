import tensorflow as tf
import numpy as np

"""
밴딧 문제

    강화학습의 가장 기본 모델은 밴딧 문제이다.
    예를 들어, 레버를 누르면 주어진 알고리즘에 따라 점수를 주는 머신이 있다고 하자.
    어느 머신이 가장 높은 점수를 줄 것인가? 를 스스로 판단하고 행동하는 모델이 밴딧 모델이다.
    
    행동하게끔 설계된 신경망, 알고리즘을 에이전트라고 한다. 따라서, 앞으로 대부분의 네이밍을 에이전트라고 하겠다.
    
    결과적으로, 에이전트는 가장 높은 점수를 주는 머신의 레버만을 계속 당기게 된다.
    
    
    
밴딧 문제의 의존성
    
    액션 의존성 : 각 액션은 다른 보상을 가져온다. 즉, 보상은 액션에 종속적이다. 밴딧 문제에서 어떤 머신의 손잡이를 당기느냐에 따라 보상이 달라진다.
    시간 의존성 : 보상은 일정 시간 이후에 주어지는 경우가 있다. 에이전트가 체스 경기를 둔다고 생각해보자. 현재 둔 수가 나중에 어떤 결과를 가져올 지 모른다.
                즉, 현재 둔 수의 완전한 보상이 늦을 수 있다. 나중에 그게 결과로 나타나야 결과적으로 보상의 정도를 파악할 수 있다.
                즉, 에이전트는 보상을 지연된 시점에 학습하게 되고, 그 시기도 드물다.
                일반 지도학습처럼, 시간개념이 존재하지 않는 학습이 아니다.
    상태 의존성 : 액션에 따른 보상은 상태에 따라 좌우된다. 주위 상태를 살피지 않고 같은 수를 남발할 수는 없는 법이다. 
                상태에 따른 행동이 달라지므로, 에이전트는 행동 전에 상태를 확인해야 한다.
    
                
                
멀티암드, N-암드 밴딧

    시작으로, 팔이 달린 머신을 설계하는 것이 좋은 이유는, 시간 의존성과 상태 의존성이 없기 떄문이다.
    즉각적인 보상이 나오므로 시간에 따른 의존성이 없고, 주위 상태는 그대로이기 떄문에 상태에 따른 의존성도 없다. 즉, 액션 의존성만 존재한다.
    
    학습시 경사하강법을 통해 학습한다.
    에이전트의 가치함수 (value function)을 학습하는 다른 방법들이 있고, 나중에 천천히 다룰 것이다.
        가치함수는 일종의 가중치라고 생각하면 된다.
    
        
        
정책 경사
    
    정책경사 네트워크는, 분명한 출력값을 산출하는 네트워크다.
    Supervised Learning에서 많이 쓰듯이, 이 신경망은 분명한 출력값을 산출한다.
    
    밴딧 문제로 돌아와, 각 머신의 손잡이에 가중치를 하나하나 할당해주면 된다.
    모두를 1로 초기화시키면, 어느정도 낙관적으로 바라보고 있다고 가정한다.
    
    이후, 단순히 e의 확률로 랜덤한 확률을 취하는, greedy한 정책을 가지는 손잡이 하나를 생각한다.
    이건, 예이전트는 어느정도 에이전트가 예상한 가치함수에 따라 행동하지만, 가끔씩 e의 확률로 랜덤하게 액션을 선택한다 (더 높은 보상을 주는 밴딧을 찾아내기 위해)
    
    여기서 Loss = -log(pie) * A (Advantage) 이다.
    즉, A는 에이전트가 취한 액션이 어떤 기준선 (에이전트가 상정한 가중치) 보다 얼마나 더 나은지의 정도를 의미한다.
    해당 함수를 이용해 에이전트를 학습시킨다.
"""


class multi_armed_bandit:

    def __init__(self):
        self.bandit_arms = [0.2, 0, -0.2, 2] #랜덤한 정규분포 수 중 하나보다 값이 크면 1점을 얻으므로, 여기선 4번째 손잡이의 보상이 가장 크다.
        self.num_bandit_arms = 4
        self.weights = tf.Variable(tf.ones([self.num_bandit_arms]))
        self.output = tf.nn.softmax(self.weights)


        self.action_holder = tf.Variable(tf.ones[1])

        self.responsible_output = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_output)*reward_holder)

        self.reward_result = np.zeros(self.num_bandit_arms)


    def pullBandit(self, bandit):
        result = np.random.randn(1)
        if result < bandit:
            return 1
        else:
            return -1


    def training(self):
        for i in range(1000):
            actions = tf.nn.softmax(self.weights)
            a = np.random.choice(actions, p=actions)
            action = np.argmax(actions == a)

            reward = self.pullBandit(self.bandit_arms[action])

            




if __name__=="__main__":

    test = multi_armed_bandit()
    print(test.weights, test.output)














