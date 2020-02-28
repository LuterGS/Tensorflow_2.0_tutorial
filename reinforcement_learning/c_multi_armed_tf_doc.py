import abc
import numpy as np
import tensorflow as tf
import scipy.special

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step



#tf.reset_default_graph()

tf.compat.v1.enable_resource_variables()
tf.compat.v1.enable_v2_behavior()
nest = tf.compat.v2.nest

"""
Environments

에이전트에게 상태를 주고, 액션을 받으며, 한 에피소드 종료 후 상태 리셋하는 클래스
말 그대로의 "환경"

그렇지만 밴딧 문제에서는 에피소드는 없다. 한 학습 이후 무조건 초기화되기 때문에.

그래서 모든 관찰, 모든 트레이닝이 독립적임을 확실히 하기 위해, 
서브클래스인 PyEnvironment와 TFEnvironment를 import 한다.
       바로, BanditPyEnvironment, BanditTFEnvironment 이다!
       
       
observation : 한 action을 주고 환경변화를 받는 것이 한 obsevation 이다.
"""


class BanditPyEnvironment(py_environment.PyEnvironment):

    def __init__(self, observation_spec, action_spec):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        super(BanditPyEnvironment, self).__init__()

    # Helper func
    def action_spec(self):
        return self.action_spec()

    def observation_spec(self):
        return self.observation_spec()

    def _empty_observation(self):
        return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype), self.observation_spec())

    #이 두개는 자식 클래스에서 변경되면 안됨!
    #_reset 함수는 observation시의 환경을 reset하는 함수. ts.restart로 적혀있는걸 봐서...
    def _reset(self):
        return ts.restart(self._observe(), batch_size=self.batch_size)

    # step 함수는 bandit의 한 번의 행동을 의미하는것 같음. 액션 후 환경에 반영해주는게 없다면, 이건 그냥 액션만을 취하는 함수일수도 있음
    def _step(self, action):
        reward = self._apply_action(action)
        return ts.termination(self._observe(), reward)

    #이 두개는 자식클래스에게 implemented 되야 함
    @abc.abstractclassmethod
    def _observe(self):
        """관찰 결과 리턴"""

    @abc.abstractclassmethod
    def _apply_action(self, action):
        """
        환경에 액션 부여후 보상 return
        아니면, 그냥 apply_action이라는 의미대로 액션을 취했을 때 환경을 변경하는 것이라고 할 수도 있을듯
        """



"""
간단한 환경을 설계해보자. 한 번의 observation이 -2 ~ 2 사이
3가지 가능한 액션이 있음 (0,1,2)
보상은 액션 * observation
"""

class SimplePyEnvironment(BanditPyEnvironment):

    def __init__(self):
        action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'
        )
        observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=-2, maximum=2, name='observation'
        )
        super(SimplePyEnvironment, self).__init__(observation_spec, action_spec)

    def _observe(self):
        self._observation = np.random.randint(-2, 3, (1,))
        return self._observation

    def _apply_action(self, action):
        return action * self._observation



environment = SimplePyEnvironment()
observation = environment.reset().observation
print("observation: %d" % observation)

action = 2

print("action: %d" % action)
reward = environment.step(action).reward
print("reward: %f" % reward)

test = scipy.special.xlog1py(1, 2)