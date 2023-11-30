# import the gym and stable baselines 3 libraries
import rlgym
from stable_baselines3.ppo import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction

# setup the RLGym environment
gym_env = rlgym.make(use_injector=True, self_play=True, obs_builder=DefaultObs(), state_setter=DefaultState(),
                     action_parser=DefaultAction())

# wrap the RLGym environment with the single instance wrapper
env = SB3SingleInstanceEnv(gym_env)

# create a PPO instance and start learning
learner = PPO(policy="MlpPolicy", env=env, verbose=1)
learner.learn(1_000)
