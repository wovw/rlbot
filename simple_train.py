import rlgym
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.obs_builders import DefaultObs, AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from terminal import NectoTerminalCondition

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward
from reward import NectoRewardFunction
from action import NectoAction
from obs import NectoObsBuilder

from rocket_learn.utils.scoreboard import Scoreboard

from rlgym.envs import Match

ep_len_minutes = 4.0
game_speed = 100
tick_skip = 8
spawn_opponents = True
random_resets = False
team_size = 1
reward_fn = LiuDistancePlayerToBallReward()
reward_fn = NectoRewardFunction()
terminal_conditions = [TimeoutCondition(225)]
terminal_conditions = [NectoTerminalCondition()]
obs_builder = DefaultObs()
obs_builder = NectoObsBuilder(Scoreboard(), None, 1)
obs_builder = AdvancedObs()
state_setter = DefaultState()
action_parser = DefaultAction()
action_parser = NectoAction()

# Make the default rlgym environment
# env = rlgym.make()

# Works
rl_env = rlgym.make(
        reward_fn=reward_fn,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        # state_setter=DefaultState(),
        # action_parser=DefaultAction(), # These are good. Technically same as Necto
        # team_size=1, # Not working
        # spawn_opponents=False, # Not working
        use_injector=True,
        self_play=True,
        auto_minimize=False
)

env = SB3SingleInstanceEnv(rl_env)

# Initialize PPO from stable_baselines3
# model = PPO("MlpPolicy", env=env, verbose=1)

model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=30, gamma=0.995,
            gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100,
            tensorboard_log=None, policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True)

# Train our agent!
model.learn(total_timesteps=1_000)

print("Finished Learning, Saving Model...")

model.save("./checkpoints/trained_bot.zip")
