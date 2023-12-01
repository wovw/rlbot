import rlgym
from stable_baselines3 import PPO
from rocket_learn.ppo import PPO

from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

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
obs_builder = DefaultObs()
obs_builder = NectoObsBuilder(Scoreboard(), None, 1)
state_setter = DefaultState()
action_parser = DefaultAction()
action_parser = NectoAction()

# Make the default rlgym environment
# env = rlgym.make()

# Works
# env = rlgym.make(
#         reward_fn=reward_fn,
#         terminal_conditions=[TimeoutCondition(225)],
#         obs_builder=DefaultObs(),
#         state_setter=DefaultState(),
#         action_parser=DefaultAction()
# )

env = rlgym.make(
        reward_fn=reward_fn,
        terminal_conditions=[TimeoutCondition(225)],
        obs_builder=obs_builder,
        state_setter=DefaultState(),
        action_parser=action_parser
)

# Doesn't work: ValueError: The environment is of type <class 'rlgym.envs.match.Match'>, not a Gymnasium environment. In this case, we expect OpenAI Gym to be installed and the environment to be an OpenAI Gym environment.
# env = Match(
#         reward_function=DefaultReward(),
#         terminal_conditions=[TimeoutCondition(225)],
#         obs_builder=DefaultObs(),
#         state_setter=DefaultState(),
#         action_parser=DefaultAction()
#     )


# Initialize PPO from rocket_learn
alg = PPO(
        rollout_gen,
        agent,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        ent_coef=logger.config.ent_coef,
        logger=logger,
    )

# Train our agent!
model.learn(total_timesteps=10_000)

print("Finished Learning, Saving Model...")

model.save("./checkpoints/most_trained_bot.zip")
