import rlgym
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from reward import NectoRewardFunction
from rlgym.utils.obs_builders import DefaultObs, AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from terminal import NectoTerminalCondition
import time

reward_fn = NectoRewardFunction()
obs_builder = AdvancedObs()
state_setter = DefaultState()
action_parser = DefaultAction()
terminal_conditions = [NectoTerminalCondition()]

# Continuous Training
for step in range(1, 1000):
    rl_env = rlgym.make(
            reward_fn=reward_fn,
            terminal_conditions=terminal_conditions,
            obs_builder=obs_builder,
            state_setter=DefaultState(),
            action_parser=DefaultAction(), # These are good. Technically same as Necto
            team_size=1, # Not working
            # spawn_opponents=False, # Not working
            use_injector=True,
            self_play=True,
            auto_minimize=True
    )

    env = SB3SingleInstanceEnv(rl_env)

    print("Setting Model")
    # model = PPO("MlpPolicy", env=env, verbose=1)

    model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=30, gamma=0.995,
                gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100,
                tensorboard_log=None, policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True)

    print("Loading model")
    model.load("checkpoints/trained_bot.zip")

    print(model.get_parameters())

    # Restart training
    print("LEARNING")
    # Train our agent!
    model.learn(total_timesteps=1_000)

    print(f"Finished Learning at step {step/10} Million, Saving Model...")

    print(model.get_parameters())

    model.save("./checkpoints/trained_bot.zip")

    env.close()
    time.sleep(2)
