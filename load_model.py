import rlgym
from stable_baselines3 import PPO

env = rlgym.make()
model = PPO("MlpPolicy", env=env, verbose=1)

model.load("checkpoints.zip")

print(model.get_parameters())
