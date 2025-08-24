# train_ddpg_push.py
"""
Train DDPG on PandaPush. Same structure as reach script but tuned hyperparams.
Saves model to: ./models/ddpg_pandapush.zip
"""

import os
import gymnasium as gym
import numpy as np
import panda_gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from callbacks import SuccessLoggerCallback

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

ENV_ID = "PandaPush-v3"
TIMESTEPS = 800_000  # pushing tends to need more samples

def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env, filename="logs/monitor_push.csv")
    return env

def main():
    env = make_env()
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))

    model = DDPG(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        buffer_size=200000,
        learning_rate=5e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        tensorboard_log="./tensorboard/ddpg_push"
    )

    success_cb = SuccessLoggerCallback(out_csv="logs/success_push.csv", verbose=0)
    model.learn(total_timesteps=TIMESTEPS, callback=success_cb)

    model.save("models/ddpg_pandapush")
    env.close()

if __name__ == "__main__":
    main()
