# train_ddpg_reach.py
"""
Train DDPG on PandaReach-v2 (panda-gym v2 compatibility).
Saves:
 - model to ./models/ddpg_pandareach.zip
 - Monitor log to ./logs/monitor_reach.csv
 - success log (from callback) to ./logs/success_reach.csv
"""


import os
import gymnasium as gym
import numpy as np
import panda_gym  # registers environments
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from callbacks import SuccessLoggerCallback

# paths
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

ENV_ID = "PandaReach-v3"  # for panda-gym v2 compatibility; if you use v3 adjust as needed
TIMESTEPS = 400_000

def make_env(render=False):
    # render_mode in v3; for v2 simple use gym.make
    env = gym.make(ENV_ID)
    env = Monitor(env, filename="logs/monitor_reach.csv")
    return env

def main():
    env = make_env(render=False)

    # DDPG needs action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        buffer_size=100000,
        learning_rate=1e-3,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        tensorboard_log="./tensorboard/ddpg_reach"
    )

    # Evaluation callback: evaluate every 5k steps on separate env
    eval_env = gym.make(ENV_ID)
    eval_env = Monitor(eval_env, filename=None)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)  # adjust threshold for your reward scale
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=5000,
        best_model_save_path="./models/best_ddpg_reach",
        verbose=1,
        n_eval_episodes=5,
    )

    success_cb = SuccessLoggerCallback(out_csv="logs/success_reach.csv", verbose=0)

    model.learn(total_timesteps=TIMESTEPS, callback=[eval_callback, success_cb])

    model.save("models/ddpg_pandareach")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
