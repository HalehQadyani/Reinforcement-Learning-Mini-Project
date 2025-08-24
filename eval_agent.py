# eval_agent.py
"""
Load a saved model and run a few episodes, optionally saving frames as images.
Usage:
  python eval_agent.py --model models/ddpg_pandareach.zip --env PandaReach-v2 --episodes 5 --save_frames
"""

import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
import panda_gym
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", default="PandaReach-v2")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--out_dir", default="eval_output")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    env = gym.make(args.env, render_mode="rgb_array")  # try rgb_array to capture frames
    model = DDPG.load(args.model)

    for ep in range(args.episodes):
        obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
        done = False
        step = 0
        frames = []
        ep_reward = 0.0
        while not done and step < 1000:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            ep_reward += reward
            if args.save_frames:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
            step += 1

        print(f"Episode {ep} reward: {ep_reward}")

        if args.save_frames and frames:
            # save first 200 frames to a GIF
            pil_frames = [Image.fromarray(f) for f in frames[:200]]
            out_file = os.path.join(args.out_dir, f"ep{ep}.gif")
            pil_frames[0].save(out_file, save_all=True, append_images=pil_frames[1:], duration=40, loop=0)
            print(f"Saved GIF: {out_file}")

    env.close()

if __name__ == "__main__":
    main()
