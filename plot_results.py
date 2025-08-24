# plot_results.py
"""
Plot training reward learning curve from Monitor CSV and success rate CSV.
Usage:
  python plot_results.py --monitor logs/monitor_reach.csv --success logs/success_reach.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_monitor(monitor_path):
    # monitor files have header lines starting with '#'
    with open(monitor_path, 'r') as f:
        lines = f.readlines()
    # skip header lines that start with '#'
    data_lines = [l for l in lines if not l.startswith('#')]
    if len(data_lines) == 0:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(data_lines)))
    return df

def plot_monitor(monitor_path, out="reward_plot.png"):
    df = parse_monitor(monitor_path)
    if df.empty:
        print("No data in monitor file:", monitor_path)
        return
    # cumulative timesteps vs episodic reward (we'll smooth)
    df['r_smooth'] = df['r'].rolling(window=20, min_periods=1).mean()
    plt.figure(figsize=(10,6))
    plt.plot(df['l'], df['r'], alpha=0.2, label="episodic r")
    plt.plot(df['l'], df['r_smooth'], label="smoothed (win=20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Training Reward")
    plt.tight_layout()
    plt.savefig(out)
    print("Saved", out)

def plot_success(success_csv, out="success_rate.png"):
    df = pd.read_csv(success_csv)
    # compute rolling success rate by episode
    # df: timesteps, episode, success
    grouped = df.groupby('episode')['success'].mean().reset_index()
    grouped['rate_smooth'] = grouped['success'].rolling(window=50, min_periods=1).mean()
    plt.figure(figsize=(10,6))
    plt.plot(grouped['episode'], grouped['success'], alpha=0.2, label="success per episode")
    plt.plot(grouped['episode'], grouped['rate_smooth'], label="smoothed success rate (50)")
    plt.xlabel("Episode")
    plt.ylabel("Success (0/1)")
    plt.legend()
    plt.title("Success Rate Over Training")
    plt.tight_layout()
    plt.savefig(out)
    print("Saved", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", default="logs/monitor_reach.csv")
    parser.add_argument("--success", default="logs/success_reach.csv")
    args = parser.parse_args()
    plot_monitor(args.monitor, out="reward_plot.png")
    plot_success(args.success, out="success_rate.png")
