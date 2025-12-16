import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.core import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class SpaceInvadersRewardShaping(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_lives = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.current_lives = info.get('lives', 5)
        return observation, info

    def step(self, action):
        return self.env.step(action)

def make_eval_env(render_mode='human'):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode) 
    env = AtariWrapper(env)
    return env

def collect_evaluation_rewards(model_path: str, num_episodes: int = 20):
    env_test = make_eval_env(render_mode=None) 
    
    try:
        model = PPO.load(model_path, env=env_test)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Skipping.")
        return [], 0
    
    episode_rewards = []
    
    print(f"Starting data collection for {model_path}...")
    
    for _ in range(num_episodes):
        obs, info = env_test.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_test.step(action)
            done = terminated or truncated
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
    env_test.close()
    mean_reward = np.mean(episode_rewards)
    print(f"Collected {len(episode_rewards)} episodes. Raw Mean Reward: {mean_reward:.2f}")
    
    return episode_rewards, mean_reward

def plot_comparison(rs_rewards, no_rs_rewards, rs_mean, no_rs_mean):
    plt.figure(figsize=(10, 6))
    
    episodes = np.arange(1, len(rs_rewards) + 1)
    
    plt.plot(episodes, rs_rewards, 'r-', alpha=0.6, label=f'Reward Shaping Model')
    
    plt.plot(episodes, no_rs_rewards, 'b-', alpha=0.6, label=f'No Reward Shaping Model')
    
    plt.title('Performance Comparison')
    plt.xlabel('Episode')
    plt.ylabel(' Reward')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("evaluation_raw_reward_comparison.png", dpi=300)
    print("\nEvaluation plot saved at: evaluation_raw_reward_comparison.png")
    plt.show()

def run_visualization(model_path: str, render_mode='human'):
    print(f"\n--- Running visualization for {model_path} ---")
    env_render = make_eval_env(render_mode=render_mode)
    
    try:
        model = PPO.load(model_path, env=env_render)
    except FileNotFoundError:
        print("Model not found.")
        return

    obs, info = env_render.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated
    
    env_render.close()
    print("Finished.")

if __name__ == "__main__":
    
    MODEL_RS_PATH = "ppo_spaceinvaders_rs_v2.zip"
    MODEL_NO_RS_PATH = "ppo_spaceinvaders_no_rs.zip"
    EVAL_EPISODES = 100
    
    rs_rewards, rs_mean = collect_evaluation_rewards(MODEL_RS_PATH, num_episodes=EVAL_EPISODES)
    
    no_rs_rewards, no_rs_mean = collect_evaluation_rewards(MODEL_NO_RS_PATH, num_episodes=EVAL_EPISODES)

    if rs_rewards and no_rs_rewards:
        plot_comparison(rs_rewards, no_rs_rewards, rs_mean, no_rs_mean)
    
    run_visualization(MODEL_RS_PATH, render_mode='human')
    run_visualization(MODEL_NO_RS_PATH, render_mode='human')

    print("\nEvaluation completed.")