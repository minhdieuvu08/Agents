import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os 

class SpaceInvadersRewardShaping(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_lives = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.current_lives = info.get('lives', 5)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        aux_reward = 0.0
        
        ACTION_RIGHT = 2
        ACTION_LEFT = 3
        if action in [ACTION_RIGHT, ACTION_LEFT, 4, 5]:
            aux_reward += 0.001 

        new_lives = info.get('lives', self.current_lives)
        if new_lives < self.current_lives:
            aux_reward -= 1.0  
        
        self.current_lives = new_lives

        shaped_reward = reward + aux_reward
        
        return observation, shaped_reward, terminated, truncated, info

def make_env(use_reward_shaping=False):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = AtariWrapper(env)
    if use_reward_shaping:
        env = SpaceInvadersRewardShaping(env) 
    return env

class RewardCollectorCallback(BaseCallback):
    def __init__(self, check_freq: int, model_label: str, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.model_label = model_label 

        self.recent_rewards = deque(maxlen=100)
        self.mean_rewards = []
        self.timesteps = []
        self.current_episode_reward = 0.0
        self.last_print_timestep = 0
        self.print_interval = 10000
        
    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0] 
        self.current_episode_reward += reward

        if self.locals['dones'][0]:
            self.recent_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            
            if len(self.recent_rewards) == self.recent_rewards.maxlen:
                current_mean = np.mean(self.recent_rewards)
                self.mean_rewards.append(current_mean)
                self.timesteps.append(self.num_timesteps)

                if self.num_timesteps - self.last_print_timestep >= self.print_interval:
                    print(f"[{self.model_label}] Timestep: {self.num_timesteps}/{self.model._total_timesteps} | Mean Reward (100 episodes): {current_mean:.2f}")
                    self.last_print_timestep = self.num_timesteps

        return True 


def plot_learning_curves(timesteps_1, rewards_1, label_1, timesteps_2, rewards_2, label_2, total_timesteps, filename="ppo_rs_comparison_v2.png"):
    """Plots a comparison chart of the learning curves."""
    save_dir = "../../assets/ppo"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    
    plt.plot(timesteps_1, rewards_1, label=label_1, color='red', linewidth=1.5)
    plt.plot(timesteps_2, rewards_2, label=label_2, color='blue', linewidth=1.5)
    
    plt.title(f'PPO Space Invaders Mean Reward Comparison ({total_timesteps} Timesteps)')
    plt.xlabel('Timesteps')
    plt.ylabel(f'Mean Reward')
    
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(full_path, dpi=300)
    print(f"\nChart saved at: {filename}")
    plt.show()


def main():
    LOG_DIR = "./ppo_logs/"
    os.makedirs(LOG_DIR, exist_ok=True)
    TOTAL_TIMESTEPS = 200000

    # A. TRAIN MODEL WITH REWARD SHAPING (RS)
    print("--- 1. Starting PPO training WITH Reward Shaping (RS) ---")
    env_rs = make_env(use_reward_shaping=True)

    model_rs = PPO("CnnPolicy", env_rs, verbose=0, device="auto",
                   learning_rate=2.5e-4, n_steps=128, batch_size=128, gamma=0.99)
    
    callback_rs = RewardCollectorCallback(check_freq=100, model_label="RS_MODEL") 
    
    model_rs.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_rs)
    model_rs.save("ppo_spaceinvaders_rs_v2")
    print("Training WITH RS complete. Model saved.")

    # B. TRAIN MODEL WITHOUT REWARD SHAPING (NO RS)
    print("\n--- 2. Starting PPO training WITHOUT Reward Shaping (NO RS) ---")
    env_no_rs = make_env(use_reward_shaping=False)

    model_no_rs = PPO("CnnPolicy", env_no_rs, verbose=0, device="auto",
                      learning_rate=2.5e-4, n_steps=128, batch_size=128, gamma=0.99)
    
    callback_no_rs = RewardCollectorCallback(check_freq=100, model_label="NO_RS_MODEL")
    
    model_no_rs.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_no_rs)
    model_no_rs.save("ppo_spaceinvaders_no_rs")
    print("Training complete. Model saved.")
    
    plot_learning_curves(
        timesteps_1=callback_rs.timesteps,
        rewards_1=callback_rs.mean_rewards,
        label_1='PPO with Reward Shaping (RS)',
        timesteps_2=callback_no_rs.timesteps,
        rewards_2=callback_no_rs.mean_rewards,
        label_2='PPO (NO RS)',
        total_timesteps=TOTAL_TIMESTEPS
    )

if __name__ == "__main__":
    main()