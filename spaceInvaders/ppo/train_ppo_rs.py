import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper
import numpy as np

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

        ACTION_FIRE = 1
        if action == ACTION_FIRE:
            aux_reward += 0.005 

        new_lives = info.get('lives', self.current_lives)
        if new_lives < self.current_lives:
            aux_reward -= 5.0
            print(f"Agent lost a life! Applying penalty: {aux_reward}")
        
        self.current_lives = new_lives

        shaped_reward = reward + aux_reward
        
        return observation, shaped_reward, terminated, truncated, info

def make_env():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = AtariWrapper(env)
    env = SpaceInvadersRewardShaping(env) 
    return env

def main():
    env = make_env()

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        clip_range=0.1,
        gamma=0.99,
        device="auto"
    )

    print("Starting PPO training with Reward Shaping...")
    model.learn(total_timesteps=200000)  

    model.save("ppo_spaceinvaders_rs")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()