import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper
import numpy as np
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

        ACTION_FIRE = 1
        if action == ACTION_FIRE:
            aux_reward += 0.005 

        new_lives = info.get('lives', self.current_lives)
        if new_lives < self.current_lives:
            aux_reward -= 5.0
        
        self.current_lives = new_lives

        shaped_reward = reward + aux_reward
        
        return observation, shaped_reward, terminated, truncated, info

def make_env(render_mode=None):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode)
    env = AtariWrapper(env)
    env = SpaceInvadersRewardShaping(env) 
    return env

def main():
    MODEL_PATH = "ppo_spaceinvaders_rs"
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Error: Khong tim thay file {MODEL_PATH}.zip.")
        return

    try:
        load_env = make_env()
        model = PPO.load(MODEL_PATH, env=load_env)
    except Exception as e:
        print(f"Loi khi tai mo hinh: {e}")
        return

    eval_env = make_env()
    # Đánh giá trên 10 episodes
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print("\n--- RESULTS ---")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    eval_env.close()

    test_env = make_env(render_mode="human")
    obs, _ = test_env.reset()

    try:
        for _ in range(2000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            if terminated or truncated:
                obs, _ = test_env.reset()
    except KeyboardInterrupt:
        print("Stop.")
    finally:
        test_env.close()
        print("Done.")

if __name__ == "__main__":
    main()

# Agent có cải thiện hơn ở việc tránh đạn, thấy nhiều trường hợp sau khi đạn tới sát người thì next step nó sẽ né và tranh thủ trong lúc di chuyển rồi quay về bắn đạn để tăng khả năng bắn trúng
# Agent thường đứng yên ở góc trái sau đó bắn.