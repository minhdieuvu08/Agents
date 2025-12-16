import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder

MODEL_PATH = "/home/minh/projects/agents/spaceInvaders/ppo_spaceinvaders.zip"
SAVE_VIDEO = False # bi loi khi luu video             
VIDEO_PATH = "videos/"
MAX_STEPS = 5000               

# Create ENV 
def make_env():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human" if not SAVE_VIDEO else "rgb_array")
    env = AtariWrapper(env)  
    if SAVE_VIDEO:
        os.makedirs(VIDEO_PATH, exist_ok=True)
        env = VecVideoRecorder(
            env,
            VIDEO_PATH,
            record_video_trigger=lambda x: x == 0,
            video_length=MAX_STEPS,
            name_prefix="ppo_spaceinvaders"
        )
    return env

env = make_env()

# model
model = PPO.load(MODEL_PATH)

# test agent
obs, _ = env.reset()
total_reward = 0
episode = 1

for step in range(MAX_STEPS):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode {episode} reward: {total_reward}")
        obs, _ = env.reset()
        total_reward = 0
        episode += 1

env.close()
print("Finished testing agent!")
