import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    terminated = False
    total_reward = 0
    episode = 1

    for _ in range(2000):
        print(env.render)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if done or truncated:
            print(f"Episode {episode} reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            episode += 1
    env.close()

    # while not terminated:
    #     print(env.render())
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     key = cv2.waitKey(100) # timeout to see the movement
    # print("Game Over!")
