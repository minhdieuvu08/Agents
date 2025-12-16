import gymnasium as gym

env = gym.make("ALE/Tetris-v5", render_mode="human")
obs, info = env.reset()
total_reward = 0
episode = 1

for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

    if done or truncated:
        print(f"Episode {episode} reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0
        episode += 1

env.close()
