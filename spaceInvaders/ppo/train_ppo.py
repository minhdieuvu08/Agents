import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

def make_env():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = AtariWrapper(env)
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
    )

    # Train
    model.learn(total_timesteps=200000)  

    # Save model
    model.save("ppo_spaceinvaders")

    # Evaluate
    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print("Mean reward:", mean_reward, "Std:", std_reward)

    # Test agent
    test_env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    obs, _ = test_env.reset()

    for _ in range(2000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated or truncated:
            obs, _ = test_env.reset()

if __name__ == "__main__":
    main()
