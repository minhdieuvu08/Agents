import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

def make_env():
    env = gym.make("ALE/Tetris-v5", render_mode=None)
    env = AtariWrapper(env)
    return env

def main():
    # create vectorized env
    vec_env = DummyVecEnv([make_env] * 4)
    # stack frame
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        clip_range=0.1,
        gamma=0.99,
        tensorboard_log="./ppo_tetris_tb/"
    )

    # Train
    model.learn(total_timesteps=3000000)  

    # Save model
    model.save("ppo_tetris")

    # Evaluate
    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print("Mean reward:", mean_reward, "Std:", std_reward)

    # Test agent
    test_env = gym.make("ALE/Tetris-v5", render_mode="human")
    test_env = AtariWrapper(test_env)
    obs, _ = test_env.reset()

    done = False
    total = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total += reward
        if terminated or truncated:
            break
    print("Test reward: ", total)
    test_env.close()
    

if __name__ == "__main__":
    main()
