import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from train import DiscretizedObservationWrapper, QLearningAgent, GRID_SIZE

# Note: These model files are ignored by git. 
# Run train_q_shaping_v3.py first to generate them.
STANDARD_MODEL = "q_table_spaceinvaders.pkl"
SHAPED_MODEL = "q_shaping_v3.pkl" 
TEST_EPISODES = 100

def get_rewards(agent, env, num_episodes):
    rewards = []
    print(f"Running {num_episodes} episodes...", end="", flush=True)
    for i in range(num_episodes):
        state, _ = env.reset(seed=i)
        total = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
        rewards.append(total)
        if (i+1)%10==0: print(".", end="", flush=True)
    print(" Done!")
    return np.array(rewards)

def main():
    if not os.path.exists(STANDARD_MODEL) or not os.path.exists(SHAPED_MODEL):
        print("Missing model files. Please check filenames.")
        print(f"Standard: {STANDARD_MODEL}")
        print(f"Shaped:   {SHAPED_MODEL}")
        return

    # Setup environment
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = DiscretizedObservationWrapper(env, grid_size=GRID_SIZE)
    action_size = env.action_space.n

    # Load Agents
    print(">>> Loading Agents...")
    agent_std = QLearningAgent(action_size)
    agent_std.load(STANDARD_MODEL)
    agent_std.epsilon = 0.0

    agent_shaped = QLearningAgent(action_size)
    agent_shaped.load(SHAPED_MODEL)
    agent_shaped.epsilon = 0.0

    # Get Data
    print("\n>>> Testing Standard Agent:")
    r_std = get_rewards(agent_std, env, TEST_EPISODES)
    
    print("\n>>> Testing Shaped Agent:")
    r_shaped = get_rewards(agent_shaped, env, TEST_EPISODES)
    
    env.close()

    mean_std = np.mean(r_std)
    mean_shaped = np.mean(r_shaped)
    wins = np.sum(r_shaped > r_std)
    draws = np.sum(r_shaped == r_std)
    losses = np.sum(r_shaped < r_std)
    win_rate = (wins / TEST_EPISODES) * 100

    print("\n" + "="*50)
    print("SUCCESS EVALUATION TABLE")
    print("="*50)
    print(f"1. AVERAGE SCORE:")
    print(f"   - Standard: {mean_std:.2f}")
    print(f"   - Q-Shaping: {mean_shaped:.2f}")
    
    print(f"\n2. HEAD-TO-HEAD WIN RATE:")
    print(f"   - Q-Shaping Wins: {wins}/{TEST_EPISODES} ({win_rate:.1f}%)")
    
    print("\n>>> CONCLUSION:")
    if mean_shaped > mean_std:
        print("SUCCESS: Q-Shaping improved average score.")
    else:
        print("NEEDS IMPROVEMENT: Standard agent performed better.")
    print("="*50)

    # --- PLOTTING ANALYSIS ---
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 15))

    episodes = range(1, TEST_EPISODES + 1)

    # Raw Scores per Episode
    ax0.plot(episodes, r_std, 'k-o', label=f'Standard (Avg: {mean_std:.1f})', alpha=0.5, linewidth=1.5, markersize=4)
    ax0.plot(episodes, r_shaped, 'r-s', label=f'Q-Shaping (Avg: {mean_shaped:.1f})', linewidth=2.0, markersize=5)
    
    ax0.vlines(x=episodes, ymin=np.minimum(r_std, r_shaped), ymax=np.maximum(r_std, r_shaped), 
               colors='gray', linestyles=':', alpha=0.3)
    
    ax0.set_title('Raw Reward per Episode Comparison', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Reward')
    ax0.legend()
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.set_xlim(1, TEST_EPISODES)

    # Cumulative Sum
    cumsum_std = np.cumsum(r_std)
    cumsum_shaped = np.cumsum(r_shaped)
    
    ax1.plot(episodes, cumsum_std, 'k--', label='Standard Cumulative')
    ax1.plot(episodes, cumsum_shaped, 'r-', linewidth=2, label='Shaping Cumulative')
    
    ax1.fill_between(episodes, cumsum_std, cumsum_shaped, where=(cumsum_shaped > cumsum_std), 
                     color='red', alpha=0.1, label='Shaping Leads')
    ax1.fill_between(episodes, cumsum_std, cumsum_shaped, where=(cumsum_shaped <= cumsum_std), 
                     color='gray', alpha=0.1, label='Standard Leads')

    ax1.set_title('Cumulative Reward', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Accumulated Score')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(1, TEST_EPISODES)

    # PLOT 3: Score Difference 
    diff = r_shaped - r_std
    colors = ['red' if x > 0 else 'gray' for x in diff]
    
    ax2.bar(episodes, diff, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    
    ax2.set_title('Score Difference (Shaped - Standard)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score Difference')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_xlim(1, TEST_EPISODES)

    plt.tight_layout()
    save_dir = "../../assets/q_learning"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "q_learning_evaluation_v3plots.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nAnalysis plots saved at '{save_path}'")
    plt.show()

if __name__ == "__main__":
    main()