import gymnasium as gym
import os
import time
from train import DiscretizedObservationWrapper, QLearningAgent, GRID_SIZE

MODEL_TO_RECORD = "q_shaping_v3.pkl" 
NUM_RECORD_EPISODES = 15

def main():
    if not os.path.exists(MODEL_TO_RECORD):
        print(f"Error: Model file '{MODEL_TO_RECORD}' not found.")
        return

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    
    env = DiscretizedObservationWrapper(env, grid_size=GRID_SIZE)
    
    action_size = env.action_space.n
    agent = QLearningAgent(action_size)
    agent.load(MODEL_TO_RECORD)
    
    agent.epsilon = 0.0 

    print(f">>> Starting demo for: {MODEL_TO_RECORD}")
    print(">>> Preparation: Open screen recording")

    for i in range(NUM_RECORD_EPISODES):
        state, _ = env.reset(seed=i+42) 
        terminated = False
        truncated = False
        total_reward = 0
        
        print(f"--- Episode {i+1} ---")
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            time.sleep(0.01) 
            
        print(f"Episode {i+1} Finished. Total Score: {total_reward}")
        time.sleep(1) 

    env.close()
    print(">>> Demo finished. You can now convert your recording to a GIF!")

if __name__ == "__main__":
    main()