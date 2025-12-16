import gymnasium as gym
import numpy as np
import random
import pickle
import os
from gymnasium import Wrapper

LOG_DIR = "./q_learning_logs/"
MODEL_FILE = "q_table_spaceinvaders.pkl"
EPISODES = 5000         
LEARNING_RATE = 0.1     
DISCOUNT_FACTOR = 0.99  
EPSILON_START = 1.0    
EPSILON_END = 0.1      
EPSILON_DECAY = 0.9995  


GRID_SIZE = (14, 14) 

class DiscretizedObservationWrapper(Wrapper):
    """
    This wrapper helps 'compress' the game screen into a simplified state
    that the Q-Table can store.
    """
    def __init__(self, env, grid_size=GRID_SIZE):
        super().__init__(env)
        self.grid_size = grid_size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.process_observation(obs), reward, terminated, truncated, info

    def process_observation(self, obs):
        # Crop the playing area 
        cropped = obs[30:195, :, :]

        # Convert to Grayscale
        gray = np.mean(cropped, axis=2)

        # Downsample 
        h, w = gray.shape
        h_step = h // self.grid_size[0]
        w_step = w // self.grid_size[1]
        
        small_obs = gray[::h_step, ::w_step]
        small_obs = small_obs[:self.grid_size[0], :self.grid_size[1]]
        binary_obs = (small_obs > 50).astype(int)

        # Convert to tuple
        return tuple(binary_obs.flatten())

class QLearningAgent:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n
        self.q_table = {} 
        self.epsilon = EPSILON_START

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_n - 1) # Explore 
        return np.argmax(self.get_q_values(state))

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_n)
        return self.q_table[state]

    def update(self, state, action, reward, next_state):
        current_q = self.get_q_values(state)[action]
        max_next_q = np.max(self.get_q_values(next_state))
        
        # Q-Learning Formula:
        # Q_new = Q_old + Alpha * (Reward + Gamma * Max_Q_next - Q_old)
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
        
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Saved Q-table with {len(self.q_table)} states.")

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(self.q_table)} states.")
        else:
            print("Save file not found, initializing new Q-table.")

def train_q_learning():
    os.makedirs(LOG_DIR, exist_ok=True)
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = DiscretizedObservationWrapper(env, grid_size=GRID_SIZE)
    
    agent = QLearningAgent(env.action_space.n)
    
    if os.path.exists(MODEL_FILE):
        agent.load(MODEL_FILE)

    print(f"Starting training with grid {GRID_SIZE}...")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Update knowledge for the Agent
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward} | Epsilon: {agent.epsilon:.2f} | States explored: {len(agent.q_table)}")
        if (episode + 1) % 1000 == 0:
            agent.save(MODEL_FILE)

    env.close()

if __name__ == "__main__":
    train_q_learning()