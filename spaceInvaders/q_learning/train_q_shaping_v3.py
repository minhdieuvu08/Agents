import gymnasium as gym
import numpy as np
import random
import copy
import time
import re
import os
import getpass  
from tqdm import tqdm
import google.generativeai as genai

try:
    from train import DiscretizedObservationWrapper, QLearningAgent, GRID_SIZE
except ImportError:
    print("Error: Ensure 'train.py' is in the same directory.")
    exit()

POPULATION_SIZE = 20
INITIAL_STEPS = 5000       
POST_SHAPING_STEPS = 10000 
POPULATION_MODEL_FILE = "q_shaping_v3.pkl"
GEMINI_API_KEY = None

# --- DEFAULT HEURISTIC CODE ---
DEFAULT_HEURISTIC_CODE = """
def heuristic_logic(grid_14x14):
    # grid_14x14 là numpy array 14x14 (0 hoặc 1)
    # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT
    
    # Chiến thuật: Tấn công là phòng thủ tốt nhất
    # Ưu tiên bắn (FIRE) nếu có địch ở ngay trên đầu
    
    # Tìm vị trí người chơi (thường ở hàng cuối - index 13)
    player_row = 13
    
    # Kiểm tra xem có địch (1) ở các cột phía trên người chơi không
    # Lưới 14x14, ta quét các hàng từ 0 đến 12
    enemy_above = False
    for r in range(12, -1, -1): # Quét từ dưới lên
        if np.sum(grid_14x14[r, :]) > 0: # Có địch ở hàng này
             enemy_above = True
             break
    
    if enemy_above:
        # Nếu có địch, ưu tiên BẮN hoặc DI CHUYỂN + BẮN
        return 1, 0 # Good: FIRE (1), Bad: NOOP (0)
        
    # Nếu không thấy địch rõ ràng, di chuyển ngẫu nhiên để tìm
    return 1, 0 # Vẫn ưu tiên bắn để clear map
"""

class ShapedQLearningAgent(QLearningAgent):
    def __init__(self, action_space_n, agent_id):
        super().__init__(action_space_n)
        self.id = agent_id
        self.performance = 0.0 

    # --- Q-SHAPING ---
    def q_shaping(self, good_set, bad_set, intensity=1.0): 
        # Tang intensity len 1.0
        for state, action in good_set:
            self.get_q_values(state)
            self.q_table[state][action] += intensity 
            
        for state, action in bad_set:
            self.get_q_values(state)
            self.q_table[state][action] -= intensity 
        return len(good_set)

    def explore(self, env, steps):
        total_reward = 0
        current_steps = 0
        episodes = 0
        original_epsilon = self.epsilon
        self.epsilon = 0.5 
        state, _ = env.reset()
        
        disable_bar = (self.id != 0)
        pbar = tqdm(total=steps, desc=f"Agent {self.id} Exploring", leave=False, disable=disable_bar)
        
        while current_steps < steps:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            current_steps += 1
            pbar.update(1)

            if terminated or truncated:
                state, _ = env.reset()
                episodes += 1
        
        pbar.close()
        self.epsilon = original_epsilon 
        avg_reward = total_reward / max(1, episodes)
        self.performance = avg_reward
        return avg_reward

def generate_heuristic_function_from_gemini():
    """
    Hỏi Gemini để viết hàm Python xử lý logic game.
    """
    global GEMINI_API_KEY
    if not GEMINI_API_KEY:
        print("   [LLM] No Key. Using Default Heuristic.")
        return DEFAULT_HEURISTIC_CODE

    try:
        genai.configure(api_key=GEMINI_API_KEY, transport='rest')
        model = genai.GenerativeModel('gemini-2.5-flash') 

        prompt = """
        You are a Python Expert in Reinforcement Learning.
        Write a Python function named `heuristic_logic(grid_14x14)` to act as a heuristic for Space Invaders.
        
        INPUT: 
        - `grid_14x14`: A numpy array of shape (14, 14) containing 0s (empty) and 1s (objects).
        
        LOGIC REQUIRED:
        1. Identify if there are enemies (1s) in the grid.
        2. Aggressive Strategy: If enemies exist, the GOOD action is always FIRE (1) or RIGHTFIRE (4) or LEFTFIRE (5).
        3. Avoid Standing Still: The BAD action is NOOP (0).
        4. Do NOT use complex imports. Use only `numpy` as `np`.
        
        OUTPUT FORMAT:
        - Return ONLY the python code for the function. 
        - The function must return two integers: `good_action_id`, `bad_action_id`.
        - Do not use markdown blocks. Just the code.
        """
        
        response = model.generate_content(prompt)
        code = response.text
        code = code.replace("```python", "").replace("```", "").strip()
        
        print("   [LLM] Code generated successfully!")
        return code

    except Exception as e:
        print(f"   [LLM] Error generating code: {e}. Using Fallback.")
        return DEFAULT_HEURISTIC_CODE

def apply_generated_heuristic(agents, heuristic_code_str):
    try:
        local_scope = {'np': np} 
        exec(heuristic_code_str, {}, local_scope)
        heuristic_func = local_scope['heuristic_logic']
    except Exception as e:
        print(f"CRITICAL ERROR executing LLM code: {e}")
        return
    
    all_states = list(set().union(*[ag.q_table.keys() for ag in agents]))
    print(f"   [Shaping] Processing {len(all_states)} unique states...")

    good_set = []
    bad_set = []

    for state_tuple in all_states:
        grid = np.array(state_tuple).reshape(GRID_SIZE)
        
        try:
            # Goi ham do Gemini viet
            g_act, b_act = heuristic_func(grid)
            good_set.append((state_tuple, g_act))
            bad_set.append((state_tuple, b_act))
        except:
            continue
            
    print(f"   [Shaping] Generated {len(good_set)} shaping rules. Injecting into agents...")
    for agent in agents:
        agent.q_shaping(good_set, bad_set, intensity=1.0)

def run_algorithm_v3():
    global GEMINI_API_KEY
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=None)
    env = DiscretizedObservationWrapper(env, grid_size=GRID_SIZE)
    action_size = env.action_space.n

    print("Initializing 20 Agents...")
    population = [ShapedQLearningAgent(action_size, i) for i in range(POPULATION_SIZE)]

    print(f"Exploration ({INITIAL_STEPS} steps)...")
    for agent in population:
        agent.explore(env, steps=INITIAL_STEPS)

    print("Generating Heuristic Logic via Gemini...")
    heuristic_code = generate_heuristic_function_from_gemini()

    print("Mass-Applying Heuristics...")
    
    try:
        local_scope = {'np': np}
        exec(heuristic_code, {}, local_scope)
        heuristic_func = local_scope['heuristic_logic']
        
        all_states = list(set().union(*[ag.q_table.keys() for ag in population]))
        print(f"Scanning {len(all_states)} states...")
        
        good_set = []
        bad_set = []
        for state_tuple in all_states:
            grid = np.array(state_tuple).reshape(GRID_SIZE)
            try:
                g_act, b_act = heuristic_func(grid)
                good_set.append((state_tuple, g_act))
                bad_set.append((state_tuple, b_act))
            except: continue
        
        for agent in population:
            #
            agent.q_shaping(good_set, bad_set, intensity=1.5) # tang intensity
            
    except Exception as e:
        print(f"Error applying heuristics: {e}")

    print(f"Post-Shaping Exploration ({POST_SHAPING_STEPS} steps)...")
    for agent in population:
        agent.explore(env, steps=POST_SHAPING_STEPS)
    
    population.sort(key=lambda x: x.performance, reverse=True)
    
    print("\n--- Leaderboard ---")
    for ag in population[:5]: 
        print(f"Agent {ag.id}: {ag.performance:.2f}")

    survivors = population[:POPULATION_SIZE // 2]
    best_agent = survivors[0]
    best_agent.save(POPULATION_MODEL_FILE)
    print(f"\nSaved Best Agent to {POPULATION_MODEL_FILE}")
    env.close()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("POPULATION TRAINING V3 (CODE GENERATION)")
    print("="*50)
    
    try:
        input_key = getpass.getpass(prompt="Gemini API Key: ")
        if input_key.strip():
            GEMINI_API_KEY = input_key.strip()
        else:
            GEMINI_API_KEY = None
    except:
        GEMINI_API_KEY = input("Gemini API Key: ").strip()

    run_algorithm_v3()