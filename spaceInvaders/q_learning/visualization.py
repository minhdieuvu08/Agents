import gymnasium as gym
import time
import os
import numpy as np

try:
    from train import DiscretizedObservationWrapper, QLearningAgent, GRID_SIZE
except ImportError:
    print("Lỗi: Không tìm thấy file 'train_qlearning_spaceinvaders.py'.")
    print("Vui lòng đảm bảo file này nằm cùng thư mục.")
    exit()

MODEL_FILE = "q_shaping_v3.pkl" 

GAME_SPEED = 0.02

NUM_EPISODES_TO_WATCH = 5

def watch_agent():
    if not os.path.exists(MODEL_FILE):
        print(f"Lỗi: Không tìm thấy file model '{MODEL_FILE}'.")
        return

    print(f"\n>>> Đang khởi động Game Space Invaders...")
    print(f">>> Đang tải bộ não Agent từ: {MODEL_FILE}")

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    env = DiscretizedObservationWrapper(env, grid_size=GRID_SIZE)
    
    action_size = env.action_space.n
    agent = QLearningAgent(action_size)
    agent.load(MODEL_FILE)
    
    # Quan trọng: Tắt chế độ ngẫu nhiên (Epsilon = 0)
    # Để Agent chơi bằng 100% thực lực, không đi bừa nữa
    agent.epsilon = 0.0 

    for episode in range(NUM_EPISODES_TO_WATCH):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        print(f"\n--- Đang xem Ván {episode + 1}/{NUM_EPISODES_TO_WATCH} ---")
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            step += 1
            
            time.sleep(GAME_SPEED)
            
        print(f"Kết thúc Ván {episode + 1}. Tổng điểm: {total_reward}. Số bước đi: {step}")
        
        time.sleep(1.0)

    env.close()
    print("\n>>> Đã xem xong!")

if __name__ == "__main__":
    try:
        watch_agent()
    except KeyboardInterrupt:
        print("\n>>> Đã dừng xem theo yêu cầu người dùng.")