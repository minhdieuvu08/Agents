import gymnasium as gym
import time
import os
import numpy as np

# --- 1. IMPORT CÁC CLASS CẦN THIẾT ---
# Bắt buộc phải có file train nằm cùng thư mục để Python hiểu cấu trúc Agent và Wrapper
try:
    from train import DiscretizedObservationWrapper, QLearningAgent, GRID_SIZE
except ImportError:
    print("Lỗi: Không tìm thấy file 'train_qlearning_spaceinvaders.py'.")
    print("Vui lòng đảm bảo file này nằm cùng thư mục.")
    exit()

# --- 2. CẤU HÌNH ---
# Tên file model bạn muốn xem (Ưu tiên V3 là bản mạnh nhất)
MODEL_FILE = "q_shaping_v3.pkl" 

# Tốc độ game (Giây). 
# 0.01 = Rất nhanh, 0.05 = Bình thường, 0.1 = Chậm (Matrix style)
GAME_SPEED = 0.02

# Số ván muốn xem
NUM_EPISODES_TO_WATCH = 5

def watch_agent():
    # Kiểm tra file model
    if not os.path.exists(MODEL_FILE):
        print(f"Lỗi: Không tìm thấy file model '{MODEL_FILE}'.")
        print("Bạn đã chạy file train (population_training_v3_codegen.py) chưa?")
        return

    print(f"\n>>> Đang khởi động Game Space Invaders...")
    print(f">>> Đang tải bộ não Agent từ: {MODEL_FILE}")
    print(">>> Nhấn Ctrl+C trên terminal để dừng xem bất cứ lúc nào.")

    # Khởi tạo môi trường với chế độ hiển thị hình ảnh ('human')
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    
    # Áp dụng Wrapper để Agent "nhìn" thấy lưới 14x14 (giống lúc train)
    # Lưu ý: Cửa sổ game vẫn hiện hình ảnh gốc đẹp đẽ cho mắt người xem
    env = DiscretizedObservationWrapper(env, grid_size=GRID_SIZE)
    
    # Khởi tạo Agent và load dữ liệu học
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
            # 1. Agent chọn hành động dựa trên những gì nó nhìn thấy
            action = agent.get_action(state)
            
            # 2. Thực hiện hành động trong game
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 3. Cập nhật điểm số
            total_reward += reward
            state = next_state
            step += 1
            
            # 4. Làm chậm game để mắt người xem kịp (Render đã được gym xử lý tự động)
            time.sleep(GAME_SPEED)
            
        print(f"Kết thúc Ván {episode + 1}. Tổng điểm: {total_reward}. Số bước đi: {step}")
        
        # Nghỉ 1 giây giữa các ván
        time.sleep(1.0)

    env.close()
    print("\n>>> Đã xem xong!")

if __name__ == "__main__":
    try:
        watch_agent()
    except KeyboardInterrupt:
        print("\n>>> Đã dừng xem theo yêu cầu người dùng.")