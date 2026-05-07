# LLM-Augmented Reinforcement Learning for Atari Space Invaders 👾

<p align="center">
  <img src="assets/q_learning/videos/space_invaders_ufo_kill.gif" width="500" alt="Space Invaders Agent Demo">
  <br>
  <i>Agent trained with LLM-guided Q-Shaping successfully sniping a high-value UFO.</i>
</p>

This repository explores advanced Reinforcement Learning (RL) techniques to solve the **Atari Space Invaders** environment. The project focuses on two main research directions: **Reward Shaping** with PPO and **LLM-Guided Q-Shaping** using Google Gemini.

## 🚀 Key Research Components

### 1. PPO with Reward Shaping (Stable-Baselines3)
We utilize Proximal Policy Optimization (PPO) combined with a custom environment **Wrapper** to address the sparse reward problem and improve agent survival.
- **Penalty Logic:** Implementing significant penalties (up to `-5.0`) when the agent loses a life.
- **Incentive Logic:** Small auxiliary rewards for firing and strategic lateral movements.
- **Comparative Analysis:** A dedicated experimental script trains two models (Baseline vs. RS) and generates a comparative learning curve to visualize the impact of shaping on convergence.

### 2. LLM-Guided Q-Shaping (Q-Learning + Gemini)
A novel approach that integrates Large Language Models (LLMs) into the training loop of a population of 20 agents.
- **Dynamic Heuristic Generation:** Using **Google Gemini 2.5 Flash** to analyze the game's spatial grid and generate Python heuristic logic on-the-fly.
- **Q-Injection:** The LLM-generated logic identifies "Good" vs "Bad" actions for discovered states, injecting these biases directly into the Q-table to accelerate learning.
- **Mass-Shaping:** Automatically applying shaping rules across the entire agent population to optimize exploration.

### 3. Tetris Exploration (Work in Progress)
A baseline implementation for Tetris using PPO with frame stacking techniques.
- **State Representation:** Utilizing `VecFrameStack` (n=4) to provide the agent with temporal information, essential for understanding falling speeds and placement.
- **Current Status:** Basic training is implemented to establish a performance baseline. Further optimization via custom reward shaping (similar to the Space Invaders project) is planned to improve line-clearing efficiency.

---

## 📁 Project Structure
```text
.
├── assets/                       # Visualizations and media
│   ├── ppo/                      # Performance plots for PPO experiments
│   │   ├── ppo_rs_comparison.png
│   │   └── evaluation_raw_reward_comparison.png
│   └── q_learning/               # Visualizations for Q-Learning
│       ├── videos/               # Gameplay recordings (GIFs)
│       │   └── space_invaders_ufo_kill.gif
│       ├── q_agent_comparision_plot.png
│       └── q_learning_evaluation_v3plots.png
├── spaceInvaders/                # Space Invaders RL implementations
│   ├── ppo/                      # Proximal Policy Optimization experiments
│   │   ├── ppo_logs/             # TensorBoard event logs (Ignored by Git)
│   │   ├── train_ppo_rs.py       # PPO training with Reward Shaping
│   │   ├── test_ppo_compare.py   # Evaluation and comparison script
│   │   └── ...                   
│   └── q_learning/               # Q-Learning with Reward Shaping
│       ├── q_learning_logs/      # Training statistics (Ignored by Git)
│       ├── train_q_shaping_v3.py              # training logic
│       └── ...                   # Versioned shaping experiments (v1, v2, v3)
├── tetris/                       # Tetris RL experiment modules
├── requirements.txt              
├── .gitignore                    
└── README.md                     
```

---

## 🛠 Tech Stack
- **AI/RL:** Stable-Baselines3, PyTorch, Q-Learning.
- **LLM:** Google Gemini API (`google-generativeai`).
- **Environments:** Gymnasium Atari (ALE).
- **Analysis:** Matplotlib, NumPy, TQDM, Tensorboard.

---

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/minhdieuvu08/LLM-Augmented-RL.git
   cd LLM-Augmented-RL
   ```

2. **Setup Environment:**
   ```bash
   python3 -m venv agents_env
   source agents_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Gemini API Key:** 
   To use the LLM-Guided scripts, ensure you have a valid API Key from [Google AI Studio](https://aistudio.google.com/).

---

## 📈 Methodology

### 1. PPO Reward Shaping Analysis
By running `test_ppo_compare.py`, the system evaluates the PPO models (Baseline vs. RS) over a 100-episode window. The **Reward Shaping (RS)** model typically exhibits more stable survival rates in the early training stages, as visualized in the comparison plots within `assets/ppo/`.

### 2. Q-Learning & LLM-Guided Comparison
The script `test_comparision_v3.py` is used to conduct a rigorous statistical comparison between the standard Q-Learning agent and the LLM-Shaped agent. It generates three types of visualizations:
- **Raw Reward per Episode:** Direct score comparison across 100 test episodes.
- **Cumulative Reward:** Visualizes the "learning lead" or total performance advantage over time.
- **Score Difference:** A bar chart highlighting specific episodes where Reward Shaping outperformed the baseline.

### 3. LLM Q-Shaping Workflow
1. **Explore:** Agents perform an initial exploration to collect state-action data and discover relevant game states.
2. **Consult:** The system sends a simplified grid representation of the environment to the Gemini API (Google Gemini 1.5 Flash).
3. **Generate:** Gemini returns a Python function `heuristic_logic(grid_14x14)` containing spatial reasoning rules.
4. **Inject:** The Q-values are adjusted based on the LLM's heuristic logic, significantly reducing the "random walk" exploration phase.