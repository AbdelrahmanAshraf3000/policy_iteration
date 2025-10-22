import json
import numpy as np
from grid_maze import GridMazeEnv
from gymnasium.wrappers import RecordVideo

SAVE_FILE = "all_optimal_policies.json"
ACTION_MAP = {
    "Up": 0,
    "Down": 1,
    "Left": 2,
    "Right": 3,
    
    None: 0
}
def make_key(agent, goal, mines):
    """Ensure consistent key formatting"""
    agent = tuple(int(x) for x in agent)
    goal = tuple(goal[0]) if isinstance(goal, list) else tuple(goal)
    mines = [tuple(m) for m in mines]
    return f"A{agent}_G{goal}_M{mines[0]}_{mines[1]}"

def load_policy(agent, goal, mines):
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)
    key = make_key(agent, goal, mines)
    print(f"Looking for key: {key}")
    return data.get(key, None)

def test_environment():
    env = GridMazeEnv(
        grid_size=5,
        render_mode="rgb_array",
        rnd=True
    )
    env.mines = [tuple(mine) for mine in env.mines if tuple(mine) != tuple(env.agent_pos)]
    print(f"Testing environment with Agent at {env.agent_pos}, Goal at {env.goal_pos}, Mines at {env.mines}")
    
    video_env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True, fps=2)
    obs, _ = video_env.reset()
    done = False
    total_reward = 0
    
    
    entry = load_policy(env.agent_pos, env.goal_pos, env.mines)
    if not entry:
        print("❌ Policy not found for this configuration.")
        return

    optimal_policy = entry["policy"]



    while not done:
        row, col = obs  
        action_str = optimal_policy[row][col]
        action = ACTION_MAP.get(action_str, 0)
        obs, reward, done, _, _ = video_env.step(action)
        actionTaken = ["Up", "Down", "Left", "Right"][action]
        print(f"Action taken: {actionTaken}")
        total_reward += reward
        video_env.render()

    print(f"Episode finished with total reward: {total_reward}")
    video_env.close()

if __name__ == "__main__":
    test_environment()
