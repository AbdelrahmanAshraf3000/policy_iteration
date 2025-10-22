import itertools
import json
import numpy as np
from tqdm import tqdm
from policy_iteration import PolicyIteration 
from grid_maze import GridMazeEnv

GRID_SIZE = 5
SAVE_FILE = "all_optimal_policies.json"

# Actions mapping for readability
ACTIONS = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

def env_key(agent, goal, mines):
    """Make a unique key for each environment configuration"""
    return f"A{agent}_G{goal}_M{mines[0]}_{mines[1]}"

def compute_policy(env):
    """Run policy iteration and return a 5x5 grid of best actions."""
    pi = PolicyIteration(env)
    policy = pi.solve()
    grid_policy = np.full((GRID_SIZE, GRID_SIZE), None)
    for state, action in policy.policy.items():
        grid_policy[state] = ACTIONS[action]
    return grid_policy.tolist()

def generate_all_environments(limit=None):
    positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
    all_configs = list(itertools.permutations(positions, 4))
    if limit:
        all_configs = all_configs[:limit]

    data = {}
    for (agent, goal, mine1, mine2) in tqdm(all_configs, desc="Processing environments"):
        if len({agent, goal, mine1, mine2}) < 4:
            continue  # skip overlapping positions

        env = GridMazeEnv(
            grid_size=GRID_SIZE,
            goal_pos=[goal],
            mines=[mine1, mine2],
            render_mode=None,
            rnd=False
        )
        env.agent_pos = np.array(agent)

        policy = compute_policy(env)
        key = env_key(agent, goal, [mine1, mine2])
        data[key] = {
            "agent": agent,
            "goal": goal,
            "mines": [mine1, mine2],
            "policy": policy
        }

    with open(SAVE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(data)} environments to {SAVE_FILE}")

if __name__ == "__main__":
    generate_all_environments()
