

from gymnasium.wrappers import RecordVideo
import random
from grid_maze import GridMazeEnv




class TabularPolicy:
    def __init__(self):
        self.policy = {}

    def select_action(self, state, actions):
        return self.policy.get(state, random.choice(actions))

    def update(self, state, action):
        self.policy[state] = action

class TabularValueFunction:
    def __init__(self):
        self.values = {}

    def get_value(self, state):
        return self.values.get(state, 0.0)

    def add(self, state, value):
        self.values[state] = value

class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.policy = TabularPolicy()
        self.gamma = 0.9
        for s in env.get_states():
            if not env.is_terminal(s):
                self.policy.update(s, random.choice(env.get_actions(s)))
                
    def print_policy(self):
        grid_size = self.env.grid_size
        policy_grid = [['' for _ in range(grid_size)] for _ in range(grid_size)]
        action_mapping = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        
        for state in self.env.get_states():
            if self.env.is_terminal(state):
                policy_grid[state[1]][state[0]] = 'G' if state in self.env.goal_pos else 'M'
            else:
                action = self.policy.select_action(state, self.env.get_actions(state))
                policy_grid[state[1]][state[0]] = action_mapping[action]
        
        for row in policy_grid[::-1]:
            print(' '.join(row))

    def policy_evaluation(self, values, theta=0.0001):
        
        while True:
            delta = 0
            for state in self.env.get_states():
                if self.env.is_terminal(state):
                    continue
                
                old_value = values.get_value(state)
                action = self.policy.select_action(state, self.env.get_actions(state))
                
                new_value = 0
                for next_state, prob in self.env.get_transitions(state, action).items():
                    reward = self.env.get_reward(state, action, next_state)
                    new_value += prob * (reward + self.gamma * values.get_value(next_state))
                
                values.add(state, new_value)
                delta = max(delta, abs(old_value - new_value))
            
            if delta < theta:
                break
        
        return values

    def policy_improvement(self, values):
        policy_stable = True
        
        for state in self.env.get_states():
            if self.env.is_terminal(state):
                continue
                
            old_action = self.policy.select_action(state, self.env.get_actions(state))
            
            # Find the action that maximizes the value
            best_value = float('-inf')
            best_action = None
            
            for action in self.env.get_actions(state):
                action_value = 0
                for next_state, prob in self.env.get_transitions(state, action).items():
                    reward = self.env.get_reward(state, action, next_state)
                    action_value += prob * (reward + self.gamma * values.get_value(next_state))
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            self.policy.update(state, best_action)
            
            if best_action != old_action:
                policy_stable = False
        
        return policy_stable

    def solve(self, max_iterations=20000, theta=0.0001):
        values = TabularValueFunction()
        for s in self.env.get_states():
            values.add(s, 0.0)
        for i in range(max_iterations):
            # Policy Evaluation
            values = self.policy_evaluation(values,theta)
            
            # Policy Improvement
            policy_stable = self.policy_improvement(values)
            
            if policy_stable:
                print(f"Policy converged after {i+1} iterations")
                print("Optimal Policy:")
                self.print_policy()
                return self.policy
        
        print(f"Maximum iterations ({max_iterations}) reached")
        return self.policy

def run_policy_iteration():
    # Create environment
    env = GridMazeEnv(grid_size=5, goal_pos=[(4,4)], mines=[(1,3), (3,2)], 
                      rnd=True, render_mode="rgb_array")
    

    
    # Test the optimal policy
    video_env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True, fps=2)
    
    obs, _ = video_env.reset()
    done = False
    total_reward = 0
    # Create and run policy iteration
    pi = PolicyIteration(env)
    optimal_policy = pi.solve()
    
    while not done:
        action = optimal_policy.select_action(tuple(obs), env.get_actions(tuple(obs)))
        obs, reward, done, _, _ = video_env.step(action)
        if action ==0:
            actionTaken = "Up"
        elif action ==1:
            actionTaken = "Down"
        elif action ==2:
            actionTaken = "Left"
        else:
            actionTaken = "Right"
        print(f"Action taken: {actionTaken}")
        total_reward += reward
        video_env.render()
    
    print(f"Episode finished with total reward: {total_reward}")
    video_env.close()

if __name__ == "__main__":
    run_policy_iteration()
