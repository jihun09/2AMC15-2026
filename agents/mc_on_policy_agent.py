"""MC on-policy agent.

This is an agent that takes actions from the available action space, based on the policy derived from simulations.
"""
from random import randint

from agents import BaseAgent
from pathlib import Path
import numpy as np
import random 


class McOnPolicyAgent(BaseAgent):
    """Agent that performs an action based on  the policy derived from simulations. """
    def __init__(self, epsilon, epsilon_decay, epsilon_min):
        self.state_action_space = {}  # keeps all pairs of states and actions
        self.state_action_indexer = {}
        self.policy = {}
        self.returns = {}
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def decay_epsilon(self):
        """Multiplicative epsilon decay applied once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    # For stopping criterion
    def get_greedy_action(self):
        """Returns the greedy policy as a list of best actions per state."""
        return [int(np.argmax(self.state_action_space[s])) for s in range(len(self.state_action_space))]

    def update(self, state: tuple[int, int], reward: float, action):
        """Updates values in the model, if the state-action pair was visited for the first time.
        It performs following operations:
        1. Appends g to correct state-action pair.
        2. Pairs Q(s,a) is updated with the average of the values in returns for this pair.
        3. Optimal action A* is computed.
        4. According to epsilon, probabiities in the policy are updated for that pair."""
        # 1. Appending value to returns
        idx_state = self.state_action_indexer[state] # transforms tuple (x,y) to int z 
        self.returns[(idx_state, action)].append(reward)

        # 2. Updating state_action_space
        self.state_action_space[idx_state][action] = np.average(self.returns[(idx_state, action)])

        # 3. Checking optimal action A* in S.
        optimal_action_in_s = np.argmax(self.state_action_space[idx_state]) # chooses the first max indx

        # 4. Updating probabilities in the policy
        for a in range(len(self.state_action_space[idx_state])):
            if a == optimal_action_in_s:
                self.policy[idx_state][a] = (1 - self.epsilon) + (self.epsilon / sum(self.state_action_space[idx_state] != -9999))
            elif self.state_action_space[idx_state][a] != -9999:
                self.policy[idx_state][a] = self.epsilon / sum(self.state_action_space[idx_state] != -9999)


    def take_action(self, state: tuple[int, int]) -> int:
        #  taking an action according to policy
        indx = self.state_action_indexer[state]
        probs = self.policy[indx]
        return random.choices(range(4), weights=probs, k=1)[0]
    
    def feasible_actions(self, grid, position_x, position_y):
        """Helper function for creating list of fesible actions for every feasible agent's state"""
        actions = np.full(4, -9999, dtype=float)
        # checking move down
        if grid[(position_x, position_y + 1)] in {0, 3, 4}:
            actions[0] = 0

        # checking move up
        if grid[(position_x, position_y - 1)] in {0, 3, 4}:
            actions[1] = 0

        # checking move left
        if grid[(position_x - 1, position_y)] in {0, 3, 4}:
            actions[2] = 0

        # checking move right
        if grid[(position_x + 1, position_y)] in {0, 3, 4}:
            actions[3] = 0

        return actions
    
    
    def intitialize_probabilities(self, actions):
        """Function assigns uniform probabilities fro feasible actions."""
        possible_moves = np.sum(actions == 0)

        action_prob = np.zeros_like(actions, dtype=float)
        action_prob[actions == 0] = round(1 / possible_moves, 2)

        return action_prob
    
    def create_state_action_space(self, grid):
        """Function used for creating state action space and teh indexer used for accessing their values"""
        rows, cols = grid.shape
        boundry_rows = (1, rows-1) # we skip iteration over boundry cells 
        boundery_columns = (1, cols-1)

        # storres all empty and target cells
        empty_cells = []

        # a 2d array storing actions per location
        state_action_space = []

        # a 2d  storing probabilities for soft_policy
        soft_policy = []

        # iteration over the grid
        for idx_row in range(boundry_rows[0], boundry_rows[1]):
            for idx_col in range(boundery_columns[0], boundery_columns[1]):
                if grid[idx_row, idx_col] in {0, 3, 4}:
                    # all feasible actions at given state, feasible actions get value of 0 and infeasible get -9999
                    actions = self.feasible_actions(grid, idx_row, idx_col)

                    # weights for initial soft-policy, all feasible actions get uniform probability, where infeasible get prob of 0
                    probs = self.intitialize_probabilities(actions)
                    state_action_space.append(actions)
                    soft_policy.append(probs)

                    # used for creating the indexer
                    empty_cells.append((idx_row, idx_col))
         
        # assign the state_action_space
        self.state_action_space = state_action_space
        # assigns soft_policy to policy
        self.policy = soft_policy
        # create an indexer of state spaces
        state_action_indexer = {location : indx for indx, location in enumerate(empty_cells)}
        self.state_action_indexer = state_action_indexer


        ####### creating returns ######
        returns = {}

        for row in range(len(state_action_space)):
            for action_idx, action_value in enumerate(state_action_space[row]):
                if action_value != -9999:
                    returns[(row, action_idx)] = []

        self.returns = returns

    # this method is done outside the create_state_action_space as look_up_table has to be unique for every episode
    def look_up_first_visited(self):
        """Function using the state_action_space creates a lookup table, 
        It assumes that location is passed using the location indexer.
        The inital value is set as -1. Values of these kesys, correspond to 
        steps where state action pairs were first explored, form teh beggining of the episode.

        NOTE: This function should be only initalised, once create_state_action_space was called!
        
        """
        look_up_table = {}

        for row in range(len(self.state_action_space)):
            for action_idx, action_value in enumerate(self.state_action_space[row]):
                if action_value != -9999:
                    look_up_table[(row, action_idx)] = -1          
        return look_up_table


# getting th grid
# self.grid = Grid.load_grid(self.grid_fp).cells

# grid_path = Path("grid_configs") / "small_grid.npy"
# grid = np.load(grid_path)
# print(grid)

# # print("###")


# new = McOnPolicyAgent(0.8)

# new.create_state_action_space(grid)

# print(new.state_action_space)

# lookup = new.look_up_first_visited()
# print("lookup")
# print(lookup)


# action = new.take_action((6,6))
# print(action)



