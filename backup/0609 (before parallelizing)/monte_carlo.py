import math
import time
import random
from game_state import *

class MCTSNode:
    def __init__(self, state: GameState, parent=None, action_taken=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.action_taken = action_taken
        self.untried_actions = state.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.41):
        best_score = float("-inf")
        best_child = None
        for child in self.children:
            exploit = child.total_reward / (child.visits + 1e-4)
            explore = math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4))
            score = exploit + exploration_weight * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        action = self.untried_actions.pop()
        new_state = self.state.apply_action(action)
        child_node = MCTSNode(new_state, parent=self, action_taken=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, reward, player_index):
        self.visits += 1
        if self.parent:
            # If not the root, invert reward for alternating players
            if self.state.get_current_player() != player_index:
                reward = -reward
            self.parent.backpropagate(reward, player_index)
        self.total_reward += reward


def mcts_search(root_state: GameState, time_limit=1.0):
    root = MCTSNode(root_state)
    player_index = root_state.get_current_player()
    end_time = time.time() + time_limit

    while time.time() < end_time:
        node = root
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # 2. Expansion
        if not node.state.is_terminal and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation
        state = node.state.clone()
        while not state.is_terminal:
            actions = state.get_legal_actions()
            if not actions:
                break
            action = random.choice(actions)
            state = state.apply_action(action)

        # 4. Backpropagation
        reward = state.get_reward(player_index)
        node.backpropagate(reward, player_index)

    # Return the action that leads to the best child
    best = max(root.children, key=lambda c: c.visits)
    return best.action_taken


def mcts_fn(state):
    return mcts_search(state, time_limit=0.1)  # Fast simulations

def simulate_game_with_mcts(game_state: GameState, mcts_fn, verbose=False):
    state = game_state
    turn = 0
    player_index = state.current_player

    trajectory = []  # (state, action) tuples

    while not state.is_terminal:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            state.current_player = (state.current_player + 1) % len(state.players)
            continue

        action = mcts_fn(state)  # Replace random with MCTS!
        trajectory.append((state.clone(), action))

        if verbose:
            print(f"Turn {turn}: Player {state.current_player} takes action {action.action_type}")
        state = state.apply_action(action)
        turn += 1

    reward = state.get_reward(player_index)
    return trajectory, reward