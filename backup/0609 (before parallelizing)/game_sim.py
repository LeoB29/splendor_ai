import random
from game_state import GameState, Card, Noble, PlayerState
from cards_init import *
from monte_carlo import *


#### simulate game with random actions
def simulate_game_random(game_state: GameState, verbose=False):
    state = game_state
    turn = 0

    while not state.is_terminal:
        legal_actions = state.get_legal_actions()

        if not legal_actions:
            if verbose:
                print(f"No legal actions for player {state.current_player}. Skipping turn.")
            state.current_player = (state.current_player + 1) % len(state.players)
            continue

        action = random.choice(legal_actions)
        if verbose:
            print(f"Turn {turn}: Player {state.current_player} takes action {action.action_type}")
        state = state.apply_action(action)
        turn += 1

    if verbose:
        print(f"Game over! Winner: Player {state.winner} with {state.players[state.winner].points} points")

    return state.winner, state.players[state.winner].points


def simulate_game_mt(game_state: GameState, verbose=False):
    state = game_state
    turn = 0

    while not state.is_terminal:
        legal_actions = state.get_legal_actions()

        if not legal_actions:
            if verbose:
                print(f"No legal actions for player {state.current_player}. Skipping turn.")
            state.current_player = (state.current_player + 1) % len(state.players)
            continue

        action = mcts_search(state, time_limit=0.5)  # Replace random.choice with MCTS
        if verbose:
            print(f"Turn {turn}: Player {state.current_player} takes action {action.action_type}")
        state = state.apply_action(action)
        turn += 1

    if verbose:
        print(f"Game over! Winner: Player {state.winner} with {state.players[state.winner].points} points")

    return state.winner, state.players[state.winner].points



# Run simulation
if __name__ == "__main__":
    game = setup_game(num_players=2)
    winner, score = simulate_game_mt(game, verbose=True)
    print(f"\nFinal Result — Winner: Player {winner}, Score: {score}")












