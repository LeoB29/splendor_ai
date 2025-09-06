
from game_state import *
import random
from cards_init import get_full_deck
from cards_init import get_full_nobles
from nn_input_output import flatten_game_state  


def setup_game(num_players=2):
    deck = get_full_deck()

    # Shuffle each tier's deck
    for tier in deck:
        random.shuffle(deck[tier])

    # Deal 4 cards from each tier to the board
    board = {tier: deck[tier][:4] for tier in deck}
    for tier in deck:
        deck[tier] = deck[tier][4:]

    # Initialize tokens (standard Splendor counts for 2 players)
    tokens = {
        'diamond': 4, 'sapphire': 4, 'emerald': 4,
        'ruby': 4, 'obsidian': 4, 'gold': 5
    }

    # Sample nobles (each game includes number of players + 1)
    noble_pool = [
        Noble({'diamond': 3, 'sapphire': 3, 'emerald': 3}),
        Noble({'emerald': 4, 'ruby': 4}),
        Noble({'sapphire': 4, 'obsidian': 4}),
        Noble({'diamond': 4, 'ruby': 4}),
        Noble({'emerald': 3, 'obsidian': 3, 'ruby': 3}),
        Noble({'diamond': 4, 'obsidian': 4}),
    ]
    random.shuffle(noble_pool)
    nobles = noble_pool[:num_players + 1]

    players = [PlayerState() for _ in range(num_players)]
    return GameState(players, current_player=0, tokens=tokens, board=board, nobles=nobles, deck=deck)


test = setup_game()

print(test.board)


test_state = test
flat_state = flatten_game_state(test_state)
print("Flattened game state length:", len(flat_state))


action = GameState.get_legal_actions(test_state)
print(action)
print(len(action))

for index, type in enumerate(action):
    print(action[index].action_type)
    

test_aclength = build_fixed_action_space()
print(len(test_aclength))


for index, type in enumerate(test_aclength):
    print(test_aclength[index].action_type)

print(test_aclength[40].action_type)





random_action = random.choice(action)
print(f"takes action {random_action.action_type}")

test_state = test_state.apply_action(random_action)

