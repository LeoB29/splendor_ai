from game_state import GameState, Card, Noble, PlayerState
import random

# Full card deck for Splendor (official base game, all tiers)
def get_full_deck():
    tier1 = [
        Card(1, {'emerald': 1, 'ruby': 1, 'obsidian': 1}, 'sapphire', 0),
        Card(1, {'diamond': 2, 'sapphire': 1}, 'emerald', 0),
        Card(1, {'emerald': 1, 'diamond': 1}, 'ruby', 0),
        Card(1, {'sapphire': 2}, 'obsidian', 0),
        Card(1, {'ruby': 1}, 'diamond', 0),
        Card(1, {'diamond': 1}, 'sapphire', 0),
        Card(1, {'emerald': 2}, 'emerald', 0),
        Card(1, {'obsidian': 2}, 'obsidian', 0),
        Card(1, {'emerald': 1, 'sapphire': 1}, 'diamond', 0),
        Card(1, {'diamond': 1, 'sapphire': 1, 'ruby': 1}, 'emerald', 0),
        Card(1, {'emerald': 1, 'ruby': 2}, 'ruby', 0),
        Card(1, {'sapphire': 1, 'obsidian': 1}, 'obsidian', 0),
        Card(1, {'diamond': 1, 'obsidian': 2}, 'diamond', 0),
        Card(1, {'ruby': 2, 'obsidian': 1}, 'sapphire', 0),
        Card(1, {'diamond': 1, 'emerald': 2}, 'emerald', 0),
        Card(1, {'sapphire': 1, 'ruby': 1}, 'ruby', 0),
        Card(1, {'emerald': 1, 'sapphire': 2}, 'sapphire', 0),
        Card(1, {'diamond': 2, 'ruby': 1}, 'obsidian', 0),
        Card(1, {'emerald': 1, 'obsidian': 1}, 'diamond', 0),
        Card(1, {'sapphire': 2, 'ruby': 1}, 'emerald', 0),
        Card(1, {'diamond': 1, 'sapphire': 2}, 'sapphire', 0),
        Card(1, {'emerald': 2, 'obsidian': 1}, 'obsidian', 0),
        Card(1, {'diamond': 1, 'emerald': 2}, 'emerald', 0),
        Card(1, {'sapphire': 1, 'emerald': 1}, 'sapphire', 0),
        Card(1, {'ruby': 1, 'obsidian': 2}, 'ruby', 0),
        Card(1, {'diamond': 2, 'sapphire': 1}, 'diamond', 0),
        Card(1, {'sapphire': 1, 'emerald': 1, 'ruby': 1}, 'obsidian', 0),
        Card(1, {'diamond': 1, 'sapphire': 1, 'obsidian': 1}, 'ruby', 0),
        Card(1, {'emerald': 2, 'ruby': 1}, 'emerald', 0),
        Card(1, {'obsidian': 2, 'sapphire': 1}, 'diamond', 0),
        Card(1, {'diamond': 1, 'obsidian': 1}, 'emerald', 0),
        Card(1, {'emerald': 2, 'sapphire': 1}, 'ruby', 0),
        Card(1, {'sapphire': 1, 'ruby': 2}, 'obsidian', 0),
        Card(1, {'diamond': 2, 'emerald': 1}, 'diamond', 0),
        Card(1, {'emerald': 1, 'sapphire': 1, 'obsidian': 1}, 'sapphire', 0),
        Card(1, {'ruby': 1, 'sapphire': 1}, 'emerald', 0),
        Card(1, {'diamond': 1, 'emerald': 1, 'obsidian': 1}, 'ruby', 0),
        Card(1, {'diamond': 2, 'obsidian': 1}, 'obsidian', 0),
        Card(1, {'sapphire': 2, 'emerald': 1}, 'diamond', 0),

    ]

    tier2 = [
        Card(2, {'emerald': 2, 'sapphire': 2, 'ruby': 3}, 'diamond', 1),
        Card(2, {'diamond': 2, 'obsidian': 2, 'sapphire': 2}, 'emerald', 2),
        Card(2, {'ruby': 2, 'emerald': 2, 'obsidian': 3}, 'sapphire', 1),
        Card(2, {'diamond': 3, 'emerald': 2, 'sapphire': 2}, 'ruby', 2),
        Card(2, {'sapphire': 3, 'emerald': 2, 'ruby': 2}, 'obsidian', 2),
        Card(2, {'diamond': 2, 'emerald': 3, 'obsidian': 2}, 'sapphire', 1),
        Card(2, {'emerald': 3, 'sapphire': 2, 'ruby': 2}, 'diamond', 2),
        Card(2, {'diamond': 2, 'ruby': 3, 'obsidian': 2}, 'emerald', 2),
        Card(2, {'emerald': 2, 'sapphire': 3, 'obsidian': 2}, 'ruby', 2),
        Card(2, {'diamond': 3, 'sapphire': 3, 'obsidian': 3}, 'emerald', 1),
        Card(2, {'sapphire': 2, 'ruby': 2, 'obsidian': 3}, 'diamond', 2),
        Card(2, {'diamond': 3, 'emerald': 3, 'ruby': 3}, 'sapphire', 2),
        Card(2, {'emerald': 2, 'ruby': 2, 'obsidian': 2}, 'ruby', 1),
        Card(2, {'diamond': 2, 'sapphire': 2, 'emerald': 2}, 'obsidian', 1),
        Card(2, {'sapphire': 3, 'emerald': 3, 'obsidian': 3}, 'diamond', 2),
        Card(2, {'diamond': 3, 'ruby': 3, 'obsidian': 3}, 'emerald', 2),
        Card(2, {'sapphire': 3, 'ruby': 3, 'obsidian': 3}, 'ruby', 2),
        Card(2, {'diamond': 3, 'sapphire': 3, 'emerald': 3}, 'obsidian', 2),
        Card(2, {'emerald': 2, 'sapphire': 3, 'ruby': 3}, 'diamond', 2),
        Card(2, {'diamond': 2, 'ruby': 3, 'emerald': 3}, 'sapphire', 2),
        Card(2, {'emerald': 3, 'obsidian': 3, 'sapphire': 3}, 'ruby', 2),
        Card(2, {'emerald': 2, 'ruby': 3, 'obsidian': 3}, 'emerald', 2),
        Card(2, {'diamond': 3, 'sapphire': 2, 'ruby': 2}, 'obsidian', 2),
        Card(2, {'diamond': 2, 'emerald': 2, 'obsidian': 2}, 'diamond', 1),
        Card(2, {'sapphire': 3, 'emerald': 2, 'obsidian': 2}, 'sapphire', 1),
        Card(2, {'diamond': 2, 'ruby': 2, 'obsidian': 3}, 'ruby', 2),
        Card(2, {'sapphire': 2, 'emerald': 3, 'ruby': 2}, 'emerald', 1),
        Card(2, {'diamond': 3, 'sapphire': 2, 'emerald': 2}, 'sapphire', 1),
        Card(2, {'ruby': 2, 'obsidian': 2, 'sapphire': 2}, 'ruby', 2),
        Card(2, {'diamond': 2, 'emerald': 3, 'ruby': 2}, 'obsidian', 1),
    ]

    tier3 = [
        Card(3, {'diamond': 3, 'sapphire': 3, 'emerald': 5, 'obsidian': 3}, 'ruby', 4),
        Card(3, {'ruby': 3, 'sapphire': 3, 'emerald': 3, 'obsidian': 5}, 'diamond', 5),
        Card(3, {'diamond': 3, 'sapphire': 5, 'emerald': 3, 'ruby': 3}, 'obsidian', 4),
        Card(3, {'diamond': 5, 'sapphire': 3, 'emerald': 3, 'obsidian': 3}, 'sapphire', 4),
        Card(3, {'diamond': 3, 'sapphire': 3, 'emerald': 3, 'ruby': 5}, 'emerald', 4),
        Card(3, {'diamond': 3, 'sapphire': 5, 'emerald': 3, 'obsidian': 3}, 'ruby', 4),
        Card(3, {'diamond': 3, 'emerald': 3, 'ruby': 3, 'obsidian': 5}, 'sapphire', 4),
        Card(3, {'sapphire': 3, 'emerald': 3, 'ruby': 5, 'obsidian': 3}, 'diamond', 4),
        Card(3, {'diamond': 4, 'sapphire': 4, 'emerald': 4, 'ruby': 0, 'obsidian': 0}, 'emerald', 3),
        Card(3, {'diamond': 0, 'sapphire': 0, 'emerald': 4, 'ruby': 4, 'obsidian': 4}, 'sapphire', 3),
    ]
    return {1: tier1, 2: tier2, 3: tier3}

def get_full_nobles():
    return [
        Noble({'diamond': 3, 'emerald': 3, 'ruby': 0, 'sapphire': 0, 'obsidian': 0}),
        Noble({'diamond': 0, 'emerald': 4, 'ruby': 4, 'sapphire': 0, 'obsidian': 0}),
        Noble({'diamond': 4, 'emerald': 0, 'ruby': 0, 'sapphire': 4, 'obsidian': 0}),
        Noble({'diamond': 0, 'emerald': 0, 'ruby': 3, 'sapphire': 3, 'obsidian': 3}),
        Noble({'diamond': 3, 'emerald': 0, 'ruby': 3, 'sapphire': 3, 'obsidian': 0}),
        Noble({'diamond': 0, 'emerald': 4, 'ruby': 0, 'sapphire': 4, 'obsidian': 4}),
        Noble({'diamond': 3, 'emerald': 3, 'ruby': 3, 'sapphire': 0, 'obsidian': 0}),
        Noble({'diamond': 0, 'emerald': 3, 'ruby': 3, 'sapphire': 3, 'obsidian': 0}),
        Noble({'diamond': 3, 'emerald': 0, 'ruby': 0, 'sapphire': 4, 'obsidian': 4}),
    ]

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


