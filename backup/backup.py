
def setup_game(full_deck, full_nobles, num_players):
    # Shuffle each tier deck
    deck = {}
    board = {}
    for tier in full_deck:
        shuffled = full_deck[tier][:]
        random.shuffle(shuffled)
        board[tier] = shuffled[:4]       # top 4 cards to board
        deck[tier] = shuffled[4:]        # remainder to deck

    # Shuffle nobles and pick 3 or 4 depending on players
    nobles = full_nobles[:]
    random.shuffle(nobles)
    nobles_in_play = nobles[0:3 + (num_players - 2)]  # 3 for 2 players, 4 for 3+ players

    # Setup tokens based on number of players (official rules)
    tokens_count = {2: 4, 3: 5, 4: 7}
    tokens = {
        'diamond': tokens_count[num_players],
        'sapphire': tokens_count[num_players],
        'emerald': tokens_count[num_players],
        'ruby': tokens_count[num_players],
        'obsidian': tokens_count[num_players],
        'gold': 5
    }

    return board, deck, nobles_in_play, tokens



full_deck = get_full_deck()
full_nobles = get_full_nobles()
num_players = 2


# Starting tokens (typical for 2 players)
tokens = {
    'diamond': 4,
    'sapphire': 4,
    'emerald': 4,
    'ruby': 4,
    'obsidian': 4,
    'gold': 5
}

# Initialize players
players = [PlayerState(), PlayerState()]

# Setup board with some cards from the full deck

full_deck = get_full_deck()
board = {
    1: full_deck[1][0:4],  # first 4 cards of tier 1
    2: full_deck[2][0:4],  # first 4 cards of tier 2
    3: full_deck[3][0:4],  # first 4 cards of tier 3
}

# Nobles setup example (some nobles with bonus requirements)
nobles = get_full_nobles()




game_state = GameState(players, current_player=0, tokens=tokens, board=board, nobles=nobles, deck={})

if __name__ == "__main__":
    # Create sample cards
    sample_card = Card(tier=1, cost={'ruby': 1, 'sapphire': 1}, bonus_color='diamond', points=1)
    board = {1: [sample_card for _ in range(4)], 2: [], 3: []}
    deck = {1: [], 2: [], 3: []}
    nobles = [Noble(requirement={'diamond': 1, 'sapphire': 1})]

    # Create players
    players = [PlayerState() for _ in range(2)]

    # Initialize tokens
    tokens = {'diamond': 4, 'sapphire': 4, 'emerald': 4, 'ruby': 4, 'obsidian': 4, 'gold': 5}

    # Create game state
    game = GameState(players, current_player=0, tokens=tokens, board=board, nobles=nobles, deck=deck)

    # Simulate game
    winner, points = simulate_game(game, verbose=True)
