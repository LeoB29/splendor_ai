import unittest
from copy import deepcopy
from game_state import *
from nn_input_output import *


class TestSplendorGame(unittest.TestCase):

    def setUp(self):
        self.player = PlayerState()
        self.other_player = PlayerState()

        self.card = Card(tier=1, cost={'ruby': 1}, bonus_color='ruby', points=1)
        self.noble = Noble(requirement={'sapphire': 1}, points=3)

        self.tokens = {'diamond': 5, 'sapphire': 5, 'obsidian': 5, 'ruby': 5, 'emerald': 5, 'gold': 5}
        self.board = {1: [self.card], 2: [], 3: []}
        self.deck = {1: [], 2: [], 3: []}  # not just {3: [...]}
        #self.deck = {1: []}

        self.state = GameState(
            players=[self.player, self.other_player],
            current_player=0,
            tokens=self.tokens.copy(),
            board=self.board.copy(),
            nobles=[self.noble],
            deck=self.deck.copy()
        )

    def test_take_three_tokens(self):
        actions = self.state.get_legal_actions()
        take_action = next((a for a in actions if a.action_type == "take_tokens" and len(a.tokens_taken) == 3), None)
        self.assertIsNotNone(take_action, "Should be able to take 3 tokens")

        new_state = self.state.apply_action(take_action)
        player = new_state.players[0]
        self.assertEqual(player.total_tokens(), 3)

    def test_reserve_card_with_gold(self):
        self.state.tokens['gold'] = 1  # ensure gold is available
        actions = self.state.get_legal_actions()
        reserve_action = next((a for a in actions if a.action_type == "reserve"), None)
        self.assertIsNotNone(reserve_action, "Should be able to reserve a card")

        new_state = self.state.apply_action(reserve_action)
        player = new_state.players[0]
        self.assertEqual(len(player.reserved), 1)
        self.assertEqual(player.tokens['gold'], 1)

    def test_buy_card_with_exact_tokens(self):
        self.player.tokens['ruby'] = 1
        self.assertTrue(self.state.can_afford(self.player, self.card))
        action = Action("buy_card", target=self.card)
        new_state = self.state.apply_action(action)
        player = new_state.players[0]
        self.assertIn(self.card, player.cards)
        self.assertEqual(player.points, 1)

    def test_buy_card_with_gold(self):
        self.player.tokens['gold'] = 1
        self.assertTrue(self.state.can_afford(self.player, self.card))
        action = Action("buy_card", target=self.card)
        new_state = self.state.apply_action(action)
        self.assertIn(self.card, new_state.players[0].cards)
        self.assertEqual(new_state.players[0].tokens['gold'], 0)

    def test_noble_assignment(self):
        self.player.bonuses['sapphire'] = 1
        self.player.cards.append(self.card)

        self.state.check_terminal()
        self.assertFalse(self.state.is_terminal)

        self.state.nobles = [self.noble]
        new_state = self.state.apply_action(Action("buy_card", target=self.card))
        self.assertEqual(new_state.players[0].points, 4)  # 1 from card, 3 from noble
        self.assertEqual(len(new_state.nobles), 0)


    def test_buy_tier3_card(self):
        # Setup player with enough tokens and bonuses to buy a tier 3 card
        card = Card(tier=3, cost={'diamond': 3, 'ruby': 2, 'emerald': 1}, bonus_color='obsidian', points=5)
        self.state.board[3] = [card]

        player = self.player
        player.tokens = {'diamond': 3, 'ruby': 2, 'emerald': 1, 'sapphire': 0, 'obsidian': 0, 'gold': 0}
        player.bonuses = {'diamond': 0, 'ruby': 0, 'emerald': 0, 'sapphire': 0, 'obsidian': 0}

        # Make sure player has no cards or points yet
        player.cards = []
        player.points = 0

        # Find the action to buy this tier 3 card
        actions = self.state.get_legal_actions()
        buy_action = None
        for action in actions:
            if action.action_type == 'buy_card' and action.target == card:
                buy_action = action
                break

        self.assertIsNotNone(buy_action, "No buy action found for the tier 3 card")

        # Apply the buy action
        new_state = self.state.apply_action(buy_action)
        new_player = new_state.players[0]

        # Check that the card was purchased
        self.assertIn(card, new_player.cards)
        self.assertEqual(new_player.points, 5)  # Points from the card
        self.assertEqual(new_player.bonuses['obsidian'], 1)  # Bonus color increased by 1

        # Tokens spent should be deducted from the player and returned to the bank
        self.assertEqual(new_player.tokens['diamond'], 0)
        self.assertEqual(new_player.tokens['ruby'], 0)
        self.assertEqual(new_player.tokens['emerald'], 0)


    def test_card_replacement_after_buy(self):
        # Setup tier 1 card on board and deck
        card_on_board = Card(tier=1, cost={'ruby': 1}, bonus_color='emerald', points=1)
        replacement_card = Card(tier=1, cost={'sapphire': 2}, bonus_color='diamond', points=0)

        self.state.board[1] = [card_on_board]
        self.state.deck[1] = [replacement_card]

        # Setup player tokens and bonuses to afford the card
        self.player.tokens = {'diamond': 0, 'sapphire': 0, 'obsidian': 0, 'ruby': 1, 'emerald': 0, 'gold': 0}
        self.player.bonuses = {'diamond': 0, 'sapphire': 0, 'obsidian': 0, 'ruby': 0, 'emerald': 0}

        # Buy the card
        action = Action(action_type='buy_card', target=card_on_board)
        new_state = self.state.apply_action(action)

        # Check the bought card is added to player's cards
        self.assertIn(card_on_board, new_state.players[0].cards)

        # Check the card is removed from the board
        self.assertNotIn(card_on_board, new_state.board[1])

        # Check the replacement card is now on the board
        self.assertIn(replacement_card, new_state.board[1])


    def test_buy_card_refills_exactly_one(self):
        # Put one card on tier 1 board and two in tier 1 deck
        card_on_board = Card(tier=1, cost={'ruby':1}, bonus_color='emerald', points=1)
        replacement1 = Card(tier=1, cost={'sapphire':2}, bonus_color='diamond', points=0)
        replacement2 = Card(tier=1, cost={'diamond':2}, bonus_color='ruby', points=0)

        self.state.board[1] = [card_on_board]
        self.state.deck[1] = [replacement1, replacement2]

        # Give player exactly the tokens needed
        self.player.tokens = {'diamond':0,'sapphire':0,'obsidian':0,'ruby':1,'emerald':0,'gold':0}
        self.player.bonuses = {c:0 for c in self.player.bonuses}

        # Buy the board card
        buy = next(a for a in self.state.get_legal_actions()
                   if a.action_type=='buy_card' and a.target==card_on_board)
        new_state = self.state.apply_action(buy)

        # After buy: board[1] should still have length 1,
        # containing exactly the first replacement; deck now has length 1.
        self.assertEqual(len(new_state.board[1]), 1)
        self.assertIn(replacement1, new_state.board[1])
        self.assertNotIn(card_on_board, new_state.board[1])
        self.assertEqual(len(new_state.deck[1]), 1)
        self.assertEqual(new_state.deck[1][0], replacement2)

    def test_buy_reserved_does_not_refill_board(self):
        # Reserve a card
        reserve_action = next(
            a for a in self.state.get_legal_actions()
            if a.action_type == 'reserve'
        )

        state_after_reserve = self.state.apply_action(reserve_action)

        # Give the player tokens to afford anything
        player = state_after_reserve.players[0]
        for color in player.tokens:
            player.tokens[color] = 5

        # Snapshot the board state
        board_before_buy = {tier: list(cards) for tier, cards in state_after_reserve.board.items()}

        # Get the card actually in the reserved list (correct reference)
        reserved_card = player.reserved[0]

        # Create and apply buy_reserved action using correct reference
        buy_reserved_action = Action(action_type='buy_reserved', target=reserved_card)
        state_after_buy = state_after_reserve.apply_action(buy_reserved_action)

        # Assert that the board hasn't changed
        self.assertEqual(state_after_buy.board, board_before_buy)

    def test_reserve_refills_exactly_one(self):
        # Put one card on tier 3 board and two in tier 3 deck
        card_on_board = Card(tier=3, cost={'emerald':2}, bonus_color='ruby', points=0)
        r1 = Card(tier=3, cost={'diamond':1}, bonus_color='sapphire', points=0)
        r2 = Card(tier=3, cost={'sapphire':1}, bonus_color='emerald', points=0)

        self.state.board[3] = [card_on_board]
        self.state.deck[3]  = [r1, r2]
        self.state.tokens['gold'] = 1  # allow a reserve

        # Reserve action
        reserve = next(a for a in self.state.get_legal_actions()
                       if a.action_type=='reserve' and a.target==card_on_board)
        new_state = self.state.apply_action(reserve)

        # After reserve: board[3] should length 1 with r1, deck[3] length 1 with r2
        self.assertEqual(len(new_state.board[3]), 1)
        self.assertIn(r1, new_state.board[3])
        self.assertEqual(len(new_state.deck[3]), 1)
        self.assertEqual(new_state.deck[3][0], r2)

if __name__ == '__main__':
    unittest.main()



import unittest
from itertools import combinations
from copy import deepcopy

# Assume your classes Card, Action, GameState, PlayerState, etc. are imported here
# Also assume your action_to_index and index_to_action functions are defined and imported

# Constants for gems and combos
GEM_COLORS = ['diamond', 'sapphire', 'obsidian', 'ruby', 'emerald']
GEM_TO_IDX = {c: i for i, c in enumerate(GEM_COLORS)}
IDX_TO_GEM = {i: c for c, i in GEM_TO_IDX.items()}
TAKE_3_DIFF_COMBOS = list(combinations(range(len(GEM_COLORS)), 3))


class TestSplendorActionEncoding(unittest.TestCase):
    def setUp(self):
        # Setup cards, board, players, game state for tests
        self.card1 = Card(tier=1, cost={'diamond': 1, 'sapphire': 1}, bonus_color='ruby', points=1)
        self.card2 = Card(tier=2, cost={'obsidian': 2}, bonus_color='emerald', points=2)
        self.card3 = Card(tier=3, cost={'ruby': 3}, bonus_color='diamond', points=3)

        board = {
            1: [self.card1],
            2: [self.card2],
            3: [self.card3]
        }

        deck = {1: [], 2: [], 3: []}
        nobles = []

        player = PlayerState()
        player.tokens = {'diamond': 4, 'sapphire': 4, 'obsidian': 4, 'ruby': 4, 'emerald': 4, 'gold': 1}
        player.reserved = [self.card3]

        self.game_state = GameState(
            players=[player], current_player=0,
            tokens={'diamond': 4, 'sapphire': 4, 'obsidian': 4, 'ruby': 4, 'emerald': 4, 'gold': 5},
            board=board,
            nobles=nobles,
            deck=deck
        )

    def test_action_to_index(self):
        # 1) Take 3 different tokens
        action_take_3 = Action("take_tokens", tokens_taken={'diamond': 1, 'sapphire': 1, 'obsidian': 1})
        idx = action_to_index(action_take_3, self.game_state)
        self.assertIn(idx, range(0, 10))

        # 2) Take 2 same tokens (ruby)
        action_take_2 = Action("take_tokens", tokens_taken={'ruby': 2})
        idx = action_to_index(action_take_2, self.game_state)
        self.assertIn(idx, range(10, 15))

        # 3) Buy card from board (card1)
        action_buy_card = Action("buy_card", target=self.card1)
        idx = action_to_index(action_buy_card, self.game_state)
        self.assertIn(idx, range(15, 27))

        # 4) Buy reserved card (card3)
        action_buy_reserved = Action("buy_reserved", target=self.card3)
        idx = action_to_index(action_buy_reserved, self.game_state)
        self.assertIn(idx, range(27, 30))

        # 5) Reserve a card (card2 from tier 2)
        action_reserve = Action("reserve", target=self.card2, tier=2, tokens_taken={'gold': 1})
        idx = action_to_index(action_reserve, self.game_state)
        self.assertIn(idx, range(30, 42))

        # 6) Take gold token (if implemented)
        action_take_gold = Action("take_gold")
        try:
            idx = action_to_index(action_take_gold, self.game_state)
            self.assertEqual(idx, 42)
        except ValueError:
            # If not implemented, test should still pass but print info
            print("Take gold token action not implemented in action_to_index.")

    def test_index_to_action(self):
        # Test known mappings both ways

        # Take 3 different tokens, index 0 -> tokens_taken: diamond, sapphire, obsidian
        action = index_to_action(0, self.game_state)
        self.assertEqual(set(action.tokens_taken.keys()), {'diamond', 'sapphire', 'obsidian'})
        self.assertEqual(action.action_type, "take_tokens")

        # Take 2 same tokens, index 12 (ruby is gem idx 3, 12 - 10 = 2 incorrect)
        # Let's pick index 13 for ruby which is 10 + 3
        action = index_to_action(13, self.game_state)
        self.assertEqual(action.tokens_taken, {'ruby': 2})
        self.assertEqual(action.action_type, "take_tokens")

        # Buy card index 15 corresponds to first visible card (card1)
        action = index_to_action(15, self.game_state)
        self.assertEqual(action.action_type, "buy_card")
        self.assertEqual(action.target, self.card1)

        # Buy reserved card index 27 -> first reserved card (card3)
        action = index_to_action(27, self.game_state)
        self.assertEqual(action.action_type, "buy_reserved")
        self.assertEqual(action.target, self.card3)

        # Reserve card index 31 -> second visible card (card2)
        action = index_to_action(31, self.game_state)
        self.assertEqual(action.action_type, "reserve")
        self.assertEqual(action.target, self.card2)

        # Take gold token index 42 (if implemented)
        try:
            action = index_to_action(42, self.game_state)
            self.assertEqual(action.action_type, "take_gold")
        except ValueError:
            print("Take gold token action not implemented in index_to_action.")

if __name__ == '__main__':
    unittest.main()



# The function you're testing
from nn_input_output import flatten_game_state  # Replace with actual path
from game_state import *

class TestFlattenGameState(unittest.TestCase):

    def test_flatten_game_state_basic(self):
        # Setup mock data
        cards = [
            [Card(1, {"diamond": 1}, "ruby", 1) for _ in range(4)],
            [Card(2, {"sapphire": 2}, "emerald", 2) for _ in range(4)],
            [Card(3, {"obsidian": 3}, "diamond", 3) for _ in range(4)],
        ]
        reserved = [Card(1, {"emerald": 1}, "sapphire", 1) for _ in range(2)]

        player1 = PlayerState()
        player1.tokens = {"diamond": 1, "sapphire": 2, "obsidian": 0, "ruby": 0, "emerald": 0, "gold": 1}
        player1.bonuses = {"diamond": 0, "sapphire": 0, "obsidian": 0, "ruby": 2, "emerald": 0}
        player1.points = 5
        player1.reserved = reserved

        player2 = PlayerState()
        player2.tokens = {"diamond": 0, "sapphire": 0, "obsidian": 3, "ruby": 0, "emerald": 2, "gold": 0}
        player2.bonuses = {"diamond": 1, "sapphire": 0, "obsidian": 0, "ruby": 0, "emerald": 0}
        player2.points = 3
        player2.reserved = []

        bank = {"diamond": 2, "sapphire": 2, "obsidian": 2, "ruby": 2, "emerald": 2, "gold": 2}

        players = [player1, player2]
        current_player = 0
        tokens = {'diamond': 4, 'sapphire': 4, 'obsidian': 4, 'ruby': 4, 'emerald': 4, 'gold': 5}
        board = {1: cards[0], 2: cards[1], 3: cards[2]}
        nobles = [
            Noble({"diamond": 3, "sapphire": 3}),
            Noble({"ruby": 3, "emerald": 3})
        ]
        deck = {1: [], 2: [], 3: []}  # Or mock Card objects

        state = GameState(players, current_player, tokens, board, nobles, deck)

        flattened = flatten_game_state(state)

        # Check shape
        expected_size = 12 * (5 + 5 + 1) + 2 * 3 * (5 + 5 + 1) + 2 * (6 + 5 + 1) + 6 + 10 * 5
        self.assertEqual(flattened.shape[0], expected_size)

        # Optionally, spot-check a few values
        self.assertEqual(flattened[0], 1)  # first card diamond cost
        self.assertIn(1, flattened)  # some points or tokens should be non-zero

if __name__ == '__main__':
    unittest.main()


class TestActionMapping(unittest.TestCase):
    def test_action_to_index_and_back(self):
        fixed_actions = build_fixed_action_space()
        
        # Simulate part of the board
        dummy_card = Card(tier=1, cost={"diamond": 1}, bonus_color="ruby", points=1)
        board = {1: [dummy_card], 2: [], 3: []}
        reserved = [dummy_card]

        # Simulate an actual legal action: take 3 tokens
        legal_action = Action("take_tokens", tokens_taken={"diamond": 1, "sapphire": 1, "ruby": 1})

        # Map to index
        index = map_action_to_index(legal_action, board, reserved, fixed_actions)
        self.assertIsNotNone(index)

        # Map back
        recovered_action = map_index_to_action(index, fixed_actions)
        self.assertEqual(legal_action.action_type, recovered_action.action_type)
        self.assertEqual(legal_action.tokens_taken, recovered_action.tokens_taken)

    def test_buy_card_mapping(self):
        fixed_actions = build_fixed_action_space()
        dummy_card = Card(tier=1, cost={"diamond": 1}, bonus_color="ruby", points=1)
        board = {1: [dummy_card], 2: [], 3: []}
        reserved = []

        legal_action = Action("buy_card", target=dummy_card)

        index = map_action_to_index(legal_action, board, reserved, fixed_actions)
        self.assertIsNotNone(index)

        recovered_action = map_index_to_action(index, fixed_actions)
        self.assertEqual(recovered_action.action_type, "buy_card")
        self.assertEqual(recovered_action.target, (1, 0))  # position on board

if __name__ == '__main__':
    unittest.main()


#### not connected to actual code

import torch

class DummyPolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

class TestPolicyMasking(unittest.TestCase):
    def test_masked_softmax(self):
        input_size = 10
        action_size = 5

        model = DummyPolicyNetwork(input_size, action_size)
        dummy_input = torch.rand((1, input_size))

        logits = model(dummy_input)  # shape: [1, action_size]

        # Example mask: only actions 0, 2, and 3 are legal
        mask = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.bool)

        # Set logits of illegal actions to a very negative number
        masked_logits = logits.masked_fill(~mask, float('-inf'))

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Check that illegal actions get zero probability
        self.assertTrue(torch.all(probs[0][~mask[0]] == 0))
        # Check that legal actions sum to 1
        self.assertAlmostEqual(probs[0][mask[0]].sum().item(), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()