import unittest

from game_state import GameState, Card, PlayerState, Noble, Action
from nn_input_output import (
    action_to_index,
    index_to_action,
    legal_actions_mask,
    flatten_game_state,
)


class TestNNIO(unittest.TestCase):
    def setUp(self):
        # Deterministic small state with ample bank tokens
        self.p1 = PlayerState()
        self.p2 = PlayerState()
        self.players = [self.p1, self.p2]
        self.tokens = {
            'diamond': 5, 'sapphire': 5, 'obsidian': 5,
            'ruby': 5, 'emerald': 5, 'gold': 5,
        }
        # Board with two simple tier-1 cards
        self.card_ruby = Card(tier=1, cost={'ruby': 1}, bonus_color='emerald', points=1)
        self.card_diamond = Card(tier=1, cost={'diamond': 2}, bonus_color='ruby', points=0)
        board = {1: [self.card_ruby, self.card_diamond], 2: [], 3: []}
        deck = {1: [], 2: [], 3: []}
        nobles = [Noble({'sapphire': 1})]
        self.state = GameState(self.players, 0, self.tokens.copy(), board, nobles, deck)

    def test_legal_actions_mask_marks_legals(self):
        legal = self.state.get_legal_actions()
        mask = legal_actions_mask(self.state)
        # Every legal action that can be indexed should set a mask bit
        hit = 0
        for a in legal:
            try:
                idx = action_to_index(a, self.state)
                self.assertEqual(mask[idx], 1)
                hit += 1
            except ValueError:
                # Some actions (e.g., complex return-token variants) may be skipped by mapping
                pass
        self.assertGreater(hit, 0)

    def test_action_index_roundtrip_take3_take2(self):
        legal = self.state.get_legal_actions()
        a3 = next(a for a in legal if a.action_type == 'take_tokens' and len(a.tokens_taken) == 3)
        a2 = next(a for a in legal if a.action_type == 'take_tokens' and list(a.tokens_taken.values()) == [2])

        for a in [a3, a2]:
            idx = action_to_index(a, self.state)
            a_back = index_to_action(idx, self.state)
            self.assertEqual(a_back.action_type, 'take_tokens')
            self.assertEqual(a_back.tokens_taken, a.tokens_taken)

    def test_action_index_roundtrip_buy_and_reserve(self):
        # Make buy-ruby affordable
        self.p1.tokens['ruby'] = 1
        legal = self.state.get_legal_actions()
        buy = next(a for a in legal if a.action_type == 'buy_card' and a.target == self.card_ruby)
        res = next(a for a in legal if a.action_type == 'reserve' and a.target in [self.card_ruby, self.card_diamond])

        for a in [buy, res]:
            idx = action_to_index(a, self.state)
            a_back = index_to_action(idx, self.state)
            self.assertEqual(a_back.action_type, a.action_type)
            # Target must match for buy/reserve
            self.assertEqual(a_back.target, a.target)

    def test_flatten_game_state_nonempty(self):
        flat = flatten_game_state(self.state)
        self.assertIsNotNone(flat)
        self.assertGreater(len(flat), 0)


if __name__ == '__main__':
    unittest.main()

