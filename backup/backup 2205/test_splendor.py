import unittest
from game_state import GameState, PlayerState, Card, Action

class TestGameState(unittest.TestCase):

    def setUp(self):
        self.players = [PlayerState(), PlayerState()]
        self.tokens = {
            'diamond': 5, 'sapphire': 5, 'emerald': 5,
            'ruby': 5, 'obsidian': 5, 'gold': 5
        }
        self.board = {
            1: [Card(1, {'ruby': 1}, 'diamond', points=1)],
            2: [],
            3: []
        }
        self.deck = {1: [], 2: [], 3: []}
        self.nobles = []
        self.state = GameState(self.players, 0, self.tokens, self.board, self.nobles, self.deck)

    def test_take_three_tokens(self):
        action = Action("take_tokens", tokens_taken={'diamond': 1, 'sapphire': 1, 'emerald': 1})
        new_state = self.state.apply_action(action)
        p = new_state.players[0]
        self.assertEqual(p.tokens['diamond'], 1)
        self.assertEqual(new_state.tokens['diamond'], 4)

    def test_take_two_same_tokens(self):
        action = Action("take_tokens", tokens_taken={'ruby': 2})
        new_state = self.state.apply_action(action)
        p = new_state.players[0]
        self.assertEqual(p.tokens['ruby'], 2)
        self.assertEqual(new_state.tokens['ruby'], 3)

    def test_buy_card_exact(self):
        self.players[0].tokens['ruby'] = 1
        action = Action("buy_card", target=self.board[1][0])
        new_state = self.state.apply_action(action)
        p = new_state.players[0]
        self.assertEqual(len(p.cards), 1)
        self.assertEqual(p.bonuses['diamond'], 1)
        self.assertEqual(p.points, 1)

    def test_reserve_card(self):
        action = Action("reserve_card", target=self.board[1][0])
        new_state = self.state.apply_action(action)
        p = new_state.players[0]
        self.assertEqual(len(p.reserved), 1)
        self.assertEqual(p.tokens['gold'], 1)
        self.assertEqual(new_state.tokens['gold'], 4)

if __name__ == '__main__':
    unittest.main()



