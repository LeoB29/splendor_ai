import unittest
from game_state import GameState, PlayerState, Card, Noble, Action


class TestSplendorGame(unittest.TestCase):

    def setUp(self):
        self.player = PlayerState()
        self.other_player = PlayerState()

        self.card = Card(tier=1, cost={'ruby': 1}, bonus_color='ruby', points=1)
        self.noble = Noble(requirement={'sapphire': 1}, points=3)

        self.tokens = {'diamond': 5, 'sapphire': 5, 'obsidian': 5, 'ruby': 5, 'emerald': 5, 'gold': 5}
        self.board = {1: [self.card]}
        self.deck = {1: []}

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


if __name__ == '__main__':
    unittest.main()
