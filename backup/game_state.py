from copy import deepcopy
from typing import List, Dict, Optional
from itertools import combinations


class Card:
    def __init__(self, tier, cost, bonus_color, points=0):
        self.tier = tier
        self.cost = cost            # Dict like {'red': 2, 'blue': 1}
        self.bonus_color = bonus_color
        self.points = points

class Noble:
    def __init__(self, requirement: Dict[str, int], points: int = 3):
        self.requirement = requirement  # Dict of bonus requirements
        self.points = points

class PlayerState:
    def __init__(self):
        self.tokens = {'diamond': 0, 'sapphire': 0, 'obsidian': 0, 'ruby': 0, 'emerald': 0, 'gold': 0}
        self.bonuses = {'diamond': 0, 'sapphire': 0, 'obsidian': 0, 'ruby': 0, 'emerald': 0}
        self.cards = []       # List[Card]
        self.reserved = []    # List[Card]
        self.points = 0

class GameState:
    def __init__(self, players: List[PlayerState], current_player: int,
                 tokens: Dict[str, int], board: Dict[int, List[Card]],
                 nobles: List[Noble], deck: Dict[int, List[Card]]):
        self.players = players
        self.current_player = current_player
        self.tokens = tokens
        self.board = board
        self.nobles = nobles
        self.deck = deck
        self.is_terminal = False
        self.winner: Optional[int] = None

    def clone(self):
        return deepcopy(self)

    def get_legal_actions(self):
        player = self.players[self.current_player]
        actions = []

        # Token colors excluding gold
        colors = ['diamond', 'sapphire', 'emerald', 'ruby', 'obsidian']

        # 1. Take 3 tokens of different colors (if at least 1 in each pile)
        for combo in combinations(colors, 3):
            if all(self.tokens[color] >= 1 for color in combo):
                taken = {color: 1 for color in combo}
                actions.append(Action("take_tokens", tokens_taken=taken))

        # 2. Take 2 tokens of the same color (only if at least 4 in pile)
        for color in colors:
            if self.tokens[color] >= 4:
                actions.append(Action("take_tokens", tokens_taken={color: 2}))

        # 3. Buy card from board
        for tier_cards in self.board.values():
            for card in tier_cards:
                if self.can_afford(player, card):
                    actions.append(Action("buy_card", target=card))

        # 4. Buy reserved card
        for card in player.reserved:
            if self.can_afford(player, card):
                actions.append(Action("buy_reserved", target=card))

        # 5. Reserve card (if player has <3 reserved and gold tokens available)
        if len(player.reserved) < 3 and self.tokens['gold'] > 0:
            for tier_cards in self.board.values():
                for card in tier_cards:
                    actions.append(Action("reserve_card", target=card))

        return actions

    def can_afford(self, player: PlayerState, card: Card):
        needed = 0
        for color, cost in card.cost.items():
            total = player.tokens[color] + player.bonuses[color]
            if total < cost:
                needed += cost - total

        return needed <= player.tokens['gold']


    def apply_action(self, action):
        new_state = self.clone()
        player = new_state.players[new_state.current_player]

        if action.action_type == "take_tokens":
            for color, count in action.tokens_taken.items():
                new_state.tokens[color] -= count
                player.tokens[color] += count

        elif action.action_type in ["buy_card", "buy_reserved"] and action.target:
            card = action.target
            for color, cost in card.cost.items():
                effective = max(0, cost - player.bonuses[color])
                spend = min(effective, player.tokens[color])
                player.tokens[color] -= spend
                new_state.tokens[color] += spend
                effective -= spend
                if effective > 0:
                    player.tokens['gold'] -= effective
                    new_state.tokens['gold'] += effective

            player.cards.append(card)
            player.bonuses[card.bonus_color] += 1
            player.points += card.points

            if action.action_type == "buy_card":
                for tier, cards in new_state.board.items():
                    if card in cards:
                        cards.remove(card)
                        break
            else:
                if card in player.reserved:
                    player.reserved.remove(card)

            # Check for noble visit
            nobles_to_remove = []
            for noble in new_state.nobles:
                if all(player.bonuses.get(c, 0) >= req for c, req in noble.requirement.items()):
                    player.points += noble.points
                    nobles_to_remove.append(noble)
                    break  # Only one noble may visit
            for noble in nobles_to_remove:
                new_state.nobles.remove(noble)

        elif action.action_type == "reserve_card" and action.target:
            card = action.target
            player.reserved.append(card)
            for tier, cards in new_state.board.items():
                if card in cards:
                    cards.remove(card)
                    break
            player.tokens['gold'] += 1
            new_state.tokens['gold'] -= 1

        new_state.current_player = (new_state.current_player + 1) % len(new_state.players)
        new_state.check_terminal()
        return new_state

    def check_terminal(self):
        # End game condition: player with 15 or more points
        for i, player in enumerate(self.players):
            if player.points >= 15:
                self.is_terminal = True
                self.winner = i  # First player to reach 15 wins
                return

    def get_reward(self, player_index):
        # Simple reward: prestige points
        return self.players[player_index].points
    
    def get_current_player(self) -> int:
        return self.current_player    

class Action:
    def __init__(self, action_type, target=None, tokens_taken=None):
        self.action_type = action_type        # "take_tokens", "buy_card", "buy_reserved", "reserve_card"
        self.target = target                  # Card object (if applicable)
        self.tokens_taken = tokens_taken or {}  # Dict[str, int]

