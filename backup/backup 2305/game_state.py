from copy import deepcopy
from typing import List, Dict, Optional
from itertools import combinations, chain


class Card:
    def __init__(self, tier, cost, bonus_color, points=0):
        self.tier = tier
        self.cost = cost            # Dict like {'red': 2, 'blue': 1}
        self.bonus_color = bonus_color
        self.points = points
    def __eq__(self, other):
        return isinstance(other, Card) and (
            self.tier == other.tier and
            self.cost == other.cost and
            self.bonus_color == other.bonus_color and
            self.points == other.points
        )
    def __hash__(self):
        return hash((self.tier, tuple(sorted(self.cost.items())), self.bonus_color, self.points))

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
    def total_tokens(self):
        return sum(self.tokens.values())
    def can_reserve(self):
        return len(self.reserved) < 3

class Action:
    def __init__(self, action_type: str, target=None, tokens_taken=None, tokens_returned=None, tier=None):
        self.action_type = action_type  # "take_tokens", "buy_card", "buy_reserved", "reserve"
        self.target = target            # Card (for buy/reserve)
        self.tokens_taken = tokens_taken or {}
        self.tokens_returned = tokens_returned or {}
        self.tier = tier                # For reserve card



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
        actions = []
        player = self.players[self.current_player]
        total_tokens = player.total_tokens()
        colors = [c for c in self.tokens if self.tokens[c] > 0 and c != 'gold']

        # Take 3 different tokens
        for combo in combinations(colors, 3):
            tokens_taken = {c: 1 for c in combo}
            new_total = total_tokens + 3
            if new_total <= 10:
                actions.append(Action("take_tokens", tokens_taken=tokens_taken))
            else:
                excess = new_total - 10
                self._add_return_combinations(actions, player, tokens_taken, excess, action_type="take_tokens")

        # Take 2 of one color
        for c in colors:
            if self.tokens[c] >= 4:
                tokens_taken = {c: 2}
                new_total = total_tokens + 2
                if new_total <= 10:
                    actions.append(Action("take_tokens", tokens_taken=tokens_taken))
                else:
                    excess = new_total - 10
                    self._add_return_combinations(actions, player, tokens_taken, excess, action_type="take_tokens")

        # Buy card from board
        for tier_cards in self.board.values():
            for card in tier_cards:
                if self.can_afford(player, card):
                    actions.append(Action("buy_card", target=card))

        # Buy reserved card
        for card in player.reserved:
            if self.can_afford(player, card):
                actions.append(Action("buy_reserved", target=card))

        # Reserve a card
        if player.can_reserve():
            for tier, cards in self.board.items():
                for card in cards:
                    tokens_taken = {'gold': 1} if self.tokens['gold'] > 0 else {}
                    new_total = total_tokens + sum(tokens_taken.values())
                    if new_total <= 10:
                        actions.append(Action("reserve", target=card, tier=tier, tokens_taken=tokens_taken))
                    else:
                        excess = new_total - 10
                        self._add_return_combinations(actions, player, tokens_taken, excess,
                                                    action_type="reserve", target=card, tier=tier)

        return actions

    def can_afford(self, player: PlayerState, card: Card):
        needed = 0
        for color, cost in card.cost.items():
            total = player.tokens[color] + player.bonuses[color]
            if total < cost:
                needed += cost - total

        return needed <= player.tokens['gold']
    
    def _add_return_combinations(self, actions, player, tokens_taken, excess, action_type, target=None, tier=None):
        after_take = player.tokens.copy()
        for color, count in tokens_taken.items():
            after_take[color] = after_take.get(color, 0) + count

        token_list = list(chain.from_iterable([[color] * count for color, count in after_take.items()]))
        unique_return_sets = set()

        for combo in combinations(token_list, excess):
            ret_counter = {}
            for c in combo:
                ret_counter[c] = ret_counter.get(c, 0) + 1
            unique_return_sets.add(frozenset(ret_counter.items()))

        for ret_set in unique_return_sets:
            tokens_returned = dict(ret_set)
            if all(after_take.get(c, 0) >= count for c, count in tokens_returned.items()):
                actions.append(Action(
                    action_type,
                    target=target,
                    tier=tier,
                    tokens_taken=tokens_taken,
                    tokens_returned=tokens_returned
                ))
    

    def apply_action(self, action: Action):
        new_state = self.clone()
        player = new_state.players[new_state.current_player]

        if action.action_type == 'take_tokens':
            for c, count in action.tokens_taken.items():
                new_state.tokens[c] -= count
                player.tokens[c] += count
            for c, count in action.tokens_returned.items():
                player.tokens[c] -= count
                new_state.tokens[c] += count

        elif action.action_type == 'reserve':
            card = action.target
            tier = action.tier

            if card in new_state.board[tier]:
                new_state.board[tier].remove(card)
            player.reserved.append(card)

            for c, count in action.tokens_taken.items():
                new_state.tokens[c] -= count
                player.tokens[c] += count
            for c, count in action.tokens_returned.items():
                player.tokens[c] -= count
                new_state.tokens[c] += count

            if new_state.deck[tier]:
                new_card = new_state.deck[tier].pop(0)
                new_state.board[tier].append(new_card)

        elif action.action_type in ['buy_card', 'buy_reserved']:
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
                for tier_cards in new_state.board.values():
                    if card in tier_cards:
                        tier_cards.remove(card)
                        break
            else:
                player.reserved.remove(card)

            for noble in list(new_state.nobles):
                if all(player.bonuses.get(c, 0) >= req for c, req in noble.requirement.items()):
                    player.points += noble.points
                    new_state.nobles.remove(noble)
                    break  # only one noble visit

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

