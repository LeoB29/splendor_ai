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
        # Track official end-of-round finish: when someone hits >=15, finish the round
        # starting from the initial player. Tie-breaker handled on finalize.
        self.start_player: int = current_player
        self.pending_round_end: bool = False
        # Debug guard so we only log one snapshot per state when there are no legal actions
        self._no_legal_logged: bool = False

    def clone(self):
        return deepcopy(self)

    def _clamp_tokens_nonnegative(self, tokens: Dict[str, int]) -> Dict[str, int]:
        """Clamp all token counts to be >= 0. Returns the possibly-updated dict."""
        for k in list(tokens.keys()):
            if tokens[k] < 0:
                tokens[k] = 0
        return tokens

    def _enforce_invariants(self):
        """Best-effort safety: ensure no negative token counts anywhere (incl. gold).
        This prevents cascading issues even if an upstream bug slips through.
        """
        # Clamp bank tokens
        self._clamp_tokens_nonnegative(self.tokens)
        # Clamp player tokens
        for p in self.players:
            self._clamp_tokens_nonnegative(p.tokens)

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

        # Debug snapshot if no legal actions (should be rare)
        if not actions and not self._no_legal_logged:
            try:
                bank_ge4 = [c for c in colors if self.tokens.get(c, 0) >= 4]
                board_sizes = {t: len(cs) for t, cs in self.board.items()}
                deck_sizes = {t: len(ds) for t, ds in self.deck.items()}
                print(
                    "[NoLegal] p=%d bank=%s p_tokens=%s bonuses=%s reserved=%d can_reserve=%s colors_avail=%s bank_ge4=%s board=%s deck=%s"
                    % (
                        self.current_player,
                        dict(self.tokens),
                        dict(player.tokens),
                        dict(player.bonuses),
                        len(player.reserved),
                        player.can_reserve(),
                        colors,
                        bank_ge4,
                        board_sizes,
                        deck_sizes,
                    )
                )
            except Exception:
                pass
            self._no_legal_logged = True
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
            # Guard: never take more from bank than available; never return more than player has
            for c, count in action.tokens_taken.items():
                avail = int(new_state.tokens.get(c, 0))
                take = min(int(count), max(0, avail))
                if take < int(count):
                    try:
                        print(f"[Guard] take_tokens: requested take {count} {c}, bank had {avail}; taking {take}")
                    except Exception:
                        pass
                new_state.tokens[c] = avail - take
                player.tokens[c] = int(player.tokens.get(c, 0)) + take
            for c, count in action.tokens_returned.items():
                have = int(player.tokens.get(c, 0))
                give = min(int(count), max(0, have))
                if give < int(count):
                    try:
                        print(f"[Guard] return_tokens: requested return {count} {c}, player had {have}; returning {give}")
                    except Exception:
                        pass
                player.tokens[c] = have - give
                new_state.tokens[c] = int(new_state.tokens.get(c, 0)) + give

        elif action.action_type == 'reserve':
            card = action.target
            tier = action.tier

            if card in new_state.board[tier]:
                new_state.board[tier].remove(card)
            player.reserved.append(card)

            # Token transfer with guards (same as take_tokens)
            for c, count in action.tokens_taken.items():
                avail = int(new_state.tokens.get(c, 0))
                take = min(int(count), max(0, avail))
                if take < int(count):
                    try:
                        print(f"[Guard] reserve: requested take {count} {c}, bank had {avail}; taking {take}")
                    except Exception:
                        pass
                new_state.tokens[c] = avail - take
                player.tokens[c] = int(player.tokens.get(c, 0)) + take
            for c, count in action.tokens_returned.items():
                have = int(player.tokens.get(c, 0))
                give = min(int(count), max(0, have))
                if give < int(count):
                    try:
                        print(f"[Guard] reserve return: requested return {count} {c}, player had {have}; returning {give}")
                    except Exception:
                        pass
                player.tokens[c] = have - give
                new_state.tokens[c] = int(new_state.tokens.get(c, 0)) + give

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
                    # Spend remaining cost from gold (joker) tokens. Cap by available gold.
                    gold_have = int(player.tokens.get('gold', 0))
                    gold_spend = min(int(effective), max(0, gold_have))
                    if gold_spend < int(effective):
                        try:
                            print(f"[Guard] buy: needed {effective} gold, had {gold_have}; spending {gold_spend}")
                        except Exception:
                            pass
                    player.tokens['gold'] = gold_have - gold_spend
                    new_state.tokens['gold'] = int(new_state.tokens.get('gold', 0)) + gold_spend

            player.cards.append(card)
            player.bonuses[card.bonus_color] += 1
            player.points += card.points

            if action.action_type == "buy_card":
                card = action.target
                tier = card.tier
                # Remove the bought card
                new_state.board[tier].remove(card)
                # Refill from deck if available
                if new_state.deck[tier]:
                    new_state.board[tier].append(new_state.deck[tier].pop(0))
            else:
                for c in player.reserved:
                    if c == card:
                        player.reserved.remove(c)
                        break

            for noble in list(new_state.nobles):
                if all(player.bonuses.get(c, 0) >= req for c, req in noble.requirement.items()):
                    player.points += noble.points
                    new_state.nobles.remove(noble)
                    break  # only one noble visit

        new_state.current_player = (new_state.current_player + 1) % len(new_state.players)
        # Safety: never allow negative counts in bank or player tokens (including gold).
        new_state._enforce_invariants()
        # Official Splendor end condition: if any player has >=15 points, finish the round
        # so that all players have played the same number of turns, then pick winner with
        # tie-break on fewest purchased cards.
        last_player_index = (new_state.current_player - 1) % len(new_state.players)
        new_state.check_terminal()  # update pending flag if threshold reached
        new_state._maybe_finalize_round(last_player_index)
        return new_state

    def check_terminal(self):
        # Mark pending end-of-round if any player has 15 or more points
        if self.is_terminal:
            return
        for i, p in enumerate(self.players):
            if p.points >= 15:
                self.pending_round_end = True
                break

    def _maybe_finalize_round(self, last_player_index: int):
        if self.is_terminal:
            return
        if not self.pending_round_end:
            return
        n = len(self.players)
        end_marker = (self.start_player - 1) % n
        if last_player_index == end_marker:
            # Finalize: determine winner by highest points, then fewest purchased cards
            max_pts = max(p.points for p in self.players)
            candidates = [i for i, p in enumerate(self.players) if p.points == max_pts]
            if len(candidates) == 1:
                self.winner = candidates[0]
            else:
                fewest_cards = min(len(self.players[i].cards) for i in candidates)
                tied = [i for i in candidates if len(self.players[i].cards) == fewest_cards]
                self.winner = tied[0] if len(tied) == 1 else None
            self.is_terminal = True

    def get_reward(self, player_index):
        # Simple reward: prestige points
        return self.players[player_index].points
    
    def get_current_player(self) -> int:
        return self.current_player
    
    @staticmethod
    def random():
        import random

        class DummyCard(Card):
            def __init__(self):
                tier = random.choice([1, 2, 3])
                cost = {c: random.randint(0, 3) for c in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]}
                bonus_color = random.choice(["diamond", "sapphire", "obsidian", "ruby", "emerald"])
                points = random.randint(0, 3)
                super().__init__(tier, cost, bonus_color, points)

        class DummyPlayer(PlayerState):
            def __init__(self):
                super().__init__()
                self.tokens = {c: random.randint(0, 3) for c in ["diamond", "sapphire", "obsidian", "ruby", "emerald", "gold"]}
                self.bonuses = {c: random.randint(0, 2) for c in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]}
                self.points = random.randint(0, 10)
                self.reserved = [DummyCard() for _ in range(random.randint(0, 2))]
                self.cards = [DummyCard() for _ in range(random.randint(0, 3))]

        gs = GameState(
            players=[DummyPlayer() for _ in range(2)],
            current_player=0,
            tokens={c: random.randint(0, 5) for c in ["diamond", "sapphire", "obsidian", "ruby", "emerald", "gold"]},
            board={
                1: [DummyCard() for _ in range(4)],
                2: [DummyCard() for _ in range(4)],
                3: [DummyCard() for _ in range(4)]
            },
            nobles=[],
            deck={
                1: [DummyCard() for _ in range(10)],
                2: [DummyCard() for _ in range(10)],
                3: [DummyCard() for _ in range(10)]
            }
        )

        return gs



if __name__ == "__main__":
    state = GameState.random()
    print("Current player:", state.current_player)
    print("Tokens in bank:", state.tokens)
    print("Player 0 tokens:", state.players[0].tokens)
    print("Player 0 reserved cards:", len(state.players[0].reserved))
    print("Available actions:", state.get_legal_actions())
