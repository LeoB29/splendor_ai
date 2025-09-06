from copy import deepcopy
from typing import List, Dict, Optional
from itertools import combinations, chain

# Debug toggle for guard logs
DEBUG_GUARDS = False

# Fixed gem color order to match encoding elsewhere
GEM_COLORS = ["diamond", "sapphire", "obsidian", "ruby", "emerald"]


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
    def __init__(self, action_type: str, target=None, tokens_taken=None, tokens_returned=None, tier=None, from_deck: bool = False):
        self.action_type = action_type  # "take_tokens", "buy_card", "buy_reserved", "reserve"
        self.target = target            # Card (for buy/reserve); None when reserving from deck
        self.tokens_taken = tokens_taken or {}
        self.tokens_returned = tokens_returned or {}
        self.tier = tier                # For reserve card
        self.from_deck = from_deck      # Reserve top card from deck (not visible)



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
        # Track initial token supply (per color) to enforce conservation
        try:
            supply = {c: int(tokens.get(c, 0)) for c in GEM_COLORS}
            for p in players:
                for c in GEM_COLORS:
                    supply[c] += int(p.tokens.get(c, 0))
            self._initial_supply = supply  # type: ignore[attr-defined]
            # Gold supply as well (optional)
            self._initial_gold = int(tokens.get('gold', 0)) + sum(int(p.tokens.get('gold', 0)) for p in players)  # type: ignore[attr-defined]
        except Exception:
            self._initial_supply = {c: 0 for c in GEM_COLORS}  # type: ignore[attr-defined]
            self._initial_gold = int(tokens.get('gold', 0))  # type: ignore[attr-defined]

    def clone(self):
        return deepcopy(self)

    def _dbg(self, msg: str) -> None:
        # Print debug messages only if enabled on instance or module flag
        if getattr(self, 'debug', False) or DEBUG_GUARDS:
            try:
                print(msg)
            except Exception:
                pass

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
        # Enforce per-player token cap (10) by auto-returning excess to bank (gold first, then highest counts)
        try:
            for i, p in enumerate(self.players):
                total = p.total_tokens()
                if total > 10:
                    excess = total - 10
                    order = ['gold'] + GEM_COLORS  # prefer returning gold first
                    # Re-order colors (after gold) by current count desc
                    order = ['gold'] + [c for c, _ in sorted(((c, p.tokens.get(c, 0)) for c in GEM_COLORS), key=lambda t: t[1], reverse=True)]
                    for c in order:
                        if excess <= 0:
                            break
                        have = int(p.tokens.get(c, 0))
                        if have <= 0:
                            continue
                        give = min(have, excess)
                        p.tokens[c] = have - give
                        self.tokens[c] = int(self.tokens.get(c, 0)) + give
                        excess -= give
                    if (getattr(self, 'debug', False) or DEBUG_GUARDS):
                        self._dbg(f"[AutoReturn] Player {i} returned to cap; now total={p.total_tokens()}")
        except Exception:
            pass
        # Ensure visible board doesn't exceed 4 per tier and remove None placeholders
        try:
            for t in list(self.board.keys()):
                cards = [c for c in self.board[t] if c is not None]
                if len(cards) > 4:
                    cards = cards[:4]
                self.board[t] = cards
        except Exception:
            pass
        # Conservation: bank + players should equal initial supply per colored token
        try:
            supply = getattr(self, '_initial_supply', None)
            if supply:
                for c in GEM_COLORS:
                    players_sum = sum(int(p.tokens.get(c, 0)) for p in self.players)
                    total = int(self.tokens.get(c, 0)) + players_sum
                    target = int(supply.get(c, 0))
                    if total != target:
                        # Reconcile by adjusting bank to match initial supply
                        fix = max(0, target - players_sum)
                        if fix != int(self.tokens.get(c, 0)):
                            if getattr(self, 'debug', False) or DEBUG_GUARDS:
                                self._dbg(f"[Conserve] Adjust bank {c}: {self.tokens.get(c,0)} -> {fix} (players={players_sum}, target={target})")
                            self.tokens[c] = fix
            # Optional: gold conservation (may change with different rules, so only adjust if mismatch is small)
            try:
                target_gold = int(getattr(self, '_initial_gold', 0))
                players_gold = sum(int(p.tokens.get('gold', 0)) for p in self.players)
                total_gold = int(self.tokens.get('gold', 0)) + players_gold
                if target_gold and total_gold != target_gold:
                    fixg = max(0, target_gold - players_gold)
                    self.tokens['gold'] = fixg
            except Exception:
                pass
        except Exception:
            pass

    # ------------------ Debug integrity checks (enabled when state.debug or DEBUG_GUARDS) ------------------
    def _debug_check_integrity(self) -> None:
        if not (getattr(self, 'debug', False) or DEBUG_GUARDS):
            return
        # Non-negative counts
        for k, v in self.tokens.items():
            assert v >= 0, f"[Assert] Bank {k} negative: {v}"
        for i, p in enumerate(self.players):
            for k, v in p.tokens.items():
                assert v >= 0, f"[Assert] Player {i} {k} negative: {v}"
            # Token cap per player
            assert p.total_tokens() <= 10, f"[Assert] Player {i} exceeds 10 tokens: {p.total_tokens()}"
        # Board integrity: <=4 visible and no None
        for t, cards in self.board.items():
            assert len(cards) <= 4, f"[Assert] Tier {t} has >4 cards: {len(cards)}"
            for c in cards:
                assert c is not None, f"[Assert] Tier {t} has None card"
        # Conservation per colored gem
        supply = getattr(self, '_initial_supply', None)
        if supply:
            for c in GEM_COLORS:
                players_sum = sum(int(p.tokens.get(c, 0)) for p in self.players)
                total = int(self.tokens.get(c, 0)) + players_sum
                assert total == int(supply.get(c, 0)), (
                    f"[Assert] Supply mismatch for {c}: bank={self.tokens.get(c,0)} players={players_sum} "
                    f"total={total} target={supply.get(c,0)}"
                )
        # Gold conservation (optional; only check if initial known)
        if hasattr(self, '_initial_gold'):
            tg = int(getattr(self, '_initial_gold', 0))
            if tg > 0:
                pg = sum(int(p.tokens.get('gold', 0)) for p in self.players)
                totg = int(self.tokens.get('gold', 0)) + pg
                assert totg == tg, f"[Assert] Gold mismatch: bank={self.tokens.get('gold',0)} players={pg} total={totg} target={tg}"

    def get_legal_actions(self):
        actions = []
        player = self.players[self.current_player]
        total_tokens = player.total_tokens()
        # Colors available in bank (excluding gold)
        colors = [c for c in GEM_COLORS if self.tokens.get(c, 0) > 0]

        # Take 3 different tokens (with dedupe when <3 colors available)
        avail = [c for c in GEM_COLORS if self.tokens.get(c, 0) > 0]
        rep_combos = []  # list of 3-color tuples to encode action indices consistently
        if len(avail) >= 3:
            rep_combos = [tuple(sorted(cmb)) for cmb in combinations(avail, 3)]
        elif len(avail) == 2:
            # choose the lexicographically first missing color as representative
            missing = [c for c in GEM_COLORS if c not in avail]
            third = missing[0]
            rep_combos = [tuple(sorted((avail[0], avail[1], third)))]
        elif len(avail) == 1:
            missing = [c for c in GEM_COLORS if c != avail[0]]
            third = missing[0]
            fourth = missing[1]
            rep_combos = [tuple(sorted((avail[0], third, fourth)))]
        else:
            rep_combos = []

        for combo in rep_combos:
            tokens_taken_full = {c: 1 for c in combo}
            effective_take = {c: 1 for c in combo if self.tokens.get(c, 0) > 0}
            eff_count = sum(effective_take.values())
            if eff_count <= 0:
                continue
            new_total = total_tokens + eff_count
            if new_total <= 10:
                actions.append(Action("take_tokens", tokens_taken=tokens_taken_full))
            else:
                excess = new_total - 10
                self._add_return_combinations_effective(
                    actions, player, tokens_taken_full, effective_take, excess, action_type="take_tokens"
                )

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
                if card is None:
                    continue
                if self.can_afford(player, card):
                    actions.append(Action("buy_card", target=card))

        # Buy reserved card
        for card in player.reserved:
            if card is None:
                continue
            if self.can_afford(player, card):
                actions.append(Action("buy_reserved", target=card))

        # Reserve a card (visible)
        if player.can_reserve():
            for tier, cards in self.board.items():
                for card in cards:
                    if card is None:
                        continue
                    tokens_taken = {'gold': 1} if self.tokens['gold'] > 0 else {}
                    new_total = total_tokens + sum(tokens_taken.values())
                    if new_total <= 10:
                        actions.append(Action("reserve", target=card, tier=tier, tokens_taken=tokens_taken))
                    else:
                        excess = new_total - 10
                        self._add_return_combinations(actions, player, tokens_taken, excess,
                                                    action_type="reserve", target=card, tier=tier)

        # Reserve from deck (top card), allowed even without taking gold
        if player.can_reserve():
            for tier, deck_cards in self.deck.items():
                if deck_cards:
                    tokens_taken = {'gold': 1} if self.tokens.get('gold', 0) > 0 else {}
                    eff_gain = sum(tokens_taken.values())
                    new_total = total_tokens + eff_gain
                    if new_total <= 10:
                        actions.append(Action("reserve", target=None, tier=tier, tokens_taken=tokens_taken, from_deck=True))
                    else:
                        excess = new_total - 10
                        # For reserve-from-deck, effective_take is just tokens_taken (gold or empty)
                        effective_take = tokens_taken.copy()
                        self._add_return_combinations_effective(
                            actions, player, tokens_taken, effective_take, excess,
                            action_type="reserve", target=None, tier=tier
                        )

        # Debug snapshot if no legal actions (should be rare)
        if not actions and not self._no_legal_logged:
            try:
                bank_ge4 = [c for c in colors if self.tokens.get(c, 0) >= 4]
                board_sizes = {t: len(cs) for t, cs in self.board.items()}
                deck_sizes = {t: len(ds) for t, ds in self.deck.items()}
                other_idx = (self.current_player + 1) % len(self.players)
                other_player = self.players[other_idx]
                print(
                    "[NoLegal] p=%d bank=%s p_tokens=%s opp_tokens=%s bonuses=%s reserved=%d can_reserve=%s colors_avail=%s bank_ge4=%s board=%s deck=%s"
                    % (
                        self.current_player,
                        dict(self.tokens),
                        dict(player.tokens),
                        dict(other_player.tokens),
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

    def can_afford(self, player: PlayerState, card: Optional[Card]):
        if card is None:
            return False
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

    def _add_return_combinations_effective(self, actions, player, tokens_taken_full, effective_take, excess, action_type, target=None, tier=None):
        # Build after_take using only effectively obtainable tokens (respecting bank shortages)
        after_take = player.tokens.copy()
        for color, count in effective_take.items():
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
                    tokens_taken=tokens_taken_full,  # keep 3-color encoding for indexing
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
                    self._dbg(f"[Guard] take_tokens: requested take {count} {c}, bank had {avail}; taking {take}")
                new_state.tokens[c] = avail - take
                player.tokens[c] = int(player.tokens.get(c, 0)) + take
            for c, count in action.tokens_returned.items():
                have = int(player.tokens.get(c, 0))
                give = min(int(count), max(0, have))
                if give < int(count):
                    self._dbg(f"[Guard] return_tokens: requested return {count} {c}, player had {have}; returning {give}")
                player.tokens[c] = have - give
                new_state.tokens[c] = int(new_state.tokens.get(c, 0)) + give

        elif action.action_type == 'reserve':
            card = action.target
            tier = action.tier

            from_deck = bool(getattr(action, 'from_deck', False))
            if from_deck:
                # Reserve top card from deck (do not affect board)
                if new_state.deck.get(tier):
                    reserved_card = new_state.deck[tier].pop(0)
                    player.reserved.append(reserved_card)
                else:
                    # Nothing to reserve; keep behavior no-op on the card move
                    pass
            else:
                # Reserve visible card: remove from board
                if card in new_state.board[tier]:
                    new_state.board[tier].remove(card)
                player.reserved.append(card)

            # Token transfer with guards (same as take_tokens)
            for c, count in action.tokens_taken.items():
                avail = int(new_state.tokens.get(c, 0))
                take = min(int(count), max(0, avail))
                if take < int(count):
                    self._dbg(f"[Guard] reserve: requested take {count} {c}, bank had {avail}; taking {take}")
                new_state.tokens[c] = avail - take
                player.tokens[c] = int(player.tokens.get(c, 0)) + take
            for c, count in action.tokens_returned.items():
                have = int(player.tokens.get(c, 0))
                give = min(int(count), max(0, have))
                if give < int(count):
                    self._dbg(f"[Guard] reserve return: requested return {count} {c}, player had {have}; returning {give}")
                player.tokens[c] = have - give
                new_state.tokens[c] = int(new_state.tokens.get(c, 0)) + give

            # Refill only if we reserved a visible card (board slot emptied)
            if not from_deck:
                if new_state.deck[tier]:
                    new_card = new_state.deck[tier].pop(0)
                    if new_card is not None:
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
                        self._dbg(f"[Guard] buy: needed {effective} gold, had {gold_have}; spending {gold_spend}")
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

        # Debug: verify integrity before global invariants adjust (to catch issues early)
        try:
            new_state._debug_check_integrity()
        except AssertionError as e:
            # Re-raise to make failures visible in debug mode
            raise

        new_state.current_player = (new_state.current_player + 1) % len(new_state.players)
        # Safety: never allow negative counts in bank or player tokens (including gold).
        new_state._enforce_invariants()
        # Lightweight runtime warning for token-cap breaches (non-fatal), post-fix
        try:
            warned = getattr(new_state, '_cap_warned', False)
            if not warned:
                for i, p in enumerate(new_state.players):
                    tt = p.total_tokens()
                    if tt > 10:
                        at = getattr(action, 'action_type', '?')
                        print(f"[Warn] Token cap breach: player {i} has {tt} tokens after {at}")
                        new_state._cap_warned = True
                        break
        except Exception:
            pass
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
