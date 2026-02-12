from itertools import combinations
from game_state import *
import numpy as np

# Optional torch import: only needed for masking helpers
try:
    import torch as _torch
except Exception:
    _torch = None


GEM_COLORS = ['diamond', 'sapphire', 'obsidian', 'ruby', 'emerald']
GEM_TO_IDX = {c: i for i, c in enumerate(GEM_COLORS)}
IDX_TO_GEM = {i: c for c, i in GEM_TO_IDX.items()}

TAKE_3_DIFF_COMBOS = list(combinations(range(len(GEM_COLORS)), 3))

# Fixed encoding layout sizes (used for color-permutation augmentation)
CARD_VEC_LEN = 11  # 5 cost + 5 bonus + 1 points
NUM_TIERS = 3
CARDS_PER_TIER = 4
RESERVED_PER_PLAYER = 3
NOBLES_MAX = 10


def _infer_num_players_from_flat_len(n: int) -> int:
    """Infer player count from flattened length; fall back to 2 if unclear."""
    base = NUM_TIERS * CARDS_PER_TIER * CARD_VEC_LEN + 6 + NOBLES_MAX * 5
    per_player = RESERVED_PER_PLAYER * CARD_VEC_LEN + 6 + 5 + 1
    if n >= base and (n - base) % per_player == 0:
        return int((n - base) // per_player)
    return 2


def permute_policy_colors(pi: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Permute the policy vector under a color relabeling.

    perm is a length-5 array where perm[new_idx] = old_idx.
    Only actions that depend on gem colors (take tokens) are remapped.
    """
    pi_arr = np.asarray(pi, dtype=np.float32)
    if pi_arr.shape[-1] != 43:
        return pi_arr
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape[0] != 5:
        return pi_arr

    # Map new action index -> old action index
    mapping = _build_action_permutation(perm)
    # 15-42: color-agnostic actions, unchanged
    return pi_arr[mapping].astype(np.float32)


def _build_action_permutation(perm: np.ndarray) -> list[int]:
    """Return mapping new_index -> old_index under color permutation."""
    mapping = list(range(43))
    combo_to_idx = {cmb: i for i, cmb in enumerate(TAKE_3_DIFF_COMBOS)}
    # 0-9: take 3 different
    for new_idx, cmb in enumerate(TAKE_3_DIFF_COMBOS):
        old_cmb = tuple(sorted(perm[c] for c in cmb))
        mapping[new_idx] = combo_to_idx[old_cmb]
    # 10-14: take 2 same color
    for new_color in range(5):
        mapping[10 + new_color] = 10 + int(perm[new_color])
    return mapping


def permute_action_index(a_idx: int, perm: np.ndarray) -> int:
    """Map an action index under a color relabeling."""
    if a_idx is None or a_idx < 0:
        return -1
    if a_idx >= 43:
        return a_idx
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape[0] != 5:
        return a_idx
    mapping = _build_action_permutation(perm)
    inv = [0] * len(mapping)
    for new, old in enumerate(mapping):
        inv[old] = new
    return int(inv[a_idx])


def permute_token_vector(vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Permute a 6-length token vector (5 colors + gold)."""
    v = np.asarray(vec, dtype=np.float32)
    perm = np.asarray(perm, dtype=np.int64)
    if v.shape[-1] != 6 or perm.shape[0] != 5:
        return v
    out = v.copy()
    out[..., :5] = v[..., :5][..., perm]
    out[..., 5] = v[..., 5]
    return out


def permute_colors_in_flat_state(flat_state: np.ndarray, perm: np.ndarray, num_players: int | None = None) -> np.ndarray:
    """Permute all gem-color features in a flattened state.

    perm is a length-5 array where perm[new_idx] = old_idx.
    Gold (index 5 in token vectors) is not permuted.
    """
    orig = np.asarray(flat_state, dtype=np.float32)
    out = orig.copy()
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape[0] != 5:
        return out

    n_players = int(num_players) if num_players is not None else _infer_num_players_from_flat_len(len(out))

    idx = 0
    # Board cards
    for _ in range(NUM_TIERS * CARDS_PER_TIER):
        out[idx: idx + 5] = orig[idx: idx + 5][perm]
        out[idx + 5: idx + 10] = orig[idx + 5: idx + 10][perm]
        idx += CARD_VEC_LEN

    # Reserved cards (per player)
    for _ in range(n_players * RESERVED_PER_PLAYER):
        out[idx: idx + 5] = orig[idx: idx + 5][perm]
        out[idx + 5: idx + 10] = orig[idx + 5: idx + 10][perm]
        idx += CARD_VEC_LEN

    # Player data blocks
    for _ in range(n_players):
        # Tokens (5 colors + gold)
        out[idx: idx + 5] = orig[idx: idx + 5][perm]
        out[idx + 5] = orig[idx + 5]
        idx += 6
        # Bonuses (5 colors)
        out[idx: idx + 5] = orig[idx: idx + 5][perm]
        idx += 5
        # Points (1)
        out[idx] = orig[idx]
        idx += 1

    # Bank tokens (5 colors + gold)
    out[idx: idx + 5] = orig[idx: idx + 5][perm]
    out[idx + 5] = orig[idx + 5]
    idx += 6

    # Nobles (10 * 5 colors)
    for _ in range(NOBLES_MAX):
        out[idx: idx + 5] = orig[idx: idx + 5][perm]
        idx += 5

    return out


##### input of neural network: action to index

def action_to_index(action: Action, game_state: GameState) -> int:
    player = game_state.players[game_state.current_player]

    if action.action_type == "take_tokens":
        # Count tokens_taken ignoring gold (gold cannot be taken except when reserving)
        tokens_taken = action.tokens_taken
        # Ignore gold token for "take_tokens" (only relevant for reserve)
        gem_tokens = [(GEM_TO_IDX[c], count) for c, count in tokens_taken.items() if c != 'gold' and count > 0]

        if len(gem_tokens) == 3 and all(count == 1 for _, count in gem_tokens):
            # take 3 different tokens
            gems_sorted = tuple(sorted(g for g, _ in gem_tokens))
            if gems_sorted in TAKE_3_DIFF_COMBOS:
                return TAKE_3_DIFF_COMBOS.index(gems_sorted)
            else:
                raise ValueError(f"Invalid 3 different tokens taken: {gems_sorted}")

        elif len(gem_tokens) == 1 and gem_tokens[0][1] == 2:
            # take 2 tokens same color
            gem = gem_tokens[0][0]
            return 10 + gem

        else:
            raise ValueError(f"Unexpected take_tokens action: {tokens_taken}")

    elif action.action_type == "buy_card":
        # Find card index on board
        for tier, cards in game_state.board.items():
            if action.target in cards:
                card_index = cards.index(action.target)
                # Map cards by tier together, so flatten all tiers into a list? We treat all visible cards sequentially
                # Let's flatten all board cards in tier order:
                visible_cards = []
                for t in sorted(game_state.board.keys()):
                    visible_cards.extend(game_state.board[t])
                card_index = visible_cards.index(action.target)
                if card_index >= 0 and card_index < 12:
                    return 15 + card_index
                else:
                    raise ValueError("Buy card index out of range")
        raise ValueError("Buy card not found on board")

    elif action.action_type == "buy_reserved":
        # Reserved cards max 3
        reserved_cards = player.reserved
        if action.target in reserved_cards:
            idx = reserved_cards.index(action.target)
            if idx >= 0 and idx < 3:
                return 27 + idx
        raise ValueError("Buy reserved card not found")

    elif action.action_type == "reserve":
        # Reserve visible cards max 12
        # Similar flattening as buy_card
        visible_cards = []
        for t in sorted(game_state.board.keys()):
            visible_cards.extend(game_state.board[t])
        if action.target in visible_cards:
            idx = visible_cards.index(action.target)
            if idx >= 0 and idx < 12:
                return 30 + idx
        else:
            raise ValueError("Reserve card not found on board")

    elif action.action_type == "take_gold":
        return 42

    else:
        raise ValueError(f"Unknown action_type: {action.action_type}")

### mask non-legal actions

def visible_cards_count(game_state: GameState) -> int:
    count = 0
    for tier in game_state.board:
        count += min(4, len(game_state.board[tier]))
    return count


def legal_actions_mask(game_state: GameState):
    legal = game_state.get_legal_actions()
    mask = [0] * 43
    visible_count = visible_cards_count(game_state)
    reserved_count = len(game_state.players[game_state.current_player].reserved)
    
    for action in legal:
        # Skip actions not represented in the 43-size action space (e.g., reserve-from-deck)
        if getattr(action, 'action_type', None) == 'reserve' and getattr(action, 'from_deck', False):
            continue
        try:
            idx = action_to_index(action, game_state)
            # Guard against unexpected None or out-of-range indices
            if isinstance(idx, int) and 0 <= idx < len(mask):
                mask[idx] = 1
        except Exception:
            # Ignore any actions that cannot be encoded into the fixed action space
            pass
    return mask


def flatten_visible_cards(game_state: GameState):
    visible_cards = []
    for tier in sorted(game_state.board.keys()):
        # Include up to 4 visible cards per tier; skip None entries
        tier_cards = [c for c in game_state.board[tier] if c is not None]
        visible_cards.extend(tier_cards[:4])
    return visible_cards

###### output of neural network: index to action

def _compute_tokens_returned(player_tokens: dict, tokens_taken: dict, limit: int = 10) -> dict:
    # After taking tokens_taken, return enough tokens to satisfy token cap (limit)
    # Greedy: return from colors with highest counts first
    after = player_tokens.copy()
    for c, k in tokens_taken.items():
        after[c] = after.get(c, 0) + k
    total = sum(after.values())
    excess = max(0, total - limit)
    if excess == 0:
        return {}
    # Order colors by current count desc; include gold as valid return
    order = sorted([("diamond"), ("sapphire"), ("obsidian"), ("ruby"), ("emerald"), ("gold")], key=lambda c: after.get(c, 0), reverse=True)
    ret: dict = {}
    i = 0
    while excess > 0 and i < len(order):
        c = order[i]
        available = after.get(c, 0)
        if available <= 0:
            i += 1
            continue
        take = min(available, excess)
        ret[c] = ret.get(c, 0) + take
        after[c] -= take
        excess -= take
        i += 1
        if i >= len(order) and excess > 0:
            # Restart to drain remaining counts if any
            order = sorted(order, key=lambda c: after.get(c, 0), reverse=True)
            i = 0
    return ret


def index_to_action(index: int, game_state: GameState) -> Action:
    player = game_state.players[game_state.current_player]
    visible_cards = flatten_visible_cards(game_state)
    reserved_cards = player.reserved

    if 0 <= index <= 9:
        # Take 3 different tokens
        gems_idx = TAKE_3_DIFF_COMBOS[index]
        tokens_taken = {IDX_TO_GEM[g]: 1 for g in gems_idx}
        # If this would exceed 10 tokens, return a valid set
        tokens_returned = _compute_tokens_returned(player.tokens, tokens_taken)
        return Action("take_tokens", tokens_taken=tokens_taken, tokens_returned=tokens_returned)

    elif 10 <= index <= 14:
        # Take 2 same tokens
        gem_idx = index - 10
        gem = IDX_TO_GEM[gem_idx]
        tokens_taken = {gem: 2}
        tokens_returned = _compute_tokens_returned(player.tokens, tokens_taken)
        return Action("take_tokens", tokens_taken=tokens_taken, tokens_returned=tokens_returned)

    elif 15 <= index <= 26:
        # Buy visible card
        card_index = index - 15
        if card_index < len(visible_cards):
            card = visible_cards[card_index]
            return Action("buy_card", target=card)
        else:
            raise ValueError("buy_card index out of range")

    elif 27 <= index <= 29:
        # Buy reserved card
        card_index = index - 27
        if card_index < len(reserved_cards):
            card = reserved_cards[card_index]
            return Action("buy_reserved", target=card)
        else:
            raise ValueError("buy_reserved index out of range")

    elif 30 <= index <= 41:
        # Reserve visible card
        card_index = index - 30
        if card_index < len(visible_cards):
            card = visible_cards[card_index]
            # You also need to get the tier of this card for reserve action
            # Find tier
            tier = None
            for t, cards in game_state.board.items():
                if card in cards:
                    tier = t
                    break
            # Reserve often gives a gold; if that exceeds 10, add returns
            tokens_taken = {"gold": 1} if game_state.tokens.get("gold", 0) > 0 else {}
            tokens_returned = _compute_tokens_returned(player.tokens, tokens_taken)
            return Action("reserve", target=card, tier=tier, tokens_taken=tokens_taken, tokens_returned=tokens_returned)
        else:
            raise ValueError("reserve index out of range")

    elif index == 42:
        # Take gold token (usually only from reserve action)
        return Action("take_gold")

    else:
        raise ValueError("Invalid action index")
    



####### flatten game state

def _players_in_turn_order(game_state: GameState):
    """Return players rotated so current_player is first.

    Rationale: the network should always see the position from the
    perspective of the player to move. This removes label ambiguity
    where identical inputs would otherwise require different outputs.
    """
    try:
        n = len(game_state.players)
        if n <= 1:
            return list(game_state.players)
        cp = int(getattr(game_state, "current_player", 0)) % n
        return list(game_state.players[cp:]) + list(game_state.players[:cp])
    except Exception:
        # Fallback to original order if anything is unexpected.
        return list(game_state.players)


def flatten_game_state(game_state):
    flat_state = []
    # Always encode players from the current player's perspective.
    players = _players_in_turn_order(game_state)

    # --- Board Cards (4 cards per tier, 3 tiers = 12 cards) ---
    for tier in sorted(game_state.board.keys()):
        cards_in_tier = game_state.board[tier]
        # Encode at most 4 visible cards per tier
        for card in cards_in_tier[:4]:
            if card is None:
                flat_state.extend([0] * (5 + 5 + 1))
            else:
                flat_state.extend(encode_card(card))
        # Pad if fewer than 4 cards in this tier
        for _ in range(max(0, 4 - len(cards_in_tier))):
            flat_state.extend([0] * (5 + 5 + 1))  # cost + bonus + points

    # --- Reserved Cards (up to 3 per player) ---
    for player in players:
        reserved_cards = getattr(player, "reserved", [])  # adjusted attr name if needed
        for card in reserved_cards:
            if card is None:
                flat_state.extend([0] * (5 + 5 + 1))
            else:
                flat_state.extend(encode_card(card))
        for _ in range(3 - len(reserved_cards)):
            flat_state.extend([0] * (5 + 5 + 1))

    # --- Player Data ---
    for player in players:
        # Tokens (6 types)
        flat_state.extend([player.tokens.get(token, 0) for token in ["diamond", "sapphire", "obsidian", "ruby", "emerald", "gold"]])
        # Card bonuses (5 types)
        flat_state.extend([player.bonuses.get(card, 0) for card in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]])
        # Points
        flat_state.append(player.points)

    # --- Bank Tokens ---
    flat_state.extend([game_state.tokens.get(token, 0) for token in ["diamond", "sapphire", "obsidian", "ruby", "emerald", "gold"]])

    # --- Nobles ---
    for noble in game_state.nobles:
        flat_state.extend([noble.requirement.get(gem, 0) for gem in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]])
    # Pad up to 10 nobles
    for _ in range(10 - len(game_state.nobles)):
        flat_state.extend([0] * 5)

    return np.array(flat_state, dtype=np.float32)

def encode_card(card):
    if card is None:
        return [0] * (5 + 5 + 1)
    # Cost: 5 gem types
    cost_vector = [card.cost.get(gem, 0) for gem in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]]
    # Bonus: one-hot for gem type (5 slots)
    bonus_vector = [1 if card.bonus_color == gem else 0 for gem in ["diamond", "sapphire", "obsidian", "ruby", "emerald"]]
    # Points
    points = [card.points]
    return cost_vector + bonus_vector + points



##### functions to define the action space

def build_fixed_action_space(max_cards_per_tier=4, num_tiers=3, max_reserved_per_player=3, num_players=2):
    fixed_actions = []
    colors = ["diamond", "sapphire", "obsidian", "ruby", "emerald"]

    # Take tokens - 3 different
    for combo in combinations(colors, 3):
        tokens_taken = {c: 1 for c in combo}
        fixed_actions.append(Action("take_tokens", tokens_taken=tokens_taken))

    # Take tokens - 2 same
    for c in colors:
        fixed_actions.append(Action("take_tokens", tokens_taken={c: 2}))

    # Buy board card
    for tier in range(1, num_tiers + 1):
        for card_idx in range(max_cards_per_tier):
            fixed_actions.append(Action("buy_card", target=(tier, card_idx)))

    # Buy reserved card
    for player_idx in range(num_players):
        for reserved_idx in range(max_reserved_per_player):
            fixed_actions.append(Action("buy_reserved", target=(player_idx, reserved_idx)))

    # Reserve board card
    for tier in range(1, num_tiers + 1):
        for card_idx in range(max_cards_per_tier):
            fixed_actions.append(Action("reserve", target=(tier, card_idx)))

    return fixed_actions


def map_action_to_index(action, board, current_player_reserved, fixed_actions):
    for i, fixed_action in enumerate(fixed_actions):
        if action.action_type != fixed_action.action_type:
            continue

        if action.action_type == "take_tokens":
            if action.tokens_taken == fixed_action.tokens_taken:
                return i

        elif action.action_type == "buy_card" or action.action_type == "reserve":
            for tier, cards in board.items():
                if action.target in cards:
                    card_index = cards.index(action.target)
                    if (tier, card_index) == fixed_action.target:
                        return i

        elif action.action_type == "buy_reserved":
            if action.target in current_player_reserved:
                reserved_index = current_player_reserved.index(action.target)
                if (0, reserved_index) == fixed_action.target:  # assuming current_player = 0
                    return i

    return None



def map_index_to_action(index, fixed_actions):
    return fixed_actions[index]


def build_action_mask(legal_actions, board, current_player_reserved, fixed_actions):
    mask = [0] * len(fixed_actions)
    for action in legal_actions:
        idx = map_action_to_index(action, board, current_player_reserved, fixed_actions)
        if idx is not None:
            mask[idx] = 1
    return mask


def mask_illegal_actions(q_values, legal_action_indices):
    if _torch is None:
        raise ImportError("PyTorch is required for mask_illegal_actions but is not installed.")
    mask = _torch.full_like(q_values, float('-inf'))  # Start with -inf everywhere
    mask[legal_action_indices] = 0  # Set legal actions to 0
    masked_q_values = q_values + mask
    return masked_q_values
