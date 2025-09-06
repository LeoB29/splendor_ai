import random
import numpy as np

from cards_init import setup_game
from game_state import Action, GEM_COLORS
from nn_input_output import legal_actions_mask, index_to_action, action_to_index


def _sum_color_supply(state):
    totals = {c: int(state.tokens.get(c, 0)) for c in GEM_COLORS}
    for p in state.players:
        for c in GEM_COLORS:
            totals[c] += int(p.tokens.get(c, 0))
    return totals


def _has_no_negative_tokens(state):
    if any(v < 0 for v in state.tokens.values()):
        return False
    for p in state.players:
        if any(v < 0 for v in p.tokens.values()):
            return False
    return True


def test_token_transfer_guards_and_cap():
    state = setup_game(num_players=2)
    player = state.players[state.current_player]

    # Force a small bank for a color and ask to take more than available
    state.tokens['diamond'] = 1
    before_bank = state.tokens['diamond']
    before_player = player.tokens['diamond']

    # Also ask to return more than the player has of another color
    player.tokens['ruby'] = 0
    act = Action(
        'take_tokens',
        tokens_taken={'diamond': 2},  # request 2, only 1 available
        tokens_returned={'ruby': 3},  # request return 3, have 0
    )
    state2 = state.apply_action(act)

    # Guarded: bank never negative; only up to available taken/returned
    assert state2.tokens['diamond'] >= 0
    assert state2.players[state.current_player].tokens['diamond'] - before_player == min(2, before_bank)

    # Returned ruby cannot exceed what player had (0)
    assert state2.players[state.current_player].tokens['ruby'] == 0

    # Now test token cap auto-return to 10 (gold first)
    state3 = setup_game(num_players=2)
    p = state3.players[state3.current_player]
    # Give player 12 tokens, including 3 gold so excess=2 should consume gold first
    p.tokens.update({'diamond': 3, 'sapphire': 3, 'obsidian': 3, 'ruby': 0, 'emerald': 0, 'gold': 3})
    # Apply a harmless action to trigger invariants (reserve from deck)
    act2 = Action('reserve', target=None, tier=1, from_deck=True)
    state4 = state3.apply_action(act2)
    # The player we modified is the same index as before the action
    p4 = state4.players[state3.current_player]
    assert p4.total_tokens() == 10
    # Gold should be reduced by up to the excess first
    # We expect at least 1 gold remaining (3-2=1) with this setup
    assert p4.tokens['gold'] >= 1


def test_conservation_and_board_invariants_over_random_play():
    state = setup_game(num_players=2)
    initial = _sum_color_supply(state)

    steps = 30
    for _ in range(steps):
        legal = state.get_legal_actions()
        if not legal:
            break
        a = random.choice(legal)
        state = state.apply_action(a)
        # Per-color conservation holds (bank + players == initial)
        now = _sum_color_supply(state)
        for c in GEM_COLORS:
            assert now[c] == initial[c]
        # No negative tokens anywhere
        assert _has_no_negative_tokens(state)
        # Visible per tier <= 4 and no None
        for t, cards in state.board.items():
            assert len(cards) <= 4
            assert all(c is not None for c in cards)


def test_legal_actions_mask_roundtrip_and_skip_out_of_head():
    state = setup_game(num_players=2)
    mask = legal_actions_mask(state)
    assert isinstance(mask, list) and len(mask) == 43

    legal = state.get_legal_actions()

    # For each mask=1 index, index_to_action should map to a currently legal action
    on_idxs = [i for i, v in enumerate(mask) if v]
    for idx in on_idxs:
        act = index_to_action(idx, state)
        # Find an equivalent legal action by type/target/tokens
        found = False
        for la in legal:
            if la.action_type != act.action_type:
                continue
            if act.action_type in ("buy_card", "reserve", "buy_reserved"):
                if la.target is act.target:
                    found = True
                    break
            elif act.action_type == "take_tokens":
                if la.tokens_taken == act.tokens_taken:
                    # tokens_returned may differ by variant; just require subset match
                    found = True
                    break
            elif act.action_type == "take_gold":
                found = True
                break
        assert found, f"Mask index {idx} did not map to a legal action"

    # Mask should not include reserve-from-deck actions (not in 43-head)
    from_deck_legals = [a for a in legal if a.action_type == 'reserve' and getattr(a, 'from_deck', False)]
    if from_deck_legals:
        # None of the from-deck actions should have a valid index in [0,42]
        for a in from_deck_legals:
            raised = False
            try:
                _ = action_to_index(a, state)
            except Exception:
                raised = True
            assert raised
