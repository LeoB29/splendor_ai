import tkinter as tk
from tkinter import messagebox
from typing import List

from game_state import GameState, Action, Card
from cards_init import setup_game
from monte_carlo import mcts_fn


def describe_action(a: Action) -> str:
    t = a.action_type
    if t == "take_tokens":
        take = ",".join(f"{k}:{v}" for k, v in a.tokens_taken.items() if v)
        ret = ",".join(f"{k}:{v}" for k, v in a.tokens_returned.items() if v)
        return f"take [{take}]" + (f" return [{ret}]" if ret else "")
    if t in ("buy_card", "buy_reserved"):
        c: Card = a.target
        pts = getattr(c, 'points', 0)
        bonus = getattr(c, 'bonus_color', '?')
        cost = ",".join(f"{k}:{v}" for k, v in getattr(c, 'cost', {}).items())
        src = "board" if t == "buy_card" else "reserved"
        return f"buy {src} (pts={pts}, bonus={bonus}, cost=[{cost}])"
    if t == "reserve":
        if getattr(a, 'from_deck', False):
            return f"reserve DECK T{a.tier}" + (" +gold" if a.tokens_taken.get('gold', 0) else "")
        else:
            c: Card = a.target
            pts = getattr(c, 'points', 0)
            bonus = getattr(c, 'bonus_color', '?')
            cost = ",".join(f"{k}:{v}" for k, v in getattr(c, 'cost', {}).items())
            g = "+gold" if a.tokens_taken.get('gold', 0) else ""
            return f"reserve T{a.tier} (pts={pts}, bonus={bonus}, cost=[{cost}]) {g}"
    return t


class SplendorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Splendor — Human vs AI (quick)")

        # Layout
        self.canvas = tk.Canvas(root, width=900, height=620, bg="forest green")
        self.canvas.grid(row=0, column=0, rowspan=6, padx=6, pady=6)

        self.info_label = tk.Label(root, text="Welcome to Splendor", bg="lightgrey")
        self.info_label.grid(row=0, column=1, sticky="ew", padx=6, pady=3)

        self.last_action_label = tk.Label(root, text="", bg="lightgrey", anchor="w", justify="left")
        self.last_action_label.grid(row=1, column=1, sticky="ew", padx=6, pady=3)

        self.actions_list = tk.Listbox(root, width=48, height=20)
        self.actions_list.grid(row=2, column=1, sticky="nsew", padx=6)

        self.play_button = tk.Button(root, text="Play Selected", command=self.play_selected)
        self.play_button.grid(row=3, column=1, sticky="ew", padx=6, pady=3)

        self.ai_button = tk.Button(root, text="AI Move", command=self.ai_move)
        self.ai_button.grid(row=4, column=1, sticky="ew", padx=6, pady=3)

        self.reset_button = tk.Button(root, text="New Game", command=self.reset)
        self.reset_button.grid(row=5, column=1, sticky="ew", padx=6, pady=3)

        # Game state
        self.human_index = 0
        self.game: GameState = setup_game(num_players=2)
        self.legal_actions: List[Action] = []

        self.refresh()

    def reset(self):
        self.game = setup_game(num_players=2)
        self.last_action_label.config(text="")
        self.refresh()

    def draw_board(self):
        self.canvas.delete("all")
        # Bank tokens
        x_offset = 720
        y_offset = 40
        self.canvas.create_text(x_offset, y_offset - 20, text="Bank", fill="white", font=("Arial", 12, "bold"))
        for idx, (color, count) in enumerate(self.game.tokens.items()):
            self.canvas.create_text(x_offset, y_offset + idx * 18, text=f"{color[:3]}: {count}", fill="white", font=("Arial", 10))

        # Players panel
        p0 = self.game.players[0]
        p1 = self.game.players[1]
        def p_text(p):
            toks = ",".join(f"{k[:1]}:{v}" for k, v in p.tokens.items())
            bons = ",".join(f"{k[:1]}:{v}" for k, v in p.bonuses.items())
            return f"pts={p.points}  tok[{toks}]  bon[{bons}]  res={len(p.reserved)}"
        self.canvas.create_text(740, 260, text=f"P1 (You)\n{p_text(p0)}", fill="white", font=("Arial", 10), anchor="n")
        self.canvas.create_text(740, 360, text=f"P2 (AI)\n{p_text(p1)}", fill="white", font=("Arial", 10), anchor="n")

        # Board cards (up to 4 per tier)
        y = 40
        for tier in sorted(self.game.board.keys()):
            x = 30
            tier_cards = [c for c in self.game.board[tier] if c is not None][:4]
            for card in tier_cards:
                self.canvas.create_rectangle(x, y, x+110, y+150, fill="white")
                self.canvas.create_text(x + 55, y + 12, text=f"T{tier}", font=("Arial", 10, "bold"))
                self.canvas.create_text(x + 55, y + 30, text=f"Pts: {card.points}")
                self.canvas.create_text(x + 55, y + 45, text=f"Bonus: {card.bonus_color}")
                cost_y = y + 65
                for color, cost in card.cost.items():
                    self.canvas.create_text(x + 55, cost_y, text=f"{color[:3]}: {cost}", font=("Arial", 9))
                    cost_y += 12
                x += 130
            y += 170

        # Turn indicator
        self.canvas.create_text(450, 600, text=f"Turn: Player {self.game.current_player+1}", fill="white", font=("Arial", 12))

    def refresh_actions(self):
        self.actions_list.delete(0, tk.END)
        self.legal_actions = self.game.get_legal_actions()
        if not self.legal_actions:
            self.actions_list.insert(tk.END, "No legal actions (pass)")
            return
        for i, a in enumerate(self.legal_actions):
            self.actions_list.insert(tk.END, describe_action(a))

    def refresh(self):
        if self.game.is_terminal:
            winner = self.game.winner
            messagebox.showinfo("Game Over", f"Player {winner+1} wins!")
        self.draw_board()
        if self.game.current_player == self.human_index:
            self.info_label.config(text="Your turn: select an action and click Play")
            self.refresh_actions()
        else:
            self.info_label.config(text="AI's turn: click AI Move")
            self.actions_list.delete(0, tk.END)
            self.actions_list.insert(tk.END, "Waiting for AI...")

    def play_selected(self):
        if self.game.is_terminal:
            return
        if self.game.current_player != self.human_index:
            messagebox.showinfo("Info", "It's not your turn.")
            return
        if not self.legal_actions:
            # pass turn if no actions
            self.game.current_player = (self.game.current_player + 1) % len(self.game.players)
            self.refresh()
            return
        sel = self.actions_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select an action in the list.")
            return
        a = self.legal_actions[sel[0]]
        self.game = self.game.apply_action(a)
        self.last_action_label.config(text=f"You: {describe_action(a)}")
        self.refresh()

    def ai_move(self):
        if self.game.is_terminal:
            return
        if self.game.current_player == self.human_index:
            messagebox.showinfo("Info", "It's your turn.")
            return
        legal = self.game.get_legal_actions()
        if not legal:
            # pass
            self.game.current_player = (self.game.current_player + 1) % len(self.game.players)
            self.refresh()
            return
        # Use simple MCTS policy for AI
        try:
            action = mcts_fn(self.game)
        except Exception:
            # Fallback to random
            import random
            action = random.choice(legal)
        self.game = self.game.apply_action(action)
        self.last_action_label.config(text=f"AI: {describe_action(action)}")
        self.refresh()


if __name__ == "__main__":
    root = tk.Tk()
    app = SplendorGUI(root)
    root.mainloop()

