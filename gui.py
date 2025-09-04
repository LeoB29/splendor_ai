import tkinter as tk
from tkinter import messagebox
from game_state import *
from game_sim import *
from monte_carlo import *
#from Splendor_Gamestate import setup_game, mcts_fn

class SplendorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Splendor Game")
        self.canvas = tk.Canvas(root, width=800, height=600, bg="forest green")
        self.canvas.pack()

        self.game = setup_game()

        self.info_label = tk.Label(root, text="Welcome to Splendor", bg="lightgrey")
        self.info_label.pack(fill=tk.X)

        self.last_action_label = tk.Label(root, text="", bg="lightgrey")
        self.last_action_label.pack(fill=tk.X)

        self.next_button = tk.Button(root, text="Next AI Move", command=self.next_move)
        self.next_button.pack()

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")

        # Bank tokens
        x_offset = 600
        y_offset = 50
        self.canvas.create_text(x_offset, y_offset - 20, text="Bank Tokens", fill="white", font=("Arial", 12, "bold"))
        for idx, (color, count) in enumerate(self.game.tokens.items()):
            self.canvas.create_text(x_offset, y_offset + idx * 20, text=f"{color.capitalize()}: {count}", fill="white", font=("Arial", 10))

        # Display cards
        y = 50
        for tier, cards in self.game.board.items():
            x = 50
            for card in cards:
                self.canvas.create_rectangle(x, y, x+100, y+150, fill="white")
                self.canvas.create_text(x + 50, y + 10, text=f"T{tier}", font=("Arial", 10, "bold"))
                self.canvas.create_text(x + 50, y + 30, text=f"Pts: {card.points}")
                self.canvas.create_text(x + 50, y + 50, text=f"Bonus: {card.bonus_color}")

                cost_y = y + 70
                for color, cost in card.cost.items():
                    self.canvas.create_text(x + 50, cost_y, text=f"{color}: {cost}", font=("Arial", 8))
                    cost_y += 10

                x += 120
            y += 170

        self.canvas.create_text(400, 550, text=f"Player {self.game.current_player + 1}'s turn", fill="white", font=("Arial", 14))

    def next_move(self):
        if self.game.is_terminal:
            messagebox.showinfo("Game Over", f"Player {self.game.winner + 1} wins!")
            return

        action = mcts_fn(self.game)

        # Show last action taken
        self.last_action_label.config(text=f"Last action: {action}")

        # Replace card from deck if needed
        if hasattr(action, 'type') and action.type in ['purchase', 'reserve'] and hasattr(action, 'tier'):
            tier = action.tier
            if self.game.deck[tier]:
                replacement = self.game.deck[tier].pop(0)
                for i, card in enumerate(self.game.board[tier]):
                    if card == action.card:
                        self.game.board[tier][i] = replacement
                        break
            else:
                self.game.board[tier] = [c for c in self.game.board[tier] if c != action.card]

        self.game = self.game.apply_action(action)
        self.draw_board()


if __name__ == "__main__":
    root = tk.Tk()
    app = SplendorGUI(root)
    root.mainloop()


    