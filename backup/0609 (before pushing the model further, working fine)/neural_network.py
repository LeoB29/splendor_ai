import torch
import torch.nn as nn
import torch.nn.functional as F
from game_state import *
from nn_input_output import *



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---- Dummy GameState, Action, and your imports ----
from game_state import GameState, Action  # You already have these



data = [(GameState.random(), Action("take_tokens", tokens_taken={"diamond":1, "sapphire":1, "obsidian":1})) for _ in range(10)]

# ---- Policy Network (as defined earlier) ----
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x, mask):
        logits = self.net(x)
        # Apply -inf to illegal actions
        mask_inf = (1 - mask) * -1e9
        masked_logits = logits + mask_inf
        return masked_logits


def select_action(model: nn.Module, game_state: GameState, device: str = 'cpu', greedy: bool = True):
    """Runs a forward pass using flatten_game_state + legal_actions_mask and returns an Action.

    - greedy=True picks argmax; otherwise samples from softmax over legal actions.
    """
    model.eval()
    state_vec = torch.tensor(flatten_game_state(game_state), dtype=torch.float32, device=device).unsqueeze(0)
    mask_vec = torch.tensor(legal_actions_mask(game_state), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(state_vec, mask_vec)
        if greedy:
            idx = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.multinomial(probs[0], num_samples=1).item())
    return index_to_action(idx, game_state)

# ---- Dummy Dataset ----
class GameDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of (GameState, Action)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_state, action = self.data[idx]
        state = flatten_game_state(game_state)
        target = action_to_index(action, game_state)
        mask = legal_actions_mask(game_state)
        return torch.tensor(state, dtype=torch.float32), \
               torch.tensor(target, dtype=torch.long), \
               torch.tensor(mask, dtype=torch.float32)



#### added 3108 codex

# ---- Generate Mock Data (replace with real data later) ----
def generate_mock_data(n=100):
    data = []
    for _ in range(n):
        gs = GameState.random()  # You should implement a random state generator
        legal = gs.get_legal_actions()
        if not legal: continue
        action = np.random.choice(legal)
        data.append((gs, action))
    return data



# ---- Training ----
def train():
    input_size = len(flatten_game_state(GameState.random()))
    action_size = 43

    model = PolicyNetwork(input_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='none')

    data = generate_mock_data(200)
    loader = DataLoader(GameDataset(data), batch_size=32, shuffle=True)

    for epoch in range(10):
        total_loss = 0
        for x, y, mask in loader:
            logits = model(x, mask)
            loss_raw = criterion(logits, y)
            loss_masked = (loss_raw * mask[range(len(y)), y]).mean()

            optimizer.zero_grad()
            loss_masked.backward()
            optimizer.step()
            total_loss += loss_masked.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()


