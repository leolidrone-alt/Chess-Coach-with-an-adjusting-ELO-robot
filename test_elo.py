import torch
from train_sft import ConditionalMaia, NUM_MOVES, device
model = ConditionalMaia(NUM_MOVES).to(device)
state = torch.load('checkpoints/cond_maia_best.pth', map_location=device)
model.load_state_dict(state['model'])
model.eval()
dummy = torch.randn(1, 112, 8, 8).to(device)
elo_plane = torch.zeros(1, 1, 8, 8).to(device)
_, elo_norm = model(dummy, elo_plane)
print("归一化 Elo 预测值:", elo_norm.item())