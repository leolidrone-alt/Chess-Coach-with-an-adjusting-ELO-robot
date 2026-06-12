import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, sys, glob, math
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data import build_move_indices
UCI_TO_IDX, _ = build_move_indices()
NUM_MOVES = len(UCI_TO_IDX)
print(f"策略头输出维度: {NUM_MOVES}")

# ==================== 超参数 ====================
BATCH_SIZE        = 256
EPOCHS            = 15
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.05
ELO_LOSS_WEIGHT   = 0.5
GRAD_CLIP         = 1.0
ELO_MIN, ELO_MAX  = 1100.0, 2400.0
CHUNKS_PER_GROUP  = 20

# ==================== 8层加深版模型 ====================
class DeepMiniMaia(nn.Module):
    def __init__(self, num_moves=NUM_MOVES):
        super().__init__()
        # 8 层纯卷积，无 BN，训练稳定
        self.conv = nn.Sequential(
            nn.Conv2d(113, 64, 3, padding=1), nn.ReLU(),   # 1
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 2
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 3
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 4
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 5
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 6
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 7 ← 新增
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),    # 8 ← 新增
            nn.AdaptiveAvgPool2d(1)
        )
        self.policy_fc = nn.Linear(64, num_moves)
        self.elo_fc = nn.Linear(64, 1)

    def forward(self, board, elo_plane):
        x = torch.cat([board, elo_plane], dim=1)           # (B, 113, 8, 8)
        f = self.conv(x).flatten(1)
        logits = self.policy_fc(f)
        elo_norm = torch.sigmoid(self.elo_fc(f)).squeeze(-1)
        return logits, elo_norm

# ==================== 分组数据加载器 ====================
def load_chunk_group(file_list):
    planes_list, elos_list, indices_list = [], [], []
    for f in file_list:
        data = np.load(f)
        planes_list.append(data['planes'])
        elos_list.append(data['elos'])
        indices_list.append(data['indices'])
    planes = np.concatenate(planes_list, axis=0)
    elos   = np.concatenate(elos_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)
    elos_n = (elos - ELO_MIN) / (ELO_MAX - ELO_MIN)
    return TensorDataset(
        torch.from_numpy(planes).float(),
        torch.tensor(elos_n, dtype=torch.float32),
        torch.tensor(indices, dtype=torch.long)
    )

# ==================== 训练准备 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model = DeepMiniMaia().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion_policy = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
criterion_elo    = nn.MSELoss()

all_chunk_files = sorted(glob.glob(os.path.join("data/cache/chunks", "chunk_*.npz")))
if not all_chunk_files:
    raise FileNotFoundError("未找到数据块，请先运行 prepare_data.py")
print(f"共发现 {len(all_chunk_files)} 个数据块")

# ==================== 训练循环 ====================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = total_p = total_e = 0.0
    steps = 0
    np.random.shuffle(all_chunk_files)
    num_groups = math.ceil(len(all_chunk_files) / CHUNKS_PER_GROUP)

    for group_idx in range(num_groups):
        start = group_idx * CHUNKS_PER_GROUP
        end = start + CHUNKS_PER_GROUP
        group_files = all_chunk_files[start:end]

        print(f"Epoch {epoch}, 加载第 {group_idx+1}/{num_groups} 组 ({len(group_files)} 个块)...")
        group_dataset = load_chunk_group(group_files)
        group_loader = DataLoader(group_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)

        pbar = tqdm(group_loader, desc=f"Epoch {epoch} Group {group_idx+1}/{num_groups}")
        for boards, elos_n, move_indices in pbar:
            boards = boards.to(device)
            elos_n = elos_n.to(device)
            move_indices = move_indices.to(device)

            elo_plane = (elos_n * 2.0 - 1.0).view(-1, 1, 1, 1).expand(-1, 1, 8, 8)

            logits, elo_pred = model(boards, elo_plane)
            logits = torch.clamp(logits, -50, 50)
            loss_p = criterion_policy(logits, move_indices)
            loss_e = criterion_elo(elo_pred, elos_n)
            loss = loss_p + ELO_LOSS_WEIGHT * loss_e

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_p += loss_p.item()
            total_e += loss_e.item()
            steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.3f}"})

        del group_dataset, group_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    scheduler.step()
    avg_loss = total_loss / steps
    avg_p = total_p / steps
    avg_e = total_e / steps
    print(f"Epoch {epoch} 完成 | Loss: {avg_loss:.4f} | Policy: {avg_p:.4f} | Elo: {avg_e:.4f}")

# ==================== 导出 ONNX ====================
os.makedirs("models", exist_ok=True)
model.eval()

class PolicyExport(nn.Module):
    def forward(self, board, elo_plane):
        logits, _ = model(board, elo_plane)
        return logits

class EloExport(nn.Module):
    def forward(self, board):
        dummy_elo = torch.zeros((board.size(0), 1, 8, 8), device=board.device)
        _, elo_norm = model(board, dummy_elo)
        return elo_norm * (ELO_MAX - ELO_MIN) + ELO_MIN

dummy_board = torch.randn(1, 112, 8, 8).to(device)
dummy_elo_plane = torch.randn(1, 1, 8, 8).to(device)

torch.onnx.export(
    PolicyExport(), (dummy_board, dummy_elo_plane),
    "models/deep_policy.onnx",
    input_names=['board', 'elo_plane'],
    output_names=['policy_logits'],
    opset_version=18,
    dynamic_axes={'board': {0: 'batch'}, 'elo_plane': {0: 'batch'}, 'policy_logits': {0: 'batch'}}
)

torch.onnx.export(
    EloExport(), dummy_board,
    "models/deep_elo.onnx",
    input_names=['board'],
    output_names=['elo'],
    opset_version=18,
    dynamic_axes={'board': {0: 'batch'}, 'elo': {0: 'batch'}}
)

print("8层DeepMiniMaia训练完成！模型已保存为 models/deep_policy.onnx 和 models/deep_elo.onnx")