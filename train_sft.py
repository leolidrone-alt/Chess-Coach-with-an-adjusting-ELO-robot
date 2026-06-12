#!/usr/bin/env python3
"""
价值网络训练脚本（含策略头+Elo头+价值头）
支持跳过损坏文件、混合新旧数据、从头训练或续训。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, sys, time, gc, glob, math, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data import build_move_indices

UCI_TO_IDX, IDX_TO_UCI = build_move_indices()
NUM_MOVES = len(UCI_TO_IDX)
print(f"策略头输出维度: {NUM_MOVES}")

# ==================== 可配置超参数 ====================
BATCH_SIZE        = 256
EPOCHS            = 500
LR_INIT           = 3e-4          # 从头训练推荐值
WEIGHT_DECAY      = 1e-4
LABEL_SMOOTHING   = 0.05
ELO_LOSS_WEIGHT   = 0.5
VALUE_LOSS_WEIGHT = 1.5
GRAD_CLIP         = 1.0
VAL_SPLIT         = 0.1
WARMUP_STEPS      = 1000
EARLY_STOP_PATIENCE = 3

CHUNKS_PER_GROUP  = 20
ELO_MIN, ELO_MAX  = 1100.0, 2400.0

USE_AMP           = False
USE_COMPILE       = False
NUM_WORKERS       = 0
PIN_MEMORY        = False

CHECKPOINT_DIR = "checkpoints"
MODEL_ONNX_DIR = "models"
PLOT_DIR       = "plots"
CHUNK_DIR      = "data/cache/chunks"
VALUE_CHUNK_DIR = "data/cache/value_chunks"
LOG_DIR        = "runs/value_robust"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_ONNX_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,       exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ==================== 模型定义 ====================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ConditionalMaia(nn.Module):
    def __init__(self, num_moves, channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(112, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResBlock(channels), ResBlock(channels),
            ResBlock(channels), ResBlock(channels),
            ResBlock(channels), ResBlock(channels),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(channels, num_moves)

        self.elo_fc = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        self.value_fc = nn.Sequential(
            nn.Linear(channels, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x112, elo_norm_plane):
        shared = self.encoder(x112)
        shared_pool = self.global_pool(shared).flatten(1)

        pol_in = torch.cat([shared, elo_norm_plane], dim=1)
        pol_feat = self.global_pool(self.policy_conv(pol_in)).flatten(1)
        policy_logits = self.policy_fc(pol_feat)

        elo_pred_norm = self.elo_fc(shared_pool).squeeze(-1)
        value = self.value_fc(shared_pool).squeeze(-1)

        return policy_logits, elo_pred_norm, value


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ==================== 数据加载（跳过损坏文件） ====================
def load_chunk_group(file_list):
    planes_list, elos_list, indices_list, values_list = [], [], [], []
    for f in file_list:
        try:
            data = np.load(f)
            planes_list.append(data['planes'])
            elos_list.append(data['elos'])
            indices_list.append(data['indices'])
            if 'values' in data:
                values_list.append(data['values'])
            else:
                values_list.append(np.zeros(len(data['planes']), dtype=np.float32))
        except Exception as e:
            print(f"  ⚠️ 跳过损坏文件: {f} - {e}")
            continue
    if not planes_list:
        return None
    planes = np.concatenate(planes_list, axis=0)
    elos   = np.concatenate(elos_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)
    values = np.concatenate(values_list, axis=0)
    elos_n = (elos - ELO_MIN) / (ELO_MAX - ELO_MIN)
    return TensorDataset(
        torch.from_numpy(planes).float(),
        torch.tensor(elos_n, dtype=torch.float32),
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(values, dtype=torch.float32)
    )


def find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "cond_maia_epoch*.pth"))
    if not files:
        return None, 0
    epochs = [int(os.path.basename(f).split('epoch')[1].split('.')[0]) for f in files]
    latest_epoch = max(epochs)
    return os.path.join(CHECKPOINT_DIR, f"cond_maia_epoch{latest_epoch}.pth"), latest_epoch


# ==================== 主程序 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR_INIT)
    parser.add_argument('--warmup', type=int, default=WARMUP_STEPS)
    parser.add_argument('--patience', type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument('--use_value_chunks', action='store_true',
                        help='加载 value_chunks 目录下的数据')
    args = parser.parse_args()
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    LR_INIT = args.lr
    WARMUP_STEPS = args.warmup
    EARLY_STOP_PATIENCE = args.patience

    writer = SummaryWriter(LOG_DIR)

    # 收集所有数据文件
    all_chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.npz")))
    if args.use_value_chunks:
        value_files = sorted(
            glob.glob(os.path.join(VALUE_CHUNK_DIR, "value_chunk_*.npz")) +
            glob.glob(os.path.join(VALUE_CHUNK_DIR, "mixed_*.npz"))
        )
        all_chunk_files.extend(value_files)
        print(f"加载 {len(all_chunk_files) - len(value_files)} 个普通块 + {len(value_files)} 个价值块")
    else:
        print(f"加载 {len(all_chunk_files)} 个普通块（未包含价值块）")

    if not all_chunk_files:
        raise FileNotFoundError("没有找到任何数据块")

    num_chunks = len(all_chunk_files)
    val_chunks = int(num_chunks * VAL_SPLIT)
    train_files = all_chunk_files[:-val_chunks]
    val_files = all_chunk_files[-val_chunks:]
    print(f"训练块: {len(train_files)}，验证块: {len(val_files)}")

    model = ConditionalMaia(NUM_MOVES, channels=64)
    model.apply(init_weights)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion_policy = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_elo    = nn.MSELoss()
    criterion_value  = nn.MSELoss()

    start_epoch = 1
    best_val_loss = float('inf')
    early_stop_counter = 0
    ckpt_path, ckpt_epoch = find_latest_checkpoint()
    if ckpt_path:
        print(f"加载检查点: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state['model'], strict=False)
        if missing: print(f"  缺失层（随机初始化）: {missing}")
        if unexpected: print(f"  多余层: {unexpected}")
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = ckpt_epoch + 1
        best_val_loss = state.get('best_val_loss', float('inf'))
        early_stop_counter = state.get('early_stop_counter', 0)
        print(f"从 epoch {start_epoch} 继续训练，当前最佳验证 Loss: {best_val_loss:.4f}")

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_p': [], 'val_p': [], 'train_e': [], 'val_e': [],
        'train_v': [], 'val_v': []
    }

    print(f"\n开始训练 (batch={BATCH_SIZE}, lr={LR_INIT}, warmup={WARMUP_STEPS})")
    total_start = time.time()
    global_step = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = total_p = total_e = total_v = 0.0
        steps = 0

        np.random.shuffle(train_files)
        num_train_groups = math.ceil(len(train_files) / CHUNKS_PER_GROUP)

        for group_idx in range(num_train_groups):
            start = group_idx * CHUNKS_PER_GROUP
            end = start + CHUNKS_PER_GROUP
            group_files = train_files[start:end]

            train_dataset = load_chunk_group(group_files)
            if train_dataset is None:
                print(f"  训练组 {group_idx+1} 无有效数据，跳过")
                continue
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, pin_memory=False)

            pbar = tqdm(train_loader, desc=f"Train Group {group_idx+1}/{num_train_groups}", leave=False)
            for batch_planes, batch_elos_n, batch_indices, batch_values in pbar:
                batch_planes = batch_planes.to(device)
                batch_elos_n = batch_elos_n.to(device)
                batch_indices = batch_indices.to(device)
                batch_values = batch_values.to(device)

                elo_plane = (batch_elos_n * 2.0 - 1.0).view(-1, 1, 1, 1).expand(-1, 1, 8, 8)

                if global_step < WARMUP_STEPS:
                    lr = LR_INIT * (global_step + 1) / WARMUP_STEPS
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                policy_logits, elo_pred_norm, value_pred = model(batch_planes, elo_plane)
                policy_logits = torch.clamp(policy_logits, -50, 50)
                loss_p = criterion_policy(policy_logits, batch_indices)
                loss_e = criterion_elo(elo_pred_norm, batch_elos_n)
                loss_v = criterion_value(value_pred, batch_values)
                loss = loss_p + ELO_LOSS_WEIGHT * loss_e + VALUE_LOSS_WEIGHT * loss_v

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item()
                total_p += loss_p.item()
                total_e += loss_e.item()
                total_v += loss_v.item()
                steps += 1
                global_step += 1

                pbar.set_postfix({'loss': f"{loss.item():.3f}", 'v': f"{loss_v.item():.3f}"})

            del train_dataset, train_loader
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / max(1, steps)
        avg_train_p = total_p / max(1, steps)
        avg_train_e = total_e / max(1, steps)
        avg_train_v = total_v / max(1, steps)

        # 验证
        model.eval()
        val_loss = val_p = val_e = val_v = 0.0
        val_steps = 0
        np.random.shuffle(val_files)
        num_val_groups = math.ceil(len(val_files) / CHUNKS_PER_GROUP)

        with torch.no_grad():
            for group_idx in range(num_val_groups):
                start = group_idx * CHUNKS_PER_GROUP
                end = start + CHUNKS_PER_GROUP
                group_files = val_files[start:end]

                val_dataset = load_chunk_group(group_files)
                if val_dataset is None:
                    continue
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                for batch_planes, batch_elos_n, batch_indices, batch_values in val_loader:
                    batch_planes = batch_planes.to(device)
                    batch_elos_n = batch_elos_n.to(device)
                    batch_indices = batch_indices.to(device)
                    batch_values = batch_values.to(device)

                    elo_plane = (batch_elos_n * 2.0 - 1.0).view(-1, 1, 1, 1).expand(-1, 1, 8, 8)
                    policy_logits, elo_pred_norm, value_pred = model(batch_planes, elo_plane)
                    policy_logits = torch.clamp(policy_logits, -50, 50)
                    loss_p = criterion_policy(policy_logits, batch_indices)
                    loss_e = criterion_elo(elo_pred_norm, batch_elos_n)
                    loss_v = criterion_value(value_pred, batch_values)
                    loss = loss_p + ELO_LOSS_WEIGHT * loss_e + VALUE_LOSS_WEIGHT * loss_v

                    val_loss += loss.item()
                    val_p += loss_p.item()
                    val_e += loss_e.item()
                    val_v += loss_v.item()
                    val_steps += 1

                del val_dataset, val_loader
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_p = val_p / max(1, val_steps)
        avg_val_e = val_e / max(1, val_steps)
        avg_val_v = val_v / max(1, val_steps)

        scheduler.step()

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('PolicyLoss/Train', avg_train_p, epoch)
        writer.add_scalar('PolicyLoss/Val', avg_val_p, epoch)
        writer.add_scalar('EloLoss/Train', avg_train_e, epoch)
        writer.add_scalar('EloLoss/Val', avg_val_e, epoch)
        writer.add_scalar('ValueLoss/Train', avg_train_v, epoch)
        writer.add_scalar('ValueLoss/Val', avg_val_v, epoch)

        print(f"\nEpoch {epoch}/{EPOCHS} | "
              f"Train: {avg_train_loss:.3f} (P:{avg_train_p:.3f} E:{avg_train_e:.4f} V:{avg_train_v:.4f}) | "
              f"Val: {avg_val_loss:.3f} (P:{avg_val_p:.3f} E:{avg_val_e:.4f} V:{avg_val_v:.4f})")

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_p'].append(avg_train_p)
        history['val_p'].append(avg_val_p)
        history['train_e'].append(avg_train_e)
        history['val_e'].append(avg_val_e)
        history['train_v'].append(avg_train_v)
        history['val_v'].append(avg_val_v)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stop_counter
            }, os.path.join(CHECKPOINT_DIR, "cond_maia_best.pth"))
            print(f"  ✅ 最佳模型已保存 (val_loss={avg_val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"验证损失连续 {EARLY_STOP_PATIENCE} 轮未下降，提前停止训练。")
                break

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, os.path.join(CHECKPOINT_DIR, f"cond_maia_epoch{epoch}.pth"))

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\n训练完成，总耗时: {total_time/60:.1f} 分钟")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0,0].plot(history['epoch'], history['train_loss'], 'b-o', label='Train')
    axes[0,0].plot(history['epoch'], history['val_loss'], 'r-s', label='Val')
    axes[0,0].set_title('Total Loss'); axes[0,0].legend(); axes[0,0].grid(True)
    axes[0,1].plot(history['epoch'], history['train_p'], 'g-o', label='Train Policy')
    axes[0,1].plot(history['epoch'], history['val_p'], 'y-s', label='Val Policy')
    axes[0,1].set_title('Policy Loss'); axes[0,1].legend(); axes[0,1].grid(True)
    axes[1,0].plot(history['epoch'], history['train_e'], 'm-o', label='Train Elo')
    axes[1,0].plot(history['epoch'], history['val_e'], 'c-s', label='Val Elo')
    axes[1,0].set_title('Elo Loss'); axes[1,0].legend(); axes[1,0].grid(True)
    axes[1,1].plot(history['epoch'], history['train_v'], 'purple', label='Train Value')
    axes[1,1].plot(history['epoch'], history['val_v'], 'orange', label='Val Value')
    axes[1,1].set_title('Value Loss'); axes[1,1].legend(); axes[1,1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'loss_curve_value.png'), dpi=150)
    plt.close()
    print(f"损失曲线已保存至 {PLOT_DIR}/loss_curve_value.png")

    # 导出 ONNX
    print("导出 ONNX 模型...")
    best_path = os.path.join(CHECKPOINT_DIR, "cond_maia_best.pth")
    if os.path.exists(best_path):
        print(f"  加载最佳模型: {best_path}")
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"  验证损失: {state.get('best_val_loss', 'N/A')}")
    else:
        print("  ⚠️ 未找到最佳检查点，使用当前权重导出")

    model.eval()

    class PolicyONNX(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
        def forward(self, board, elo_plane):
            logits, _, _ = self.core(board, elo_plane)
            return logits

    class EloONNX(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
        def forward(self, board):
            b = board.size(0)
            dummy_elo = torch.zeros((b, 1, 8, 8), device=board.device)
            _, elo_norm, _ = self.core(board, dummy_elo)
            return elo_norm * (ELO_MAX - ELO_MIN) + ELO_MIN

    class ValueONNX(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
        def forward(self, board):
            b = board.size(0)
            dummy_elo = torch.zeros((b, 1, 8, 8), device=board.device)
            _, _, value = self.core(board, dummy_elo)
            return value

    dummy_board = torch.randn(1, 112, 8, 8).to(device)
    dummy_elo   = torch.randn(1, 1, 8, 8).to(device)

    torch.onnx.export(PolicyONNX(model), (dummy_board, dummy_elo),
                      os.path.join(MODEL_ONNX_DIR, "cond_maia_policy.onnx"),
                      input_names=['board', 'elo_plane'], output_names=['policy_logits'],
                      opset_version=18, dynamic_axes={'board': {0: 'batch'}, 'elo_plane': {0: 'batch'}, 'policy_logits': {0: 'batch'}})

    torch.onnx.export(EloONNX(model), dummy_board,
                      os.path.join(MODEL_ONNX_DIR, "cond_maia_elo.onnx"),
                      input_names=['board'], output_names=['elo'],
                      opset_version=18, dynamic_axes={'board': {0: 'batch'}, 'elo': {0: 'batch'}})

    torch.onnx.export(ValueONNX(model), dummy_board,
                      os.path.join(MODEL_ONNX_DIR, "cond_maia_value.onnx"),
                      input_names=['board'], output_names=['value'],
                      opset_version=18, dynamic_axes={'board': {0: 'batch'}, 'value': {0: 'batch'}})

    print("✅ ONNX 导出完成！")
    writer.close()