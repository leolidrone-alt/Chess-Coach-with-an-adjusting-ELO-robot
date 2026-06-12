#!/usr/bin/env python3
"""
混合蒸馏数据生成器（修复版）
- 旧模型生成残局 → 新模型自对弈 → Stockfish 打分
- 探索率、双标签
- 在线训练自动释放/重载模型
"""

import chess, chess.engine, numpy as np, os, sys, time, glob, random, subprocess
from tqdm import tqdm
import onnxruntime as ort

# ========== 配置 ==========
OLD_POLICY_PATH = "models/deep_policy.onnx"
OLD_ELO_PATH    = "models/deep_elo.onnx"
NEW_POLICY_PATH = "models/cond_maia_policy.onnx"
NEW_ELO_PATH    = "models/cond_maia_elo.onnx"
STOCKFISH_PATH  = "stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR      = "data/cache/value_chunks"
NUM_GAMES       = 500
SAMPLES_PER_GAME= 300
TEMPERATURE     = 0.8
SF_DEPTH        = 12
SF_TIME         = 0.2
EPSILON         = 0.15
RECORD_ALL      = False
RESIDUAL_THRESHOLD = 14
TRAIN_AFTER_CHUNK = True
TRAIN_EPOCHS    = 2
TRAIN_LR        = 5e-5
TRAIN_BATCH     = 256
TRAIN_PATIENCE  = 3
TRAIN_SCRIPT    = "train_sft.py"

os.makedirs(OUTPUT_DIR, exist_ok=True)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from kits.mini_maia_ai import MiniMaiaAI
from prepare_data import board_to_planes, UCI_TO_IDX

# 加载模型
old_ai = MiniMaiaAI(policy_path=OLD_POLICY_PATH, elo_path=OLD_ELO_PATH)
new_ai = MiniMaiaAI(policy_path=NEW_POLICY_PATH, elo_path=NEW_ELO_PATH)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# ========== 辅助函数 ==========
def get_stockfish_value(board):
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH, time=SF_TIME))
        score = info["score"].white()
        if score.is_mate(): return 0.0 if score.mate() < 0 else 1.0
        cp = score.score()
        return (1.0 / (1.0 + np.exp(-cp / 400.0))) if board.turn == chess.WHITE else (1.0 - 1.0 / (1.0 + np.exp(-cp / 400.0)))
    except: return 0.5

def choose_move_with_exploration(board, model, epsilon=EPSILON):
    if random.random() < epsilon:
        probs = model.get_move_probs(board, target_elo=1500, temperature=TEMPERATURE)
        moves = list(probs.keys())
        if not moves: return None
        return np.random.choice(moves, p=np.array([probs[m] for m in moves]))
    else:
        try:
            result = engine.play(board, chess.engine.Limit(time=0.02))
            if result.move and result.move in board.legal_moves: return result.move
        except: pass
        probs = model.get_move_probs(board, target_elo=1500, temperature=0.3)
        moves = list(probs.keys())
        if not moves: return None
        return np.random.choice(moves, p=np.array([probs[m] for m in moves]))

def result_to_value(result, color):
    if result == "1-0": return 1.0 if color == chess.WHITE else 0.0
    elif result == "0-1": return 0.0 if color == chess.WHITE else 1.0
    return 0.5

def train_value_model():
    global new_ai, old_ai
    print("  >> 释放模型句柄...")
    new_ai.policy_sess = None
    new_ai.elo_sess = None
    old_ai.policy_sess = None
    old_ai.elo_sess = None

    cmd = [sys.executable, TRAIN_SCRIPT, "--use_value_chunks", "--lr", str(TRAIN_LR),
           "--epochs", str(25+TRAIN_EPOCHS), "--batch", str(TRAIN_BATCH),
           "--patience", str(TRAIN_PATIENCE), "--warmup", "200"]
    ret = subprocess.run(cmd, check=False)

    print("  >> 重新加载模型...")
    new_ai.policy_sess = ort.InferenceSession(os.path.join(base_dir, NEW_POLICY_PATH), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    new_ai.elo_sess = ort.InferenceSession(os.path.join(base_dir, NEW_ELO_PATH), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    old_ai.policy_sess = ort.InferenceSession(os.path.join(base_dir, OLD_POLICY_PATH), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    old_ai.elo_sess = ort.InferenceSession(os.path.join(base_dir, OLD_ELO_PATH), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    if ret.returncode != 0: print("  ⚠️ 增量训练退出码非零。")
    else: print("  ✅ 增量训练完成。")

# ========== 主循环 ==========
all_planes, all_elos, all_indices = [], [], []
all_instant_values, all_result_values = [], []
total_samples = 0
chunk_idx = 1
existing = glob.glob(os.path.join(OUTPUT_DIR, "mixed_*.npz"))
if existing:
    max_num = max(int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing)
    chunk_idx = max_num + 1

for game_i in tqdm(range(NUM_GAMES), desc="对局进度"):
    board = chess.Board()
    # 旧模型走棋
    while not board.is_game_over() and board.fullmove_number < 200:
        probs = old_ai.get_move_probs(board, target_elo=1500, temperature=1.2)
        moves = list(probs.keys())
        if not moves: break
        board.push(np.random.choice(moves, p=np.array([probs[m] for m in moves])))
        white_pieces = sum(1 for sq, p in board.piece_map().items() if p.color == chess.WHITE)
        black_pieces = sum(1 for sq, p in board.piece_map().items() if p.color == chess.BLACK)
        if white_pieces <= 7 and black_pieces <= 7 and (white_pieces + black_pieces) <= 14: break
    if board.fullmove_number >= 200 and len(board.piece_map()) > 14: continue

    # 新模型走棋 + 打分
    move_history, game_samples = [], 0
    while not board.is_game_over() and game_samples < SAMPLES_PER_GAME:
        piece_count = len(board.piece_map())
        record = RECORD_ALL or (piece_count <= RESIDUAL_THRESHOLD)
        if record:
            instant_value = get_stockfish_value(board)
            move = choose_move_with_exploration(board, new_ai)
            if move is None: break
            uci = move.uci()
            idx = UCI_TO_IDX.get(uci)
            if idx is not None:
                planes = board_to_planes(board)
                all_planes.append(planes); all_elos.append(1500.0); all_indices.append(idx)
                all_instant_values.append(instant_value)
                game_samples += 1; total_samples += 1
                move_history.append((board.copy(), move))
            board.push(move)
        else:
            move = choose_move_with_exploration(board, new_ai)
            if move is None: break
            board.push(move)

    result = board.result() if board.is_game_over() else "1/2-1/2"
    for rec_board, _ in move_history:
        all_result_values.append(result_to_value(result, rec_board.turn))

    if (game_i + 1) % 50 == 0 and all_planes:
        planes_arr = np.array(all_planes, dtype=np.float32)
        elos_arr = np.array(all_elos, dtype=np.float32)
        idxs_arr = np.array(all_indices, dtype=np.int64)
        inst_arr = np.array(all_instant_values, dtype=np.float32)
        res_arr = np.array(all_result_values, dtype=np.float32)
        fname = os.path.join(OUTPUT_DIR, f"mixed_{chunk_idx:06d}.npz")
        np.savez_compressed(fname, planes=planes_arr, elos=elos_arr, indices=idxs_arr,
                            values=res_arr, instant_values=inst_arr, result_values=res_arr)
        print(f"  保存 {len(all_planes)} 条样本 -> {fname}")
        chunk_idx += 1
        all_planes, all_elos, all_indices = [], [], []
        all_instant_values, all_result_values = [], []
        if TRAIN_AFTER_CHUNK: train_value_model()

if all_planes:
    planes_arr = np.array(all_planes, dtype=np.float32)
    elos_arr = np.array(all_elos, dtype=np.float32)
    idxs_arr = np.array(all_indices, dtype=np.int64)
    inst_arr = np.array(all_instant_values, dtype=np.float32)
    res_arr = np.array(all_result_values, dtype=np.float32)
    fname = os.path.join(OUTPUT_DIR, f"mixed_{chunk_idx:06d}.npz")
    np.savez_compressed(fname, planes=planes_arr, elos=elos_arr, indices=idxs_arr,
                        values=res_arr, instant_values=inst_arr, result_values=res_arr)
    print(f"  最终保存 {len(all_planes)} 条样本 -> {fname}")
    if TRAIN_AFTER_CHUNK: train_value_model()

engine.quit()
print(f"\n总共生成 {total_samples} 条混合蒸馏样本，价值头训练结束。")