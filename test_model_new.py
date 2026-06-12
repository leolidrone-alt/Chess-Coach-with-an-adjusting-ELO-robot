import chess
import chess.engine
import numpy as np
import onnxruntime as ort
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('.')
from prepare_data import board_to_planes

# ------------------------------ 配置 ------------------------------
VALUE_MODEL_PATH = "models/cond_maia_value.onnx"
ELO_MODEL_PATH   = "models/cond_maia_elo.onnx"
POLICY_MODEL_PATH= "models/cond_maia_policy.onnx"
STOCKFISH_PATH   = "stockfish/stockfish-windows-x86-64-avx2.exe"
SF_DEPTH         = 15

# ------------------------------ 加载模型 ------------------------------
value_sess = ort.InferenceSession(VALUE_MODEL_PATH)
elo_sess   = ort.InferenceSession(ELO_MODEL_PATH)
engine     = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# ------------------------------ 辅助函数 ------------------------------
def board_to_input(board):
    """将棋盘转换为 (1,112,8,8) 的 numpy 数组"""
    planes = board_to_planes(board).reshape(1, 112, 8, 8).astype(np.float32)
    return planes

def get_value(board):
    """价值头：返回当前走棋方视角的胜率 (0~1)"""
    planes = board_to_input(board)
    value = value_sess.run(None, {'board': planes})[0][0]
    # 价值头输出白方视角？请确认你的模型：实际上 EloONNX 输出 elo，ValueONNX 输出 value。
    # 根据之前导出代码，ValueONNX 直接返回 value，应该是白方视角的胜率？我们检查：
    # 在 train_value_robust.py 中，ValueONNX 返回 value，没有乘以任何东西。
    # 而训练时 value 标签是从当前走棋方视角计算的（白方走棋时标签是白方最终胜率，黑方走棋时是黑方最终胜率）。
    # 因此模型学到的就是当前走棋方的胜率。所以 value 就是当前走棋方的胜率，无需转换。
    return float(value)

def get_elo(board):
    """Elo头：返回当前走棋方的预估 Elo"""
    planes = board_to_input(board)
    elo = elo_sess.run(None, {'board': planes})[0][0]
    return float(elo)

def get_stockfish_value_and_score(board):
    """返回 (当前走棋方视角胜率, 白方centipawn)"""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH, time=0.5))
        score = info["score"].white()
        if score.is_mate():
            win_prob = 0.0 if score.mate() < 0 else 1.0
            cp = 10000 if score.mate() > 0 else -10000
        else:
            cp = score.score()
            win_prob = 1.0 / (1.0 + np.exp(-cp / 400.0))
        # win_prob 是白方胜率
        if board.turn == chess.WHITE:
            cur_win_prob = win_prob
        else:
            cur_win_prob = 1.0 - win_prob
        return cur_win_prob, cp
    except:
        return 0.5, 0.0

# ------------------------------ 测试局面库 ------------------------------
openings = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2",
    "rnbqkb1r/ppp1pppp/5n2/3p4/2P5/2N2N2/PP1PPPPP/R1BQKB1R b KQkq - 0 5",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
]
middlegames = [
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
    "r1bq1rk1/pp3ppp/2np4/2b1p1N1/2B1P3/2NP4/PPP2PPP/R1BQ1RK1 w - - 0 10",
    "r1b2rk1/pp3ppp/2np1q2/2b1p1N1/2B1P3/2NP4/PPP2PPP/R1BQ1RK1 w - - 0 11",
    "r4rk1/1b1n1ppp/1qp1p3/1p2P3/p2P4/P2B1N2/1P3PPP/R2QR1K1 w - - 0 18",
    "rn3rk1/pbppq1pp/1p2pp2/8/2PP4/2N2N2/PP2BPPP/R2QR1K1 w - - 0 12",
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2PPP3/2N2N2/PP4PP/R1BQKB1R w KQkq - 0 6",
    "r1bq1rk1/ppp2ppp/2np1n2/4p3/2PP4/2N2N2/PP2BPPP/R1BQK2R w KQkq - 0 7",
]
endgames = [
    "8/8/8/8/3k4/8/8/3QK3 w - - 0 1",      # KQvK 白优
    "8/8/8/8/8/4k3/8/4K2R w - - 0 1",      # KRvK
    "8/8/8/4k3/8/4K3/8/4N3 w - - 0 1",    # 单马和棋
    "8/8/8/8/1p6/8/1P6/1K6 w - - 0 1",    # 兵残局均势
    "8/8/8/5k2/8/5K2/8/5Q2 w - - 0 1",    # KQvK 白优
    "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",    # 兵残局白优
    "7K/8/8/8/8/8/8/7k w - - 0 1",        # 逼和
    "8/8/8/8/8/8/3k4/3KR3 w - - 0 1",     # KRvK 不同位置
    "8/8/8/8/3k4/8/2R5/2K5 w - - 0 1",    # KRvK 杀棋接近
    "8/8/8/8/8/2k5/2P5/2K5 w - - 0 1",    # KPvK 白兵升变
]

# ------------------------------ 测试执行 ------------------------------
def test_group(name, fen_list):
    model_values = []
    model_elos = []
    sf_values = []
    sf_cps = []
    for fen in tqdm(fen_list, desc=f"测试 {name}"):
        board = chess.Board(fen)
        # 价值头
        mv = get_value(board)
        model_values.append(mv)
        # Elo头
        melo = get_elo(board)
        model_elos.append(melo)
        # Stockfish
        sv, cp = get_stockfish_value_and_score(board)
        sf_values.append(sv)
        sf_cps.append(cp)
    return np.array(model_values), np.array(model_elos), np.array(sf_values), np.array(sf_cps)

print("收集测试数据...")
mv_o, melo_o, sv_o, cp_o = test_group("开局", openings)
mv_m, melo_m, sv_m, cp_m = test_group("中局", middlegames)
mv_e, melo_e, sv_e, cp_e = test_group("残局", endgames)

# ------------------------------ 价值头误差计算 ------------------------------
def compute_metrics(model_vals, sf_vals):
    ae = np.abs(model_vals - sf_vals)
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean(ae**2))
    return mae, rmse

mae_o, rmse_o = compute_metrics(mv_o, sv_o)
mae_m, rmse_m = compute_metrics(mv_m, sv_m)
mae_e, rmse_e = compute_metrics(mv_e, sv_e)

print("\n========== 价值头评估 ==========")
print(f"开局:   MAE={mae_o:.4f}, RMSE={rmse_o:.4f}")
print(f"中局:   MAE={mae_m:.4f}, RMSE={rmse_m:.4f}")
print(f"残局:   MAE={mae_e:.4f}, RMSE={rmse_e:.4f}")

# ------------------------------ Elo头分析 ------------------------------
# 计算 Elo 与 Stockfish centipawn 的线性相关系数
def correlation(model_elos, sf_cps):
    return np.corrcoef(model_elos, sf_cps)[0,1] if len(model_elos)>1 else 0.0

corr_o = correlation(melo_o, cp_o)
corr_m = correlation(melo_m, cp_m)
corr_e = correlation(melo_e, cp_e)

print("\n========== Elo头评估 (与Stockfish centipawn相关) ==========")
print(f"开局:   相关系数 = {corr_o:.4f}")
print(f"中局:   相关系数 = {corr_m:.4f}")
print(f"残局:   相关系数 = {corr_e:.4f}")
print("(系数越接近1表示 Elo 随局面优劣变化越符合预期)")

# ------------------------------ 可视化 ------------------------------
# 图1：价值头 vs Stockfish 胜率
plt.figure(figsize=(10, 6))
plt.scatter(sv_o, mv_o, color='blue', alpha=0.7, label=f'开局 (MAE={mae_o:.3f})')
plt.scatter(sv_m, mv_m, color='orange', alpha=0.7, label=f'中局 (MAE={mae_m:.3f})')
plt.scatter(sv_e, mv_e, color='red', alpha=0.7, label=f'残局 (MAE={mae_e:.3f})')
plt.plot([0,1], [0,1], 'k--', label='理想线')
plt.xlabel('Stockfish 胜率')
plt.ylabel('模型胜率')
plt.title('价值网络全局评估 vs Stockfish')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/value_global_test.png', dpi=150)
plt.close()

# 图2：Elo头 vs Stockfish centipawn
plt.figure(figsize=(10, 6))
plt.scatter(cp_o, melo_o, color='blue', alpha=0.7, label=f'开局 (corr={corr_o:.3f})')
plt.scatter(cp_m, melo_m, color='orange', alpha=0.7, label=f'中局 (corr={corr_m:.3f})')
plt.scatter(cp_e, melo_e, color='red', alpha=0.7, label=f'残局 (corr={corr_e:.3f})')
plt.xlabel('Stockfish centipawn (白方优势为正)')
plt.ylabel('模型 Elo 预测')
plt.title('Elo 头评估 vs Stockfish 局面评分')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/elo_global_test.png', dpi=150)
plt.close()

print("\n图表已保存至 plots/value_global_test.png 和 plots/elo_global_test.png")
engine.quit()