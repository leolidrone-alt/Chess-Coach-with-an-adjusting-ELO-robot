"""测试 cond_maia_value.onnx 价值头表现"""
import chess, chess.engine, numpy as np, onnxruntime as ort, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
from prepare_data import board_to_planes

value_sess = ort.InferenceSession('models/cond_maia_value.onnx')
engine = chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-windows-x86-64-avx2.exe')

def board_to_input(board):
    planes = board_to_planes(board).reshape(1, 112, 8, 8).astype(np.float32)
    return planes

def get_value(board):
    planes = board_to_input(board)
    value = value_sess.run(None, {'board': planes})[0][0]
    return float(value)

def get_sf_value(board):
    info = engine.analyse(board, chess.engine.Limit(depth=15, time=0.5))
    score = info['score'].white()
    if score.is_mate():
        win_prob = 0.0 if score.mate() < 0 else 1.0
        cp = 10000 if score.mate() > 0 else -10000
    else:
        cp = score.score()
        win_prob = 1.0 / (1.0 + np.exp(-cp / 400.0))
    if board.turn == chess.WHITE:
        cur_win_prob = win_prob
    else:
        cur_win_prob = 1.0 - win_prob
    return cur_win_prob, cp

# 测试局面
openings = [
    ('开局 1.e4 后黑走', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'),
    ('开局 西西里 白走', 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2'),
    ('开局 法兰西 白走', 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2'),
    ('开局 1.Nf3 后黑走', 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1'),
    ('开局 意大利 白走', 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3'),
]
middlegames = [
    ('中局 均衡 白走', 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7'),
    ('中局 白稍优', 'r1bq1rk1/pp3ppp/2np4/2b1p1N1/2B1P3/2NP4/PPP2PPP/R1BQ1RK1 w - - 0 10'),
    ('中局 复杂', 'r4rk1/1b1n1ppp/1qp1p3/1p2P3/p2P4/P2B1N2/1P3PPP/R2QR1K1 w - - 0 18'),
    ('中局 白方攻王', 'r3r1k1/pp1q1pp1/2pb1n1p/4p3/2PP4/1PN1PN2/P1Q1BPPP/R1B2RK1 w - - 0 13'),
]
endgames = [
    ('KQvK 白优', '8/8/8/8/3k4/8/8/3QK3 w - - 0 1'),
    ('KRvK 白优', '8/8/8/8/8/4k3/8/4K2R w - - 0 1'),
    ('单马和棋', '8/8/8/4k3/8/4K3/8/4N3 w - - 0 1'),
    ('兵残局 均势', '8/8/8/8/1p6/8/1P6/1K6 w - - 0 1'),
    ('KQvK 白优2', '8/8/8/5k2/8/5K2/8/5Q2 w - - 0 1'),
    ('兵残局 白优', '8/8/8/8/8/3k4/3P4/3K4 w - - 0 1'),
    ('逼和局面', '7K/8/8/8/8/8/8/7k w - - 0 1'),
    ('KRvK 杀棋', '8/8/8/8/8/8/3k4/3KR3 w - - 0 1'),
    ('KPvK 白兵升变', '8/8/8/8/8/2k5/2P5/2K5 w - - 0 1'),
]

print('=' * 80)
print('  cond_maia_value.onnx 价值头测试')
print('=' * 80)

all_results = []
for group_name, positions in [('开局', openings), ('中局', middlegames), ('残局', endgames)]:
    print(f'\n--- {group_name} ---')
    model_vals = []
    sf_vals = []
    for desc, fen in positions:
        board = chess.Board(fen)
        mv = get_value(board)
        sv, cp = get_sf_value(board)
        model_vals.append(mv)
        sf_vals.append(sv)
        turn_str = '白方' if board.turn == chess.WHITE else '黑方'
        diff = mv - sv
        status = '✓' if abs(diff) < 0.2 else ('⚠' if abs(diff) < 0.35 else '✗')
        print(f'  {status} {desc:<16s} | 模型={mv:.4f}  SF={sv:.4f}  SF_cp={cp:+5d}  diff={diff:+.4f}  走子方={turn_str}')
        all_results.append((group_name, desc, mv, sv, cp, diff))

    ae = np.abs(np.array(model_vals) - np.array(sf_vals))
    mae = np.mean(ae)
    rmse = np.sqrt(np.mean(ae ** 2))
    print(f'  >> {group_name} 汇总: MAE={mae:.4f}  RMSE={rmse:.4f}')

print('\n' + '=' * 80)
all_ae = np.abs(np.array([r[2] for r in all_results]) - np.array([r[3] for r in all_results]))
print(f'  总体 MAE={np.mean(all_ae):.4f}  RMSE={np.sqrt(np.mean(all_ae**2)):.4f}')
print('=' * 80)

# 额外测试：几步棋的连续变化
print('\n' + '=' * 80)
print('  走棋过程中价值变化测试')
print('=' * 80)
board = chess.Board()
for i, move in enumerate([chess.Move.from_uci('e2e4'), chess.Move.from_uci('e7e5'),
                           chess.Move.from_uci('g1f3'), chess.Move.from_uci('b8c6'),
                           chess.Move.from_uci('f1b5'), chess.Move.from_uci('a7a6')]):
    board.push(move)
    mv = get_value(board)
    sv, cp = get_sf_value(board)
    turn = '白方' if board.turn == chess.WHITE else '黑方'
    diff = mv - sv
    print(f'  第{i+1}步 {str(move):6s} | 模型值={mv:.4f}  SF值={sv:.4f}  SF_cp={cp:+5d}  diff={diff:+.4f}  当前={turn}走棋')

try:
    engine.quit()
except:
    pass
print('\n完成！')
