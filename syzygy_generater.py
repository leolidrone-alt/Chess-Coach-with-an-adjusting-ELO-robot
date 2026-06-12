import chess, chess.syzygy, numpy as np, os, re, itertools
from prepare_data import board_to_planes

TB_PATH = "syzygy"
OUTPUT_DIR = "data/cache/value_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tb = chess.syzygy.open_tablebase(TB_PATH)

def parse_name(fname):
    m = re.match(r'([KQRBNP]+)v([KQRBNP]+)', os.path.splitext(fname)[0])
    return (list(m.group(1)), list(m.group(2))) if m else None

def piece_char(c, color):
    mp = {'K':chess.KING,'Q':chess.QUEEN,'R':chess.ROOK,'B':chess.BISHOP,'N':chess.KNIGHT,'P':chess.PAWN}
    return chess.Piece(mp[c], color)

def enumerate_legal(white_chars, black_chars):
    """返回所有合法的 (board, white_squares, black_squares) """
    white_pieces = [piece_char(c, chess.WHITE) for c in white_chars]
    black_pieces = [piece_char(c, chess.BLACK) for c in black_chars]
    total = len(white_pieces) + len(black_pieces)
    squares = list(chess.SQUARES)
    configs = []
    for pos in itertools.combinations(squares, total):
        # 分配棋子：将棋子列表排列映射到这些位置
        for perm in itertools.permutations(white_pieces + black_pieces, total):
            # 快速判断：兵不能在第一排和第八排
            bad = False
            for sq, p in zip(pos, perm):
                if p.piece_type == chess.PAWN and chess.square_rank(sq) in (0,7):
                    bad = True
                    break
            if bad: continue
            board = chess.Board()
            board.clear_board()
            for sq, p in zip(pos, perm):
                board.set_piece_at(sq, p)
            if board.is_valid():
                # 同时考虑白先和黑先（两种走棋方）
                for turn in (chess.WHITE, chess.BLACK):
                    board.turn = turn
                    if board.is_valid():
                        configs.append(board.copy())
    return configs

def main():
    files = [f for f in os.listdir(TB_PATH) if f.endswith('.rtbw')]
    all_samples = []
    for f in files:
        parsed = parse_name(f)
        if not parsed: continue
        w, b = parsed
        if len(w)+len(b) > 5:
            continue
        print(f"处理 {f} ...")
        try:
            boards = enumerate_legal(w, b)
        except:
            continue
        for bd in boards:
            try:
                wdl = tb.probe_wdl(bd)
            except (KeyError, chess.syzygy.MissingTableError):
                continue
            if wdl == 2:
                label = 1.0 if bd.turn == chess.WHITE else 0.0
            elif wdl == -2:
                label = 0.0 if bd.turn == chess.WHITE else 1.0
            else:
                label = 0.5
            planes = board_to_planes(bd)
            all_samples.append((planes, 1500.0, 0, label))
        print(f"  已收集 {len(all_samples)} 总样本")

    if not all_samples:
        print("未生成样本。"); return
    planes = np.array([s[0] for s in all_samples], dtype=np.float32)
    elos = np.array([s[1] for s in all_samples], dtype=np.float32)
    indices = np.array([s[2] for s in all_samples], dtype=np.int64)
    values = np.array([s[3] for s in all_samples], dtype=np.float32)
    out = os.path.join(OUTPUT_DIR, "value_tablebase.npz")
    np.savez_compressed(out, planes=planes, elos=elos, indices=indices, values=values)
    print(f"完成，总计 {len(all_samples)} 样本 -> {out}")

if __name__ == "__main__":
    main()