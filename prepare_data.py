import chess
import chess.pgn
import numpy as np
import os
import gzip
import shutil
import zstandard as zstd
import glob
from collections import defaultdict

# ---------------- 走法映射 & 112平面（保持不变）----------------
def build_move_indices():
    moves = []
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq: continue
            move = chess.Move(from_sq, to_sq)
            if move.promotion is None:
                moves.append(move.uci())
    for from_sq in range(64):
        rank = chess.square_rank(from_sq)
        if rank not in (6, 1): continue
        to_rank = 7 if rank == 6 else 0
        from_file = chess.square_file(from_sq)
        for df in [-1, 0, 1]:
            to_file = from_file + df
            if 0 <= to_file <= 7:
                to_sq = chess.square(to_file, to_rank)
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append(chess.Move(from_sq, to_sq, promotion=promo).uci())
    moves.append('e1g1')
    moves.append('e1c1')
    moves.append('e8g8')
    moves.append('e8c8')
    seen = set()
    unique = []
    for m in moves:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    uci_to_idx = {u: i for i, u in enumerate(unique)}
    idx_to_uci = {i: u for u, i in uci_to_idx.items()}
    return uci_to_idx, idx_to_uci

UCI_TO_IDX, _ = build_move_indices()

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = []
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for color in (chess.WHITE, chess.BLACK):
        for ptype in piece_types:
            plane = np.zeros((8, 8), dtype=np.float32)
            for sq in board.pieces(ptype, color):
                plane[7 - chess.square_rank(sq), chess.square_file(sq)] = 1.0
            planes.append(plane)
    turn_plane = np.full((8, 8), float(board.turn == chess.WHITE), dtype=np.float32)
    planes.append(turn_plane)
    planes.append(np.full((8, 8), float(board.has_kingside_castling_rights(chess.WHITE)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.has_queenside_castling_rights(chess.WHITE)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.has_kingside_castling_rights(chess.BLACK)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.has_queenside_castling_rights(chess.BLACK)), dtype=np.float32))
    planes.append(np.full((8, 8), float(board.is_stalemate()), dtype=np.float32))
    while len(planes) < 112:
        planes.append(np.zeros((8, 8), dtype=np.float32))
    return np.stack(planes[:112], axis=0).astype(np.float32)

# ---------------- 分块写入硬盘 ----------------
def clean_and_parse_pgn(pgn_path, output_dir, chunk_size=10000, max_per_side=7, **kwargs):
    """
    只提取残局样本（双方棋子数均 <= max_per_side）。
    文件编号自动续接，不会覆盖已有 chunk。
    """
    os.makedirs(output_dir, exist_ok=True)
    min_elo = kwargs.get('min_elo', 1100)
    max_elo = kwargs.get('max_elo', 2400)
    min_moves = kwargs.get('min_moves', 10)
    max_games = kwargs.get('max_games', None)

    # 自动续接编号
    existing = glob.glob(os.path.join(output_dir, "chunk_*.npz"))
    if existing:
        max_num = max(int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing)
        chunk_idx = max_num + 1
    else:
        chunk_idx = 1

    samples_batch = []
    game_count = 0
    filtered_reasons = defaultdict(int)

    def save_chunk():
        nonlocal chunk_idx, samples_batch
        if not samples_batch:
            return
        planes_arr = np.array([s[0] for s in samples_batch], dtype=np.float32)
        elos_arr = np.array([s[1] for s in samples_batch], dtype=np.float32)
        indices_arr = np.array([s[2] for s in samples_batch], dtype=np.int64)
        path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(path, planes=planes_arr, elos=elos_arr, indices=indices_arr)
        print(f"  保存 {len(samples_batch)} 条样本 -> {path}")
        chunk_idx += 1
        samples_batch.clear()

    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1
            if max_games is not None and game_count > max_games:
                break

            headers = game.headers
            if headers.get("Variant", "Standard") != "Standard":
                filtered_reasons['Variant'] += 1
                continue
            if "Rated" not in headers.get("Event", ""):
                filtered_reasons['Unrated'] += 1
                continue
            if headers.get("Result") not in ["1-0", "0-1", "1/2-1/2"]:
                filtered_reasons['Result'] += 1
                continue
            if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
                filtered_reasons['BOT'] += 1
                continue
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                filtered_reasons['EloParse'] += 1
                continue
            if not (min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo):
                filtered_reasons['EloRange'] += 1
                continue

            tc_str = headers.get("TimeControl", "")
            if tc_str:
                try:
                    base_sec = int(tc_str.split('+')[0])
                except:
                    base_sec = 0
                if base_sec < 180 and not kwargs.get('allow_tc_bullet', True):
                    filtered_reasons['TimeControl'] += 1
                    continue
                if 180 <= base_sec < 480 and not kwargs.get('allow_tc_blitz', True):
                    filtered_reasons['TimeControl'] += 1
                    continue
                if 480 <= base_sec < 1500 and not kwargs.get('allow_tc_rapid', True):
                    filtered_reasons['TimeControl'] += 1
                    continue
                if base_sec >= 1500 and not kwargs.get('allow_tc_classical', True):
                    filtered_reasons['TimeControl'] += 1
                    continue

            board = game.board()
            moves_played = 0
            temp_buffer = []
            bad_game = False
            try:
                for move in game.mainline_moves():
                    if board.turn == chess.WHITE:
                        current_elo = white_elo
                    else:
                        current_elo = black_elo
                    uci = move.uci()
                    idx = UCI_TO_IDX.get(uci)
                    if idx is None:
                        bad_game = True
                        break
                    planes = board_to_planes(board)

                    # ----- 残局过滤：双方棋子数均不超过 max_per_side -----
                    white_count = sum(1 for sq, p in board.piece_map().items() if p.color == chess.WHITE)
                    black_count = sum(1 for sq, p in board.piece_map().items() if p.color == chess.BLACK)
                    if white_count <= max_per_side and black_count <= max_per_side:
                        temp_buffer.append((planes, current_elo, idx))

                    board.push(move)
                    moves_played += 1
            except:
                bad_game = True

            if bad_game or moves_played < min_moves:
                filtered_reasons['BadMove/TooShort'] += 1
                continue

            samples_batch.extend(temp_buffer)
            if len(samples_batch) >= chunk_size:
                save_chunk()

    save_chunk()

    print(f"\n文件 {os.path.basename(pgn_path)} 完成：总处理 {game_count} 局")
    for reason, count in sorted(filtered_reasons.items()):
        print(f"    {reason}: {count}")

# ---------------- 主函数 ----------------
def main():
    raw_dir = 'D:/chessgame/data/lichess_raw'
    output_dir = 'D:/chessgame/data/cache/chunks'
    max_games_per_file = None
    max_per_side = 7          # 残局定义：双方各不超过7子

    for fname in os.listdir(raw_dir):
        fpath = os.path.join(raw_dir, fname)
        temp_path = None

        # 处理 .zst 文件 ✅
        if fname.endswith('.pgn.zst'):
            temp_path = fpath[:-4]  # 去掉 .zst
            print(f"解压 {fname} -> {os.path.basename(temp_path)} ...")
            with open(fpath, 'rb') as f_in:
                dctx = zstd.ZstdDecompressor()
                with open(temp_path, 'wb') as f_out:
                    dctx.copy_stream(f_in, f_out)
            fpath = temp_path

        # 处理 .gz 文件
        elif fname.endswith('.pgn.gz'):
            temp_path = fpath[:-3]  # 去掉 .gz
            print(f"解压 {fname} -> {os.path.basename(temp_path)} ...")
            with gzip.open(fpath, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            fpath = temp_path

        # 处理 .pgn 文件
        if fpath.endswith('.pgn'):
            print(f"\n开始处理: {os.path.basename(fpath)}")
            clean_and_parse_pgn(
                fpath,
                output_dir,
                chunk_size=10000,
                min_elo=1100,
                max_elo=2400,
                min_moves=10,
                max_games=max_games_per_file,
                max_per_side=max_per_side
            )

        # 清洗完成后删除临时解压文件
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"已删除临时文件: {os.path.basename(temp_path)}")

    print("\n所有文件处理完毕，训练数据分块存储在:", output_dir)

if __name__ == '__main__':
    main()