import chess, chess.engine, chess.pgn, numpy as np, os, gzip, shutil, zstandard as zstd, glob, re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- 走法映射 & 112平面（与之前完全相同）----------------
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

def evaluate_board(board, engine):
    """返回当前走棋方的胜率 (0~1)"""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=10, time=0.1))
        score = info["score"].white()
        if score.is_mate():
            return 0.0 if score.mate() < 0 else 1.0
        cp = score.score()
        win_prob = 1.0 / (1.0 + np.exp(-cp / 400.0))
        if board.turn == chess.WHITE:
            return win_prob
        else:
            return 1.0 - win_prob
    except:
        return 0.5

def get_next_chunk_idx(output_dir):
    """获取下一个可用的chunk编号"""
    existing = glob.glob(os.path.join(output_dir, "value_chunk_*.npz"))
    if not existing:
        return 1
    max_num = max(int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing)
    return max_num + 1

def process_file(args):
    fpath, output_dir, chunk_size, max_games, min_elo, max_elo, max_per_side, sf_path = args
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    os.makedirs(output_dir, exist_ok=True)

    samples_batch = []
    game_count = 0
    filtered_reasons = defaultdict(int)

    def save_chunk():
        nonlocal samples_batch
        if not samples_batch:
            return
        idx = get_next_chunk_idx(output_dir)  # 每次都重新扫描
        planes_arr = np.array([s[0] for s in samples_batch], dtype=np.float32)
        elos_arr = np.array([s[1] for s in samples_batch], dtype=np.float32)
        indices_arr = np.array([s[2] for s in samples_batch], dtype=np.int64)
        values_arr = np.array([s[3] for s in samples_batch], dtype=np.float32)
        path = os.path.join(output_dir, f"value_chunk_{idx:06d}.npz")
        np.savez_compressed(path, planes=planes_arr, elos=elos_arr, indices=indices_arr, values=values_arr)
        print(f"  保存 {len(samples_batch)} 条样本 -> {path}")
        samples_batch.clear()

    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1
            if max_games and game_count > max_games:
                break

            headers = game.headers
            if headers.get("Variant", "Standard") != "Standard":
                filtered_reasons['Variant'] += 1; continue
            if "Rated" not in headers.get("Event", ""):
                filtered_reasons['Unrated'] += 1; continue
            if headers.get("Result") not in ["1-0", "0-1", "1/2-1/2"]:
                filtered_reasons['Result'] += 1; continue
            if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
                filtered_reasons['BOT'] += 1; continue
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                filtered_reasons['EloParse'] += 1; continue
            if not (min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo):
                filtered_reasons['EloRange'] += 1; continue

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
                        bad_game = True; break
                    planes = board_to_planes(board)

                    white_count = sum(1 for sq, p in board.piece_map().items() if p.color == chess.WHITE)
                    black_count = sum(1 for sq, p in board.piece_map().items() if p.color == chess.BLACK)
                    if white_count <= max_per_side and black_count <= max_per_side:
                        value_label = evaluate_board(board, engine)
                        temp_buffer.append((planes, current_elo, idx, value_label))

                    board.push(move)
                    moves_played += 1
            except:
                bad_game = True

            if bad_game or moves_played < 10:
                filtered_reasons['BadMove/TooShort'] += 1; continue

            samples_batch.extend(temp_buffer)
            if len(samples_batch) >= chunk_size:
                save_chunk()

    save_chunk()
    engine.quit()
    print(f"\n文件 {os.path.basename(fpath)} 完成：处理 {game_count} 局")
    for reason, count in sorted(filtered_reasons.items()):
        print(f"    {reason}: {count}")
    return True

def main():
    raw_dir = 'D:/chessgame/data/lichess_raw'
    output_dir = 'D:/chessgame/data/cache/value_chunks'
    max_games_per_file = None
    max_per_side = 6
    sf_path = 'stockfish/stockfish-windows-x86-64-avx2.exe'
    num_workers = 4

    files_to_process = []
    for fname in os.listdir(raw_dir):
        fpath = os.path.join(raw_dir, fname)
        if fname.endswith('.pgn'):
            files_to_process.append(fpath)
        elif fname.endswith('.pgn.zst'):
            temp_path = fpath[:-4]
            print(f"解压 {fname} -> {os.path.basename(temp_path)} ...")
            with open(fpath, 'rb') as f_in:
                dctx = zstd.ZstdDecompressor()
                with open(temp_path, 'wb') as f_out:
                    dctx.copy_stream(f_in, f_out)
            files_to_process.append(temp_path)
        elif fname.endswith('.pgn.gz'):
            temp_path = fpath[:-3]
            print(f"解压 {fname} -> {os.path.basename(temp_path)} ...")
            with gzip.open(fpath, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            files_to_process.append(temp_path)

    if not files_to_process:
        print("未找到 PGN 文件。")
        return

    print(f"共 {len(files_to_process)} 个文件，使用 {num_workers} 个进程并行处理...")
    args = [(f, output_dir, 10000, max_games_per_file, 1100, 2400, max_per_side, sf_path)
            for f in files_to_process]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, arg) for arg in args]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="文件进度"):
            pass

    print("\n所有文件处理完毕，价值数据存储在:", output_dir)

if __name__ == '__main__':
    main()