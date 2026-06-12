"""Maia 走法索引映射表 (UCI <-> Index) – 符合官方 1968 维定义"""
import chess

def generate_maps():
    moves = []
    # 1. 所有非升变的“皇后移动”（像后一样走，包含横竖斜任意格，但实际是所有普通移动）
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            if move.promotion is not None:
                continue  # 非升变先跳过
            # 合法移动定义：所有可能的目标格，没有任何限制
            moves.append(move.uci())
    # 2. 升变走法：兵到达对方底线
    for from_sq in chess.SQUARES:
        from_rank = chess.square_rank(from_sq)
        if from_rank == 6:   # 白兵升变 (第7行→第8行)
            to_rank = 7
            step = 1
        elif from_rank == 1: # 黑兵升变 (第2行→第1行)
            to_rank = 0
            step = -1
        else:
            continue
        from_file = chess.square_file(from_sq)
        for d_file in [-1, 0, 1]:
            to_file = from_file + d_file
            if 0 <= to_file <= 7:
                to_sq = chess.square(to_file, to_rank)
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    move = chess.Move(from_sq, to_sq, promotion=promo)
                    moves.append(move.uci())
    # 3. 王车易位
    moves.append('e1g1')
    moves.append('e1c1')
    moves.append('e8g8')
    moves.append('e8c8')
    # 去重并保持顺序
    seen = set()
    unique = []
    for m in moves:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    # 确保 1968
    if len(unique) != 1968:
        raise ValueError(f"生成的映射表大小为 {len(unique)}，不是 1968。请检查逻辑。")
    uci_to_idx = {uci: i for i, uci in enumerate(unique)}
    idx_to_uci = {i: uci for uci, i in uci_to_idx.items()}
    return uci_to_idx, idx_to_uci

UCI_TO_IDX, IDX_TO_UCI = generate_maps()