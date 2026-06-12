import chess
import chess.engine
import random


class PuzzleGenerator:
    """生成切片训练题及变式。"""

    def __init__(self, engine=None):
        self.engine = engine

    def generate_puzzle(self, mistake: dict) -> dict:
        """将一条失误记录转换为一道训练题。"""
        board = chess.Board(mistake["fen"])
        return {
            "fen": mistake["fen"],
            "human_move": mistake["human_move"],
            "best_move": mistake["best_move"],
            "cp_loss": mistake["cp_loss"],
            "turn": mistake["turn"],
            "legal_moves": [m.uci() for m in board.legal_moves]
        }

    def generate_variations(self, puzzle: dict, n=2):
        """基于正确着法，生成 n 个变式局面（棋子微调）。"""
        variations = []
        board = chess.Board(puzzle["fen"])
        best_move = chess.Move.from_uci(puzzle["best_move"])
        board.push(best_move)

        # 获取局面中的随机棋子（排除王）
        pieces = [sq for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).piece_type != chess.KING]
        for _ in range(n):
            new_board = board.copy()
            if len(pieces) < 2:
                break
            # 随机选两个棋子交换位置（简单扰动）
            sq1, sq2 = random.sample(pieces, 2)
            piece1 = new_board.remove_piece_at(sq1)
            piece2 = new_board.remove_piece_at(sq2)
            if piece1 and piece2:
                new_board.set_piece_at(sq1, piece2)
                new_board.set_piece_at(sq2, piece1)
            # 确保王不在被攻击状态，且合法
            if new_board.is_valid() and not new_board.is_check():
                variations.append(new_board.fen())
        return variations