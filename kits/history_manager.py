import json
import chess
import chess.engine
import os
from datetime import datetime

HISTORY_DIR = "history"


class HistoryManager:
    """管理对局历史：保存、加载、提取失误步。"""

    def __init__(self, engine=None):
        self.engine = engine
        os.makedirs(HISTORY_DIR, exist_ok=True)

    def save_game(self, pgn_text, move_analysis, elo_history):
        """保存对局到 JSON 文件。"""
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        filepath = os.path.join(HISTORY_DIR, filename)
        data = {
            "pgn": pgn_text,
            "move_analysis": move_analysis,  # list of {uci, score_before, score_after, cp_loss}
            "elo_history": elo_history,  # list of (elo, velocity)
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return filepath

    def load_all_games(self):
        """加载所有历史对局。"""
        games = []
        for filename in sorted(os.listdir(HISTORY_DIR)):
            if filename.endswith('.json'):
                with open(os.path.join(HISTORY_DIR, filename), 'r', encoding='utf-8') as f:
                    games.append(json.load(f))
        return games

    def extract_mistakes(self, game_data, threshold_cp=200):
        """从一盘对局中提取失误步：cp_loss > threshold 的着法。"""
        mistakes = []
        board = chess.Board()
        moves = game_data.get("move_analysis", [])

        for i, step in enumerate(moves):
            uci = step["uci"]
            loss = step.get("cp_loss", 0)
            if loss is not None and loss > threshold_cp:
                # 保存失误前局面 FEN
                fen_before = board.fen()
                move = chess.Move.from_uci(uci)
                # 计算正确着法（用 Stockfish 快速搜索）
                best_move = None
                if self.engine:
                    result = self.engine.play(board, chess.engine.Limit(time=0.2))
                    best_move = result.move.uci()
                else:
                    # 无引擎时用概率最高的着法作为参考
                    pass

                mistakes.append({
                    "fen": fen_before,
                    "human_move": uci,
                    "best_move": best_move,
                    "cp_loss": loss,
                    "turn": "white" if board.turn == chess.WHITE else "black"
                })
            # 推进棋盘
            board.push(chess.Move.from_uci(uci))
        return mistakes

    def get_all_mistakes(self, threshold_cp=200):
        """获取所有历史对局中的失误集合。"""
        all_mistakes = []
        for game in self.load_all_games():
            all_mistakes.extend(self.extract_mistakes(game, threshold_cp))
        return all_mistakes