import tkinter as tk
from tkinter import messagebox, scrolledtext, Toplevel, filedialog
import chess
import chess.engine
import chess.syzygy
import random
import os
import threading
import queue
import json
import uuid
import sys
import onnxruntime as ort
import numpy as np
from datetime import datetime
from kits.mini_maia_ai import MiniMaiaAI
from kits.elo_tracker import EloTracker
from kits.cloud_review import CloudReviewer
from kits.history_manager import HistoryManager
from kits.mcts import MCTS


# ---------- 对弈核心 ----------
def estimate_elo_by_cp_loss(board, move, engine, baseline_elo=1500, k=1.0):
    prev_board = board.copy()
    prev_board.pop()
    try:
        info_prev = engine.analyse(prev_board, chess.engine.Limit(time=0.05))
        best_cp = info_prev["score"].white().score(mate_score=10000) if info_prev["score"].white() is not None else 0
        info_curr = engine.analyse(board, chess.engine.Limit(time=0.05))
        curr_cp = info_curr["score"].white().score(mate_score=10000) if info_curr["score"].white() is not None else 0
    except:
        return baseline_elo, 0.0
    cp_diff = best_cp - curr_cp
    est_elo = baseline_elo - k * cp_diff
    est_elo = max(800, min(2500, est_elo))
    return est_elo, cp_diff


class GameSession:
    def __init__(self, sf_path='stockfish/stockfish-windows-x86-64-avx2.exe', random_move_prob=0.09):
        self.ai_recent_moves = []
        self.board = chess.Board()
        self.ai = MiniMaiaAI()
        self.value_model = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        value_path = os.path.join(base_dir, 'models/cond_maia_value.onnx')
        if os.path.exists(value_path):
            try:
                self.value_model = ort.InferenceSession(value_path)
                print("✅ 价值模型已加载")
            except Exception as e:
                print(f"⚠️ 价值模型加载失败: {e}")
        self.elo_tracker = EloTracker(initial_elo=1500)
        self.sf_engine = chess.engine.SimpleEngine.popen_uci(os.path.join(base_dir, sf_path))
        self.ai_is_white = random.random() < 0.5
        self.temperature = 1.0
        self.random_move_prob = random_move_prob
        self.safety_threshold_cp = 200
        self.move_analysis = []
        self.last_human_cp_loss = 0
        self.reaction_threshold = 130
        # Syzygy
        self.tablebase = None
        syzygy_dir = os.path.join(base_dir, "syzygy")
        if os.path.isdir(syzygy_dir):
            try:
                self.tablebase = chess.syzygy.open_tablebase(syzygy_dir)
                print("✅ 已加载 Syzygy 表库")
            except Exception as e:
                print(f"⚠️ 加载 Syzygy 表库失败: {e}，将使用引擎计算。")
        else:
            print("ℹ️ 未找到 syzygy 文件夹，残局功能将回退到引擎。")

    # ---------- 基础工具方法 ----------
    def _sf_deep_move(self, time_limit=0.3):
        result = self.sf_engine.play(self.board, chess.engine.Limit(time=time_limit))
        return result.move

    def _get_sorted_moves(self, target_elo, temperature=1.0):
        probs = self.ai.get_move_probs(self.board, target_elo=target_elo, temperature=temperature)
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)

    def _record_ai_cp_loss(self, move):
        prev_board = self.board.copy()
        prev_board.pop()
        try:
            info_prev = self.sf_engine.analyse(prev_board, chess.engine.Limit(time=0.05))
            best_cp = info_prev["score"].white().score(mate_score=10000) if info_prev["score"].white() else 0
            info_curr = self.sf_engine.analyse(self.board, chess.engine.Limit(time=0.05))
            curr_cp = info_curr["score"].white().score(mate_score=10000) if info_curr["score"].white() else 0
            cp_loss = max(0, best_cp - curr_cp)
        except:
            cp_loss = 0.0
        self.move_analysis.append({
            'uci': move.uci(),
            'score_before': best_cp if 'best_cp' in locals() else None,
            'score_after': curr_cp if 'curr_cp' in locals() else None,
            'cp_loss': cp_loss
        })

    def _update_ai_recent_moves(self, move):
        self.ai_recent_moves.append(move.uci())
        if len(self.ai_recent_moves) > 2:
            self.ai_recent_moves.pop(0)

    def _calculate_brilliancy_bonus(self, cp_diff):
        if cp_diff >= -100:
            return 0
        base_bonus = min(150, abs(cp_diff) // 3)
        if self.last_human_cp_loss > 300:
            base_bonus *= 0.3
        if self.elo_tracker.elo > self.elo_tracker.initial_elo + 200:
            base_bonus *= 0.5
        return min(75, int(base_bonus))

    # ---------- 局面评估方法 ----------
    def _has_overwhelming_material_advantage(self):
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        our_value = 0
        opponent_value = 0
        for sq, piece in self.board.piece_map().items():
            v = piece_values.get(piece.piece_type, 0)
            if piece.color == self.board.turn:
                our_value += v
            else:
                opponent_value += v
        diff = our_value - opponent_value
        if opponent_value == 0 and diff >= 3:
            return True
        if diff >= 5:
            return True
        return False

    def _is_aggressive_context(self):
        try:
            info = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=6))
            score = info["score"].white()
            cp = score.score() if not score.is_mate() else 0
            side_advantage = cp if self.board.turn == chess.WHITE else -cp
            return side_advantage > 200
        except:
            return False

    def _evaluate_board_value(self, board):
        if not self.value_model:
            return 0.5
        from prepare_data import board_to_planes
        planes = board_to_planes(board).reshape(1, 112, 8, 8).astype(np.float32)
        value = self.value_model.run(None, {'board': planes})[0][0]
        return float(value)

    # ---------- 安全检查 ----------
    def _is_safe_move(self, move, depth=8, check_mate_threat=True):
        self.board.push(move)
        try:
            info = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            if score.is_mate() and score.mate() < 0:
                self.board.pop()
                return False
            cp = score.score() if not score.is_mate() else -10000
            move_count = self.board.fullmove_number
            threshold = 500 if move_count <= 20 else self.safety_threshold_cp
            if cp < -threshold:
                self.board.pop()
                return False

            # 后安全
            moving_piece = self.board.piece_at(move.to_square)
            if moving_piece and moving_piece.piece_type == chess.QUEEN:
                opponent_can_take_queen = any(
                    self.board.is_capture(m) and self.board.piece_at(m.to_square) and
                    self.board.piece_at(m.to_square).piece_type == chess.QUEEN
                    for m in self.board.legal_moves
                )
                if opponent_can_take_queen:
                    self.board.pop()
                    return False

            # 悬子价值检测
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn and piece.piece_type in (chess.QUEEN, chess.ROOK):
                    attackers = self.board.attackers(not self.board.turn, square)
                    if attackers:
                        attacker_value = min(piece_values[self.board.piece_at(att).piece_type]
                                             for att in attackers if self.board.piece_at(att))
                        defenders = [sq for sq in self.board.attackers(self.board.turn, square)
                                     if self.board.piece_at(sq).piece_type != chess.KING]
                        if attacker_value < piece_values[piece.piece_type] and len(defenders) < len(attackers):
                            self.board.pop()
                            return False

            # 王翼兵阵保护
            if not self._is_aggressive_context():
                king_sq = self.board.king(self.board.turn)
                if king_sq:
                    king_file = chess.square_file(king_sq)
                    king_rank = chess.square_rank(king_sq)
                    for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                        for r in [king_rank, king_rank + (1 if self.board.turn == chess.WHITE else -1)]:
                            if 0 <= r <= 7:
                                sq = chess.square(f, r)
                                piece = self.board.piece_at(sq)
                                if piece and piece.piece_type == chess.PAWN and piece.color == self.board.turn:
                                    if move.from_square == sq and move.to_square != sq:
                                        shield_count = sum(1 for ff in range(max(0, king_file - 1), min(8, king_file + 2))
                                                          for rr in [king_rank, king_rank + (1 if self.board.turn == chess.WHITE else -1)]
                                                          if 0 <= rr <= 7 and chess.square(ff, rr) != sq
                                                          and self.board.piece_at(chess.square(ff, rr))
                                                          and self.board.piece_at(chess.square(ff, rr)).piece_type == chess.PAWN
                                                          and self.board.piece_at(chess.square(ff, rr)).color == self.board.turn)
                                        if shield_count < 1:
                                            self.board.pop()
                                            return False

            # 将军有效性
            if self.board.is_check():
                try:
                    info_after = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=6))
                    score_after = info_after["score"].white()
                    if not score_after.is_mate():
                        cp_after = score_after.score()
                        if cp_after < cp - 50:
                            self.board.pop()
                            return False
                except:
                    pass

            # 防闷杀
            if check_mate_threat:
                opponent_moves = list(self.board.legal_moves)
                if len(opponent_moves) > 5:
                    scored = []
                    for om in opponent_moves:
                        self.board.push(om)
                        info2 = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=4))
                        sc = info2["score"].white().score(mate_score=10000) if not info2["score"].white().is_mate() else -10000
                        scored.append((sc, om))
                        self.board.pop()
                    scored.sort(key=lambda x: x[0])
                    opponent_moves = [m for _, m in scored[:5]]
                for om in opponent_moves:
                    self.board.push(om)
                    info3 = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=6))
                    score3 = info3["score"].white()
                    if (score3.is_mate() and score3.mate() < 0) or \
                       (not score3.is_mate() and score3.score() < -500):
                        self.board.pop()
                        self.board.pop()
                        return False
                    self.board.pop()
        except:
            pass
        self.board.pop()
        return True

    def _is_king_safe_on_square(self, move):
        self.board.push(move)
        king_sq = self.board.king(self.board.turn)
        if king_sq is None:
            self.board.pop()
            return True
        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)
        if rank in (0, 7) and file in (0, 7):
            defenders = 0
            for sq in chess.SQUARES:
                if chess.square_distance(king_sq, sq) <= 1:
                    piece = self.board.piece_at(sq)
                    if piece and piece.color == self.board.turn:
                        defenders += 1
            if defenders < 2:
                self.board.pop()
                return False
        self.board.pop()
        return True

    # ---------- 残局表库 ----------
    def _tablebase_best_move(self):
        if not self.tablebase or len(self.board.piece_map()) > 5:
            print("[表库] 条件不满足或无表库")
            return None
        try:
            wdl = self.tablebase.probe_wdl(self.board)
            print(f"[表库] 查询成功, WDL={wdl}")
        except Exception as e:
            print(f"[表库] 查询失败: {e}")
            return None
        except (KeyError, chess.syzygy.MissingTableError):
            return None

        best_move = None
        best_dtz = 9999
        best_wdl = -2
        for move in self.board.legal_moves:
            self.board.push(move)
            try:
                dtz = self.tablebase.probe_dtz(self.board)
                wdl_after = self.tablebase.probe_wdl(self.board)
            except (KeyError, chess.syzygy.MissingTableError):
                self.board.pop()
                continue

            if wdl == 2:
                if dtz >= 0 and dtz < best_dtz:
                    best_dtz = dtz
                    best_move = move
                if dtz == 0 and best_dtz > 0:
                    best_dtz = 0
                    best_move = move
            elif wdl == 0:
                if wdl_after in (0, 2) and dtz >= 0 and dtz < best_dtz:
                    best_dtz = dtz
                    best_move = move
            elif wdl == -2:
                if wdl_after == 0:
                    if best_wdl < 0:
                        best_wdl = 0
                        best_move = move
                        best_dtz = dtz
                    elif best_wdl == 0 and dtz < best_dtz:
                        best_dtz = dtz
                        best_move = move
                elif wdl_after == -2 and best_wdl == -2 and dtz > best_dtz:
                    best_dtz = dtz
                    best_move = move
            self.board.pop()
        return best_move

    # ---------- 其他局面特征 ----------
    def _promotion_threat(self):
        direction = 1 if self.board.turn == chess.WHITE else -1
        target_rank = 7 if self.board.turn == chess.WHITE else 0
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == self.board.turn:
                if abs(chess.square_rank(sq) - target_rank) <= 2:
                    return True
        return False

    def _has_passed_pawn(self):
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                color = piece.color
                blocked = False
                if color == chess.WHITE:
                    for r in range(rank + 1, 8):
                        for f in (file - 1, file, file + 1):
                            if 0 <= f <= 7:
                                p = self.board.piece_at(chess.square(f, r))
                                if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                                    blocked = True
                                    break
                        if blocked: break
                    if not blocked: return True
                else:
                    for r in range(rank - 1, -1, -1):
                        for f in (file - 1, file, file + 1):
                            if 0 <= f <= 7:
                                p = self.board.piece_at(chess.square(f, r))
                                if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                                    blocked = True
                                    break
                        if blocked: break
                    if not blocked: return True
        return False

    def _is_draw_by_repetition(self, move):
        self.board.push(move)
        is_rep = self.board.is_repetition(3)
        self.board.pop()
        return is_rep

    def _is_perpetual_check(self, move, depth=6):
        self.board.push(move)
        try:
            info = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            if not score.is_mate() and abs(score.score()) < 10:
                self.board.pop()
                return True
        except:
            pass
        self.board.pop()
        return False

    def _pawn_shield_broken(self):
        king_sq = self.board.king(self.board.turn)
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        direction = 1 if self.board.turn == chess.WHITE else -1
        shield_rank = king_rank + direction
        if not (0 <= shield_rank <= 7):
            return False
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            sq = chess.square(f, shield_rank)
            piece = self.board.piece_at(sq)
            if not piece or piece.piece_type != chess.PAWN or piece.color != self.board.turn:
                return True
        return False

    def _count_passed_pawns(self):
        count = 0
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == self.board.turn:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                color = piece.color
                blocked = False
                if color == chess.WHITE:
                    for r in range(rank + 1, 8):
                        for f in (file - 1, file, file + 1):
                            if 0 <= f <= 7:
                                p = self.board.piece_at(chess.square(f, r))
                                if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                                    blocked = True
                                    break
                        if blocked: break
                    if not blocked: count += 1
                else:
                    for r in range(rank - 1, -1, -1):
                        for f in (file - 1, file, file + 1):
                            if 0 <= f <= 7:
                                p = self.board.piece_at(chess.square(f, r))
                                if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                                    blocked = True
                                    break
                        if blocked: break
                    if not blocked: count += 1
        return count

    # ---------- 兑子/攻击模块 ----------
    def _evaluate_queen_exchange(self):
        board = self.board
        if not board.pieces(chess.QUEEN, chess.WHITE) or not board.pieces(chess.QUEEN, chess.BLACK):
            return 0
        try:
            info_before = self.sf_engine.analyse(board, chess.engine.Limit(depth=6))
            if info_before["score"].is_mate():
                return 0
            score_before = info_before["score"].white().score()
        except:
            return 0

        exchange_move = None
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece and captured_piece.piece_type == chess.QUEEN:
                    board.push(move)
                    can_recapture = any(
                        board.is_capture(m) and board.piece_at(m.to_square) and
                        board.piece_at(m.to_square).piece_type == chess.QUEEN
                        for m in board.legal_moves
                    )
                    board.pop()
                    if can_recapture:
                        exchange_move = move
                        break
        if not exchange_move:
            return 0

        board.push(exchange_move)
        for recapture in board.legal_moves:
            if board.is_capture(recapture) and board.piece_at(recapture.to_square) and \
               board.piece_at(recapture.to_square).piece_type == chess.QUEEN:
                board.push(recapture)
                break
        else:
            board.pop()
            return 0
        try:
            info_after = self.sf_engine.analyse(board, chess.engine.Limit(depth=6))
            if info_after["score"].is_mate():
                board.pop(); board.pop()
                return 0
            score_after = info_after["score"].white().score()
        except:
            board.pop(); board.pop()
            return 0
        board.pop(); board.pop()

        if score_before is None or score_after is None:
            return 0
        if board.turn == chess.WHITE:
            delta = score_after - score_before
        else:
            delta = score_before - score_after
        return delta

    def _get_exchange_moves(self):
        moves = []
        for move in self.board.legal_moves:
            piece = self.board.piece_at(move.from_square)
            captured = self.board.piece_at(move.to_square)
            if piece and captured:
                if piece.piece_type in (chess.QUEEN, chess.ROOK) and captured.piece_type in (chess.QUEEN, chess.ROOK):
                    moves.append(move)
        return moves

    def _evaluate_exchange_outcome(self, move):
        self.board.push(move)
        try:
            info = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=8))
            score = info["score"].white()
            if score.is_mate():
                adv = -5000 if score.mate() < 0 else 5000
            else:
                cp = score.score()
                adv = cp if self.board.turn == chess.WHITE else -cp
        except:
            adv = 0
        self.board.pop()
        return adv

    def _endgame_aggressive_search(self, time_limit=1.5):
        try:
            if len(list(self.board.legal_moves)) <= 3:
                return self._sf_deep_move(time_limit=0.5)
            result = self.sf_engine.analyse(self.board, chess.engine.Limit(time=time_limit), multipv=3)
            if not isinstance(result, list):
                return self._sf_deep_move(time_limit=time_limit)
            best_move = None
            best_score = -9999
            for info in result:
                move = info["pv"][0]
                score = info["score"].white()
                cp = 10000 if score.mate() and score.mate() > 0 else (-10000 if score.mate() and score.mate() < 0 else score.score())
                if self.board.turn == chess.BLACK:
                    cp = -cp
                attack_bonus = self._evaluate_threat(self.board, move, info)
                total_score = cp + attack_bonus
                if total_score > best_score:
                    best_score = total_score
                    best_move = move
            return best_move or self._sf_deep_move(time_limit=time_limit)
        except:
            return self._sf_deep_move(time_limit=1.0)

    def _evaluate_threat(self, board_before, move, info):
        board = board_before.copy()
        board.push(move)
        bonus = 0
        if board.is_check():
            bonus += 80
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn and piece.piece_type in (chess.QUEEN, chess.ROOK):
                if board.is_attacked_by(board.turn, square):
                    bonus += 50 if piece.piece_type == chess.QUEEN else 30 if not board.is_attacked_by(not board.turn, square) else 20 if piece.piece_type == chess.QUEEN else 10
        high_value_squares = [sq for sq in chess.SQUARES
                              if board.piece_at(sq) and board.piece_at(sq).color != board.turn
                              and board.piece_at(sq).piece_type in (chess.QUEEN, chess.ROOK)]
        attacked_high = [sq for sq in high_value_squares if board.is_attacked_by(board.turn, sq)]
        if len(attacked_high) >= 2:
            bonus += 70
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_pinned(not board.turn, square):
                    bonus += 60
                    break
        return bonus

    # ---------- 最终执行与审核 ----------
    def _execute_ai_move(self, move):
        if not self._is_safe_move(move) or self._is_draw_by_repetition(move) or self._is_perpetual_check(move):
            sorted_moves = self._get_sorted_moves(target_elo=self.elo_tracker.elo, temperature=0.3)
            for alt_move, _ in sorted_moves:
                if self._is_safe_move(alt_move) and not self._is_draw_by_repetition(alt_move) and not self._is_perpetual_check(alt_move):
                    move = alt_move
                    break
            else:
                move = sorted_moves[0][0]
        verified_move = self._verify_best_move(move)
        self.board.push(verified_move)
        self._update_ai_recent_moves(verified_move)
        self._record_ai_cp_loss(verified_move)
        self.last_human_cp_loss = 0
        return verified_move

    def _verify_best_move(self, candidate_move, threshold_cp=50):
        if not self.board.legal_moves:
            return candidate_move
        try:
            info = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=4, time=0.02))
            pv = info.get("pv", [])
            best_move = pv[0] if pv else candidate_move
            if best_move == candidate_move:
                return candidate_move
            self.board.push(candidate_move)
            info_cand = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=4, time=0.02))
            score_cand = info_cand["score"].white()
            self.board.pop()
            self.board.push(best_move)
            info_best = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=4, time=0.02))
            score_best = info_best["score"].white()
            self.board.pop()
            if self.board.turn == chess.WHITE:
                cp_diff = score_best.score() - score_cand.score()
            else:
                cp_diff = score_cand.score() - score_best.score()
            if cp_diff > threshold_cp:
                return best_move
            return candidate_move
        except:
            return candidate_move

    # ---------- 主要决策 ----------
    def play_ai_move(self):
        if len(self.board.piece_map()) <= 5:
            move = self._tablebase_best_move()
            if move is not None:
                return self._execute_ai_move(move)
        if self._has_overwhelming_material_advantage():
            move = self._sf_deep_move(time_limit=2.0)
            return self._execute_ai_move(move)
        if len(self.board.piece_map()) <= 6:
            move = self._sf_deep_move(time_limit=1.2)
            return self._execute_ai_move(move)

        current_elo = self.elo_tracker.elo
        if current_elo <= 870:
            target_elo_base = random.randint(900, 1000)
        else:
            target_elo_base = current_elo

        if self.last_human_cp_loss > 150:
            move = self._sf_deep_move(time_limit=0.6)
            return self._execute_ai_move(move)

        try:
            info_quick = self.sf_engine.analyse(self.board, chess.engine.Limit(depth=6))
            quick_score = info_quick["score"].white()
            if quick_score.is_mate():
                white_advantage = 10000 if quick_score.mate() > 0 else -10000
            else:
                white_advantage = quick_score.score()
        except:
            white_advantage = 0
        side_advantage = white_advantage if self.board.turn == chess.WHITE else -white_advantage
        if abs(side_advantage) < 120:
            side_advantage += self._evaluate_queen_exchange()

        move_count = self.board.fullmove_number
        piece_count = len(self.board.piece_map())
        king_sq = self.board.king(self.board.turn)
        attackers = self.board.attackers(not self.board.turn, king_sq)
        king_danger = len(attackers) >= 1
        pawn_shield_broken = self._pawn_shield_broken()
        promotion_threat = self._promotion_threat()
        has_passed_pawn = self._has_passed_pawn()
        passer_count = self._count_passed_pawns()
        multiple_passers = passer_count >= 2
        overwhelming_material = self._has_overwhelming_material_advantage()
        defensive_mode = (side_advantage < -100) and (king_danger or pawn_shield_broken)
        aggressive_mode = (side_advantage > 200) or promotion_threat or (piece_count <= 12) or \
                          (has_passed_pawn and side_advantage > 100) or multiple_passers or overwhelming_material
        sneak_mode = (100 < side_advantage <= 200) and pawn_shield_broken

        if defensive_mode and move_count > 10:
            move = self._sf_deep_move(time_limit=0.5)
            return self._execute_ai_move(move)
        if aggressive_mode and move_count > 10:
            move = self._sf_deep_move(time_limit=0.8 if side_advantage > 300 else 0.5)
            return self._execute_ai_move(move)
        if side_advantage > 300 and move_count > 10:
            move = self._sf_deep_move(time_limit=1.0)
            return self._execute_ai_move(move)
        if not aggressive_mode and side_advantage < -200 and has_passed_pawn:
            move = self._sf_deep_move(time_limit=1.5)
            return self._execute_ai_move(move)

        trigger_mcts = False
        ai_side_ranks = (0, 1, 2, 3) if self.board.turn == chess.WHITE else (4, 5, 6, 7)
        invader_count = 0
        for sq, piece in self.board.piece_map().items():
            if piece.color != self.board.turn and chess.square_rank(sq) in ai_side_ranks:
                invader_count += 1
        if move_count > 8:
            if invader_count > 5 or \
               (piece_count <= 28 and (king_danger or has_passed_pawn or pawn_shield_broken)) or \
               self.last_human_cp_loss < -200 or side_advantage < -150 or piece_count <= 16:
                trigger_mcts = True

        if trigger_mcts:
            try:
                search_time = 3.0 if side_advantage < -200 or self.last_human_cp_loss < -200 or invader_count > 5 else 1.5 if piece_count <= 16 else 2.0
                mcts = MCTS(self.ai, self.ai, stockfish_engine=self.sf_engine, tablebase=self.tablebase)
                mcts_probs = mcts.search(self.board, time_limit=search_time, c_puct=1.0)
                moves = list(mcts_probs.keys())
                probs = np.array([mcts_probs[m] for m in moves])
                temp = 0.3 if side_advantage < -100 or self.last_human_cp_loss < -200 or invader_count > 5 else 0.5 if aggressive_mode else 0.8
                probs = np.log(probs + 1e-10) / temp
                probs = np.exp(probs - np.max(probs))
                probs /= probs.sum()
                move = np.random.choice(moves, p=probs)
                return self._execute_ai_move(move)
            except:
                pass

        if not defensive_mode and not aggressive_mode and move_count > 25:
            white_count = sum(1 for sq, p in self.board.piece_map().items() if p.color == chess.WHITE)
            black_count = sum(1 for sq, p in self.board.piece_map().items() if p.color == chess.BLACK)
            if white_count <= 6 and black_count <= 6 and (white_count + black_count) <= 16:
                move = self._endgame_aggressive_search(time_limit=1.5)
                if move:
                    return self._execute_ai_move(move)

        if move_count <= 8:
            base_temp = 1.65
        elif 10 < move_count < 40 and piece_count > 20:
            base_temp = self.temperature - 0.3
        else:
            base_temp = self.temperature
        adjusted_temp = max(0.3, base_temp)
        if aggressive_mode:
            adjusted_temp = 0.3

        advantage = random.uniform(300, 450)
        target_elo = min(2800, target_elo_base + advantage)
        sorted_moves = self._get_sorted_moves(target_elo=target_elo, temperature=adjusted_temp)
        if not sorted_moves:
            return self._execute_ai_move(random.choice(list(self.board.legal_moves)))

        if not aggressive_mode and not defensive_mode and move_count > 8:
            exchange_moves = self._get_exchange_moves()
            if exchange_moves:
                best_exchange = None
                best_advantage = -999
                for move in exchange_moves:
                    adv = self._evaluate_exchange_outcome(move)
                    if adv > best_advantage:
                        best_advantage = adv
                        best_exchange = move
                if best_advantage >= 20 and best_exchange:
                    return self._execute_ai_move(best_exchange)
                elif best_advantage < -20:
                    exchange_set = set(exchange_moves)
                    sorted_moves = [(m, p) for m, p in sorted_moves if m not in exchange_set]
                    if not sorted_moves:
                        sorted_moves = self._get_sorted_moves(target_elo=target_elo, temperature=adjusted_temp)

        if move_count <= 6:
            best_prob = sorted_moves[0][1]
            threshold = best_prob * 0.4
            candidate_moves = [m for m, p in sorted_moves if p >= threshold][:4]
            if len(candidate_moves) >= 2:
                random.shuffle(candidate_moves)
                for move in candidate_moves:
                    if self._is_safe_move(move) and not self._is_draw_by_repetition(move) and not self._is_perpetual_check(move):
                        return self._execute_ai_move(move)
            move = sorted_moves[0][0]
            return self._execute_ai_move(move)

        if sneak_mode and random.random() < 0.25 and len(sorted_moves) >= 4:
            sneak_candidates = [m for m, p in sorted_moves[1:4]]
            random.shuffle(sneak_candidates)
            for move in sneak_candidates:
                if self._is_safe_move(move) and not self._is_draw_by_repetition(move) and not self._is_perpetual_check(move):
                    return self._execute_ai_move(move)

        old_random_prob = self.random_move_prob
        if aggressive_mode or defensive_mode:
            self.random_move_prob = 0.0
        if random.random() < self.random_move_prob:
            top_n = max(2, len(sorted_moves) // 10)
            candidate_moves = [m for m, p in sorted_moves[1:top_n+1]]
            if candidate_moves:
                for move in candidate_moves:
                    if self._is_safe_move(move) and not self._is_draw_by_repetition(move) and not self._is_perpetual_check(move):
                        return self._execute_ai_move(move)
            for move, _ in sorted_moves:
                if self._is_safe_move(move) and not self._is_draw_by_repetition(move) and not self._is_perpetual_check(move) and self._is_king_safe_on_square(move):
                    return self._execute_ai_move(move)
            move = sorted_moves[0][0]
            return self._execute_ai_move(move)

        if self.value_model and move_count > 10 and piece_count <= 14:
            base_value = self._evaluate_board_value(self.board)
            filtered = []
            for move, prob in sorted_moves:
                self.board.push(move)
                new_value = self._evaluate_board_value(self.board)
                self.board.pop()
                if new_value >= base_value - 0.05:
                    filtered.append((move, prob))
            if filtered:
                sorted_moves = filtered

        for move, _ in sorted_moves:
            if self._is_safe_move(move) and not self._is_draw_by_repetition(move) and not self._is_perpetual_check(move) and self._is_king_safe_on_square(move):
                return self._execute_ai_move(move)
        move = sorted_moves[0][0]
        return self._execute_ai_move(move)

    def play_human_move(self, uci_str):
        try:
            move = chess.Move.from_uci(uci_str)
            if move not in self.board.legal_moves:
                return False, None, None
            board_before = self.board.copy()
            maia_elo_obs = self.ai.predict_elo(board_before)
            self.board.push(move)
            sf_elo_obs, cp_diff = estimate_elo_by_cp_loss(self.board, move, self.sf_engine, baseline_elo=self.elo_tracker.elo)
            brilliancy_bonus = self._calculate_brilliancy_bonus(cp_diff)
            sf_elo_obs += brilliancy_bonus
            self.last_human_cp_loss = max(0, cp_diff)
            board_complexity = {'piece_count': len(self.board.piece_map()), 'legal_moves': len(list(self.board.legal_moves))}
            self.move_analysis.append({'uci': move.uci(), 'score_before': None, 'score_after': None, 'cp_loss': cp_diff})
            self.elo_tracker.update(maia_elo_obs, measurement_noise=5000, board_complexity=board_complexity, cp_loss=cp_diff)
            self.elo_tracker.update(sf_elo_obs, measurement_noise=None, board_complexity=board_complexity, cp_loss=cp_diff)
            elo = self.elo_tracker.elo
            vel = self.elo_tracker.velocity
            base_temp = max(0.5, min(1.5, 2.0 - (elo - 800) / 1200))
            self.temperature = max(0.3, min(1.8, base_temp - vel * 0.05))
            return True, elo, vel
        except Exception as e:
            print(f"评估出错: {e}")
            return False, None, None

    def cleanup(self):
        self.ai.quit()
        self.sf_engine.quit()


# ---------- 错题管理器 ----------
class PuzzleManager:
    def __init__(self, puzzles_dir="puzzles"):
        self.puzzles_dir = puzzles_dir
        os.makedirs(puzzles_dir, exist_ok=True)

    def generate_puzzles_from_game(self, game_data, threshold_cp=150, engine=None):
        puzzles = []
        board = chess.Board()
        for step in game_data.get("move_analysis", []):
            uci = step["uci"]
            loss = step.get("cp_loss", 0)
            if loss is not None and loss > threshold_cp:
                fen = board.fen()
                best_move = None
                if engine:
                    try:
                        result = engine.play(board, chess.engine.Limit(time=0.2))
                        best_move = result.move.uci()
                    except:
                        pass
                puzzles.append({"fen": fen, "human_move": uci, "best_move": best_move, "cp_loss": loss, "turn": "white" if board.turn == chess.WHITE else "black"})
            board.push(chess.Move.from_uci(uci))
        return puzzles

    def save_puzzle(self, puzzle_data, filename=None):
        if filename is None:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6] + ".puzzle"
        filepath = os.path.join(self.puzzles_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(puzzle_data, f, indent=2)
        return filepath

    def load_puzzle(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_puzzles(self):
        return sorted([f for f in os.listdir(self.puzzles_dir) if f.endswith('.puzzle')])


# ---------- 图形界面 ----------
class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("国际象棋 AI 教练 (旗舰版)")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        self.session = GameSession()
        self.selected_square = None
        self.legal_dests = []
        self.game_over = False
        self.ai_thinking = False
        self.flip_board = self.session.ai_is_white
        self.ai_thread = None
        self.ai_queue = queue.Queue()
        self.pgn_history = []
        self.reviewer = None
        self.history_mgr = HistoryManager(engine=self.session.sf_engine)
        self.puzzle_mgr = PuzzleManager()

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, width=520, height=520, bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_square_click)

        right_frame = tk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.status_label = tk.Label(right_frame, text="你执白" if not self.flip_board else "你执黑", font=("Arial", 14, "bold"))
        self.status_label.pack(pady=5)
        self.elo_label = tk.Label(right_frame, text="融合Elo: 1500\n速度: 0.00\nAI温度: 1.00", font=("Arial", 11))
        self.elo_label.pack(pady=10)
        self.move_label = tk.Label(right_frame, text="AI最后着法: --", font=("Arial", 10))
        self.move_label.pack(pady=5)
        self.log_text = tk.Text(right_frame, height=10, width=40, state=tk.DISABLED)
        self.log_text.pack(pady=10)

        self.restart_btn = tk.Button(right_frame, text="重新开始", command=self.restart_game, font=("Arial", 11))
        self.restart_btn.pack(pady=5)
        self.review_btn = tk.Button(right_frame, text="云端复盘", command=self.request_cloud_review, font=("Arial", 11), state=tk.DISABLED)
        self.review_btn.pack(pady=5)
        self.gen_puzzle_btn = tk.Button(right_frame, text="生成错题", command=self.generate_puzzles, font=("Arial", 11))
        self.gen_puzzle_btn.pack(pady=5)
        self.review_puzzle_btn = tk.Button(right_frame, text="复习错题", command=self.review_puzzles, font=("Arial", 11))
        self.review_puzzle_btn.pack(pady=5)
        self.open_puzzle_btn = tk.Button(right_frame, text="打开错题文件...", command=self.browse_puzzle, font=("Arial", 11))
        self.open_puzzle_btn.pack(pady=5)
        self.tutorial_btn = tk.Button(right_frame, text="新手教程", command=self.show_tutorial, font=("Arial", 11))
        self.tutorial_btn.pack(pady=5)
        self.analyze_btn = tk.Button(right_frame, text="分析局面", command=self.analyze_position, font=("Arial", 11))
        self.analyze_btn.pack(pady=5)

        self.init_cloud_reviewer()
        self.draw_board()
        if self.session.ai_is_white:
            self.request_ai_move()
        self.root.after(100, self.check_ai_queue)

    # ---------- 教程 ----------
    def show_tutorial(self):
        win = Toplevel(self.root)
        win.title("国际象棋新手教程")
        win.geometry("700x600")
        pages = [
            "欢迎来到国际象棋AI教练！\n\n本教程将带你快速了解国际象棋的基本规则。",
            "棋子与走法：\n♔王：横直斜各一格\n♕后：横直斜任意格\n♖车：横直任意格\n♗象：斜线任意格\n♘马：L形跳跃（可越子）\n♙兵：直走斜吃，首次可走两格",
            "特殊规则：\n- 吃过路兵：兵从起始位走两格，邻兵可斜吃\n- 王车易位：王向车方向移两格，车跳至王另一侧\n- 兵升变：兵冲至底线可升变为后、车、象、马",
            "胜负与和棋：\n- 将死对方王获胜\n- 对方认输\n- 逼和（无子可动）\n- 三次重复局面\n- 50回合无吃子无动兵",
            "基础战术：\n- 牵制：攻击对方掩护王或高价值棋子的棋子\n- 捉双：同时攻击两个目标\n- 闪击：移动中间子力暴露后面攻击\n- 引入：将对方王引入杀棋网\n- 封锁：限制对方棋子活动",
            "现在你已经了解了规则，快去和AI下一盘棋吧！"
        ]
        current_page = [0]
        text_area = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 12))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_area.insert(tk.END, pages[0])
        text_area.config(state=tk.DISABLED)

        def next_page():
            if current_page[0] < len(pages) - 1:
                current_page[0] += 1
                text_area.config(state=tk.NORMAL)
                text_area.delete(1.0, tk.END)
                text_area.insert(tk.END, pages[current_page[0]])
                text_area.config(state=tk.DISABLED)

        def prev_page():
            if current_page[0] > 0:
                current_page[0] -= 1
                text_area.config(state=tk.NORMAL)
                text_area.delete(1.0, tk.END)
                text_area.insert(tk.END, pages[current_page[0]])
                text_area.config(state=tk.DISABLED)

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="上一页", command=prev_page).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="下一页", command=next_page).pack(side=tk.LEFT, padx=5)

    # ---------- 分析 ----------
    def analyze_position(self):
        if self.game_over:
            messagebox.showinfo("提示", "对局已结束，无局面分析。")
            return
        board = self.session.board
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            messagebox.showinfo("提示", "无合法着法。")
            return
        try:
            mcts = MCTS(self.session.ai, self.session.ai, stockfish_engine=self.session.sf_engine, tablebase=self.session.tablebase)
            probs = mcts.search(board, time_limit=2.5)
            sorted_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            messagebox.showerror("错误", f"MCTS分析失败: {e}")
            return
        win = Toplevel(self.root)
        win.title("局面分析")
        win.geometry("400x300")
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 11))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        report = f"当前局面 FEN: {board.fen()}\n\n推荐着法（按胜率排序）:\n"
        for i, (move, prob) in enumerate(sorted_moves[:5]):
            san = board.san(move)
            report += f"{i+1}. {san} ({move.uci()}) - 概率: {prob:.2%}\n"
        report += "\n点击任意着法可查看详情（暂未实现）。"
        text.insert(tk.END, report)
        text.config(state=tk.DISABLED)

    # ---------- 错题管理 ----------
    def browse_puzzle(self):
        filepath = filedialog.askopenfilename(title="选择错题文件", filetypes=[("错题文件", "*.puzzle"), ("所有文件", "*.*")])
        if filepath:
            self.open_puzzle_review(filepath)

    def init_cloud_reviewer(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            self.reviewer = CloudReviewer(api_key=api_key, ai_player_name="AI")
            self.review_btn.config(state=tk.NORMAL)
        else:
            self.review_btn.config(state=tk.DISABLED)

    def request_cloud_review(self):
        if not self.reviewer:
            messagebox.showinfo("提示", "未配置 DeepSeek API Key，无法使用云端复盘。")
            return
        pgn = self.session.board.pgn() if hasattr(self.session.board, 'pgn') else " ".join(self.pgn_history)
        if not pgn.strip():
            messagebox.showinfo("提示", "暂无对局记录。")
            return
        self.append_log("正在请求云端复盘...")
        def do_review():
            result = self.reviewer.analyze(pgn)
            self.ai_queue.put(('review', result))
        threading.Thread(target=do_review, daemon=True).start()

    def show_review_report(self, report):
        win = Toplevel(self.root)
        win.title("云端复盘报告")
        win.geometry("700x600")
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Arial", 11))
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, str(report))
        text.config(state=tk.DISABLED)

    def generate_puzzles(self):
        games = self.history_mgr.load_all_games()
        existing = set()
        for fname in self.puzzle_mgr.list_puzzles():
            try:
                p = self.puzzle_mgr.load_puzzle(os.path.join(self.puzzle_mgr.puzzles_dir, fname))
                existing.add((p["fen"], p["human_move"]))
            except:
                pass
        count = 0
        for game in games:
            puzzles = self.puzzle_mgr.generate_puzzles_from_game(game, threshold_cp=150, engine=self.session.sf_engine)
            for p in puzzles:
                key = (p["fen"], p["human_move"])
                if key not in existing:
                    self.puzzle_mgr.save_puzzle(p)
                    existing.add(key)
                    count += 1
        messagebox.showinfo("完成", f"新增 {count} 道错题，保存在 {self.puzzle_mgr.puzzles_dir}")

    def review_puzzles(self):
        puzzles = self.puzzle_mgr.list_puzzles()
        if not puzzles:
            messagebox.showinfo("提示", "暂无错题，请先生成。")
            return
        filepath = os.path.join(self.puzzle_mgr.puzzles_dir, puzzles[-1])
        self.open_puzzle_review(filepath)

    def open_puzzle_review(self, filepath):
        puzzle = self.puzzle_mgr.load_puzzle(filepath)
        win = Toplevel(self.root)
        win.title("错题复习")
        win.geometry("600x650")
        win.resizable(False, False)
        board = chess.Board(puzzle["fen"])
        canvas = tk.Canvas(win, width=400, height=400)
        canvas.pack(pady=10)
        self.draw_board_on_canvas(canvas, board, flip=False)
        info = f"失误着法: {puzzle['human_move']}  损失: {puzzle['cp_loss']} cp"
        tk.Label(win, text=info, font=("Arial", 11)).pack()
        move_var = tk.StringVar()
        entry = tk.Entry(win, textvariable=move_var, font=("Arial", 12))
        entry.pack(pady=5)
        entry.focus()
        def check_move():
            uci = move_var.get().strip()
            move = chess.Move.from_uci(uci) if uci else None
            if move and move in board.legal_moves:
                if uci == puzzle["best_move"]:
                    messagebox.showinfo("正确", "好棋！这就是最佳着法。")
                else:
                    messagebox.showinfo("继续加油", f"推荐着法: {puzzle['best_move']}")
                win.destroy()
            else:
                messagebox.showerror("错误", "非法着法或格式错误")
        tk.Button(win, text="提交", command=check_move, font=("Arial", 12)).pack(pady=5)
        def show_answer():
            move = chess.Move.from_uci(puzzle["best_move"])
            board.push(move)
            self.draw_board_on_canvas(canvas, board, flip=False)
            messagebox.showinfo("答案", f"最佳着法: {puzzle['best_move']}")
            win.destroy()
        tk.Button(win, text="显示答案（动画）", command=show_answer, font=("Arial", 12)).pack(pady=5)

    # ---------- 棋盘绘制 ----------
    def draw_board_on_canvas(self, canvas, board, flip=False):
        canvas.delete("all")
        size = 50
        for row in range(8):
            for col in range(8):
                x1, y1 = col * size, row * size
                x2, y2 = x1 + size, y1 + size
                color = "#f5e6c8" if (row + col) % 2 == 0 else "#8b7355"
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        piece_unicode = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                display_rank, display_file = (7 - rank, 7 - file) if flip else (7 - rank, file)
                x = display_file * size + size // 2
                y = display_rank * size + size // 2
                color = "#000000" if piece.color == chess.BLACK else "#ffffff"
                canvas.create_text(x+1, y+1, text=piece_unicode[piece.symbol()], font=("Arial", 25), fill="#888888")
                canvas.create_text(x, y, text=piece_unicode[piece.symbol()], font=("Arial", 25), fill=color)

    def draw_board(self):
        self.canvas.delete("all")
        size = 60
        for row in range(8):
            for col in range(8):
                x1, y1 = col * size, row * size
                x2, y2 = x1 + size, y1 + size
                color = "#f5e6c8" if (row + col) % 2 == 0 else "#8b7355"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        piece_unicode = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
        }
        for square in chess.SQUARES:
            piece = self.session.board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                display_rank, display_file = (7 - rank, file) if not self.flip_board else (rank, 7 - file)
                x = display_file * size + size // 2
                y = display_rank * size + size // 2
                color = "#000000" if piece.color == chess.BLACK else "#ffffff"
                self.canvas.create_text(x+1, y+1, text=piece_unicode[piece.symbol()], font=("Arial", 30), fill="#888888")
                self.canvas.create_text(x, y, text=piece_unicode[piece.symbol()], font=("Arial", 30), fill=color)
        if self.selected_square is not None:
            sr, sf = chess.square_rank(self.selected_square), chess.square_file(self.selected_square)
            dr, df = (7 - sr, sf) if not self.flip_board else (sr, 7 - sf)
            x1, y1 = df * size, dr * size
            self.canvas.create_rectangle(x1, y1, x1+size, y1+size, outline="red", width=2)
            for dest in self.legal_dests:
                dr2, df2 = chess.square_rank(dest), chess.square_file(dest)
                dr2, df2 = (7 - dr2, df2) if not self.flip_board else (dr2, 7 - df2)
                x1, y1 = df2 * size, dr2 * size
                self.canvas.create_rectangle(x1, y1, x1+size, y1+size, outline="green", width=2)

    # ---------- 交互 ----------
    def on_square_click(self, event):
        if self.game_over or self.ai_thinking:
            return
        human_turn = (self.session.board.turn == chess.WHITE) != self.session.ai_is_white
        if not human_turn:
            return
        size = 60
        col = event.x // size
        row = event.y // size
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        rank, file = (7 - row, col) if not self.flip_board else (row, 7 - col)
        if not (0 <= file < 8 and 0 <= rank < 8):
            return
        square = chess.square(file, rank)
        if self.selected_square is None:
            piece = self.session.board.piece_at(square)
            if piece and piece.color == self.session.board.turn:
                self.selected_square = square
                self.legal_dests = [m.to_square for m in self.session.board.legal_moves if m.from_square == square]
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, square)
            if move.promotion is None and self.session.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(square) in (0, 7):
                    move.promotion = chess.QUEEN
            if move in self.session.board.legal_moves:
                uci = move.uci()
                ok, elo, vel = self.session.play_human_move(uci)
                if ok:
                    self.pgn_history.append(uci)
                    self.selected_square = None
                    self.legal_dests = []
                    self.draw_board()
                    self.update_status()
                    self.append_log(f"你走: {uci}")
                    if not self.session.board.is_game_over():
                        self.request_ai_move()
                    else:
                        self.on_game_over()
                else:
                    self.selected_square = None; self.legal_dests = []; self.draw_board()
            else:
                self.selected_square = None; self.legal_dests = []; self.draw_board()

    def request_ai_move(self):
        if self.game_over or self.ai_thinking:
            return
        self.ai_thinking = True
        self.move_label.config(text="AI思考中...")
        self.ai_thread = threading.Thread(target=self._ai_thread_func, daemon=True)
        self.ai_thread.start()

    def _ai_thread_func(self):
        try:
            move = self.session.play_ai_move()
            self.ai_queue.put(('ai_move', move))
        except Exception as e:
            self.ai_queue.put(('error', str(e)))

    def check_ai_queue(self):
        try:
            while True:
                msg = self.ai_queue.get_nowait()
                if msg[0] == 'ai_move':
                    move = msg[1]
                    self.ai_thinking = False
                    self.pgn_history.append(move.uci())
                    self.append_log(f"AI走: {move.uci()}")
                    self.move_label.config(text=f"AI最后着法: {move.uci()}")
                    self.draw_board()
                    self.update_status()
                    if self.session.board.is_game_over():
                        self.on_game_over()
                elif msg[0] == 'review':
                    self.show_review_report(msg[1])
                elif msg[0] == 'error':
                    self.ai_thinking = False
                    self.append_log(f"错误: {msg[1]}")
        except queue.Empty:
            pass
        self.root.after(100, self.check_ai_queue)

    def on_game_over(self):
        self.game_over = True
        result = self.session.board.result()
        self.append_log(f"对局结束，结果: {result}")
        pgn = self.session.board.pgn() if hasattr(self.session.board, 'pgn') else " ".join(self.pgn_history)
        elo_history = [(self.session.elo_tracker.elo, self.session.elo_tracker.velocity)]
        try:
            self.history_mgr.save_game(pgn, self.session.move_analysis, elo_history)
            self.append_log("对局数据已保存")
        except Exception as e:
            self.append_log(f"保存对局失败: {e}")
        if self.reviewer:
            if messagebox.askyesno("云端复盘", "是否调用AI教练进行对局复盘？"):
                self.request_cloud_review()

    def update_status(self):
        elo = self.session.elo_tracker.elo
        vel = self.session.elo_tracker.velocity
        temp = self.session.temperature
        self.elo_label.config(text=f"融合Elo: {elo:.0f}\n速度: {vel:+.2f}\nAI温度: {temp:.2f}")

    def append_log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def restart_game(self):
        self.ai_thinking = True
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.5)
        self.session.cleanup()
        self.session = GameSession()
        self.game_over = False
        self.ai_thinking = False
        self.selected_square = None
        self.legal_dests = []
        self.pgn_history.clear()
        self.flip_board = self.session.ai_is_white
        self.status_label.config(text="你执白" if not self.flip_board else "你执黑")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.move_label.config(text="AI最后着法: --")
        self.draw_board()
        if self.session.ai_is_white:
            self.request_ai_move()

    def close(self):
        self.ai_thinking = True
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.5)
        self.session.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.close)
    root.mainloop()