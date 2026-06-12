import chess
import chess.engine
import random
import os

from kits.mini_maia_ai import MiniMaiaAI
from kits.elo_tracker import EloTracker


def estimate_elo_by_cp_loss(board, move, engine, baseline_elo=1500, k=20):
    prev_board = board.copy()
    prev_board.pop()
    try:
        info_prev = engine.analyse(prev_board, chess.engine.Limit(time=0.05))
        best_cp = info_prev["score"].white().score(mate_score=10000) if info_prev["score"].white() is not None else 0
        info_curr = engine.analyse(board, chess.engine.Limit(time=0.05))
        curr_cp = info_curr["score"].white().score(mate_score=10000) if info_curr["score"].white() is not None else 0
    except:
        return baseline_elo, 0.0
    cp_loss = max(0, best_cp - curr_cp)
    est_elo = baseline_elo - k * cp_loss
    return max(800, min(3000, est_elo)), cp_loss


class GameSession:
    def __init__(self, stockfish_path='stockfish/stockfish-windows-x86-64-avx2.exe', random_move_prob=0.03):
        self.board = chess.Board()
        # 混合模型：V3策略头 + 8层MiniMaia Elo头
        self.ai = MiniMaiaAI()
        self.elo_tracker = EloTracker(initial_elo=1500)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        sf_path = os.path.join(base_dir, stockfish_path)
        self.sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)

        self.p_ai_white = 0.5
        self.ai_is_white = random.random() < self.p_ai_white
        self.temperature = 1.0
        self.random_move_prob = random_move_prob
        self.safety_threshold_cp = 200

    # ---------- 辅助方法 ----------
    def _sf_deep_move(self, time_limit=0.3):
        """Stockfish 深度搜索"""
        result = self.sf_engine.play(self.board,
                                     chess.engine.Limit(time=time_limit),
                                     info=chess.engine.INFO_SCORE)
        return result.move

    def _is_safe_move(self, move, depth=8, check_mate_threat=True):
        """增强安全检查：动态阈值 + 防闷杀"""
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
                        self.board.pop()  # 弹对手着法
                        self.board.pop()  # 弹AI着法
                        return False
                    self.board.pop()
        except:
            pass
        self.board.pop()
        return True

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

    def _has_passed_pawn(self):
        """检查当前局面是否存在通路兵（兼容所有 python-chess 版本）"""
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                color = piece.color

                if color == chess.WHITE:
                    # 白兵：向前推进方向（rank 增加）
                    blocked = False
                    for r in range(rank + 1, 8):
                        for f in (file - 1, file, file + 1):
                            if 0 <= f <= 7:
                                p = self.board.piece_at(chess.square(f, r))
                                if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                                    blocked = True
                                    break
                        if blocked:
                            break
                    if not blocked:
                        return True
                else:
                    # 黑兵：向前推进方向（rank 减少）
                    blocked = False
                    for r in range(rank - 1, -1, -1):
                        for f in (file - 1, file, file + 1):
                            if 0 <= f <= 7:
                                p = self.board.piece_at(chess.square(f, r))
                                if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                                    blocked = True
                                    break
                        if blocked:
                            break
                    if not blocked:
                        return True
        return False

    def play_ai_move(self):
        # ---------- 浅层搜索触发（中残局 + 战术复杂度） ----------
        if self.board.fullmove_number > 10:
            has_passed_pawn = self._has_passed_pawn()
            piece_count = len(self.board.piece_map())

            # 新增：检测王安全威胁
            king_sq = self.board.king(self.board.turn)
            attackers = self.board.attackers(not self.board.turn, king_sq)
            king_danger = len(attackers) >= 2  # 王周围有两个以上攻击子

            if has_passed_pawn or piece_count <= 20 or king_danger:
                move = self._sf_deep_move(time_limit=0.3)
                if self._is_safe_move(move) and not self._is_draw_by_repetition(move):
                    self.board.push(move)
                    return move

        # ---------- 动态温度（开局多样性） ----------
        # ---------- 动态温度（开局多样性） ----------
        move_count = self.board.fullmove_number
        piece_count = len(self.board.piece_map())

        if move_count <= 8:
            diversity_temp = 1.5
        elif 10 < move_count < 40 and piece_count > 20:
            diversity_temp = self.temperature - 0.3
        else:
            diversity_temp = self.temperature

        adjusted_temp = max(0.3, diversity_temp)

        # ----- 随机 Elo 优势 -----
        advantage = random.uniform(75, 175)
        target_elo = min(2800, self.elo_tracker.elo + advantage)

        probs = self.ai.get_move_probs(self.board,
                                       target_elo=target_elo,
                                       temperature=adjusted_temp)
        sorted_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        # ---------- 开局随机化：前6步从概率前3着法中均匀随机选择 ----------
        if move_count <= 6:
            top_k = min(3, len(sorted_moves))
            candidate_moves = [m for m, p in sorted_moves[:top_k]]
            # 确保安全，不走送子/长将/三次重复
            random.shuffle(candidate_moves)
            for move in candidate_moves:
                if (self._is_safe_move(move) and
                        not self._is_draw_by_repetition(move) and
                        not self._is_perpetual_check(move)):
                    self.board.push(move)
                    return move
            # 如果前几个都不安全，则走概率最高的那个
            move = sorted_moves[0][0]
            self.board.push(move)
            return move

        # ---------- 奇招 ----------
        if random.random() < self.random_move_prob:
            top_n = max(2, len(sorted_moves) // 10)
            candidate_moves = [m for m, p in sorted_moves[1:top_n + 1]]
            if candidate_moves:
                for move in candidate_moves:
                    if (self._is_safe_move(move) and
                            not self._is_draw_by_repetition(move) and
                            not self._is_perpetual_check(move)):
                        self.board.push(move)
                        return move
            for move, prob in sorted_moves:
                if (self._is_safe_move(move) and
                        not self._is_draw_by_repetition(move) and
                        not self._is_perpetual_check(move)):
                    self.board.push(move)
                    return move
            move = sorted_moves[0][0]
            self.board.push(move)
            return move

        # ---------- 正常选择 ----------
        for move, prob in sorted_moves:
            if (self._is_safe_move(move) and
                    not self._is_draw_by_repetition(move) and
                    not self._is_perpetual_check(move)):
                self.board.push(move)
                return move

        move = sorted_moves[0][0]
        self.board.push(move)
        return move

    def play_human_move(self, uci_str):
        try:
            move = chess.Move.from_uci(uci_str)
            if move not in self.board.legal_moves:
                return False, None, None

            board_before = self.board.copy()
            maia_elo_obs = self.ai.predict_elo(board_before)
            self.board.push(move)

            sf_elo_obs, _ = estimate_elo_by_cp_loss(
                self.board, move, self.sf_engine, baseline_elo=self.elo_tracker.elo
            )

            self.elo_tracker.update(maia_elo_obs, measurement_noise=5000)
            self.elo_tracker.update(sf_elo_obs, measurement_noise=500)
            elo = self.elo_tracker.elo
            vel = self.elo_tracker.velocity

            base_temp = max(0.5, min(1.5, 2.0 - (elo - 800) / 1200))
            vel_factor = -vel * 0.05
            self.temperature = max(0.3, min(1.8, base_temp + vel_factor))

            return True, elo, vel
        except Exception as e:
            print(f"评估出错: {e}")
            return False, None, None

    def cleanup(self):
        self.ai.quit()
        self.sf_engine.quit()


def main():
    session = GameSession(random_move_prob=0.03)

    print("=== 国际象棋教练 (V3策略+深度安全过滤+浅层搜索) ===")
    if session.ai_is_white:
        print("AI 执白，你执黑")
        move = session.play_ai_move()
        print(f"AI 首着: {move}")
    else:
        print("你执白，AI 执黑")

    print(session.board)

    while not session.board.is_game_over():
        human_turn = (session.board.turn == chess.WHITE) != session.ai_is_white

        if human_turn:
            uci = input("你的走法: ")
            if uci == 'quit':
                break
            ok, elo, vel = session.play_human_move(uci)
            if ok:
                print(f"[状态] 融合Elo: {elo:.0f}  速度: {vel:+.2f}/步  AI温度: {session.temperature:.2f}")
                print(session.board)
            else:
                print("非法走法，请重试")
        else:
            move = session.play_ai_move()
            print(f"AI 走: {move}")
            print(session.board)

    result = session.board.result()
    if session.board.is_checkmate():
        winner = "白方" if "1" in result else "黑方"
        print(f"将杀！{winner}获胜！")
    else:
        print(f"对局结束，结果: {result}")

    session.cleanup()


if __name__ == "__main__":
    main()