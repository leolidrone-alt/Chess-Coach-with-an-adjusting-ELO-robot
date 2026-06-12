import chess
import chess.syzygy
import numpy as np
import random
import math
import time

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            # 经典 UCB 公式：平均价值 + 探索项
            exploit = child.value_sum / (child.visit_count + 1e-8)
            explore = c_puct * child.prior * math.sqrt(self.visit_count + 1) / (1 + child.visit_count)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class MCTS:
    def __init__(self, policy_model, elo_model, stockfish_engine=None, tablebase=None, value_model=None):
        self.policy_model = policy_model
        self.elo_model = elo_model
        self.sf_engine = stockfish_engine
        self.tablebase = tablebase
        self.value_model = value_model  # 预留，暂未使用

    def _evaluate_board(self, board):
        """高质量局面评估：优先表库 → Stockfish 深度8 → Elo头"""
        # 1. Syzygy 表库 (绝对精确)
        if self.tablebase and len(board.piece_map()) <= 5:
            try:
                wdl = self.tablebase.probe_wdl(board)
                if wdl == 2: return 1.0
                elif wdl == -2: return 0.0
                else: return 0.5
            except Exception:
                pass

        # 2. Stockfish 深度搜索 (depth=8，比固定时间更稳定)
        if self.sf_engine:
            try:
                info = self.sf_engine.analyse(board, chess.engine.Limit(depth=8, time=0.05))
                score = info["score"].white()
                if score.is_mate():
                    return 0.0 if score.mate() < 0 else 1.0
                cp = score.score()
                win_prob = 1.0 / (1.0 + math.exp(-cp / 400.0))
                # 返回当前走棋方视角的胜率
                return win_prob if board.turn == chess.WHITE else 1.0 - win_prob
            except Exception:
                pass

        # 3. 回退到 Elo 头 (最不可靠)
        try:
            elo = self.elo_model.predict_elo(board)
            advantage = (elo - 1500) * 0.005
            win_prob = 1.0 / (1.0 + math.exp(-advantage))
            return win_prob if board.turn == chess.WHITE else 1.0 - win_prob
        except:
            return 0.5

    def search(self, board, time_limit=2.5, c_puct=1.0, dirichlet_alpha=0.3, dirichlet_frac=0.25):
        root = MCTSNode(board)
        # 获取策略先验概率
        try:
            probs = self.policy_model.get_move_probs(board, target_elo=1500, temperature=1.0)
        except:
            probs = None

        # 为根节点创建子节点，设置先验
        for move in board.legal_moves:
            prior = probs.get(move, 0.001) if probs else 1.0 / len(list(board.legal_moves))
            root.children.append(MCTSNode(board, parent=root, move=move, prior=prior))

        # 增强的 Dirichlet 噪声，增加探索
        if dirichlet_alpha > 0 and len(root.children) > 1:
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = (1 - dirichlet_frac) * child.prior + dirichlet_frac * noise[i]

        # 自适应时间：局面越复杂，给的时间越多
        move_count = len(board.legal_moves)
        piece_count = len(board.piece_map())
        if piece_count <= 6:
            time_limit = min(time_limit, 1.0)   # 残局可以快一点，但要保证足够模拟
        elif move_count > 30:
            time_limit = max(time_limit, 3.0)   # 复杂局面多给时间

        # 最小模拟次数，防止过早中断
        MIN_SIMULATIONS = 500
        start_time = time.time()
        simulations = 0

        # 主循环
        while simulations < MIN_SIMULATIONS or time.time() - start_time < time_limit:
            node = root
            # 1. 选择
            while node.is_fully_expanded() and node.children:
                node = node.best_child(c_puct)

            # 2. 扩展与模拟
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                new_board = node.board.copy()
                new_board.push(move)
                # 为新节点设置更合理的先验（取自策略网络，如果可用）
                prior = 0.001
                if probs:
                    prior = probs.get(move, 0.001)
                child = MCTSNode(new_board, parent=node, move=move, prior=prior)
                node.children.append(child)
                node = child

            # 3. 模拟：评估叶子节点
            value = self._evaluate_board(node.board)

            # 4. 回溯
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = 1.0 - value  # 切换视角
                node = node.parent

            simulations += 1

        # 返回访问次数分布
        move_probs = {}
        total_visits = sum(child.visit_count for child in root.children)
        if total_visits > 0:
            for child in root.children:
                move_probs[child.move] = child.visit_count / total_visits
        else:
            for child in root.children:
                move_probs[child.move] = 1.0 / len(root.children)
        return move_probs