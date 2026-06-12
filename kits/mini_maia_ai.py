"""
MiniMaia 双头模型推理接口
加载 ONNX 模型，提供：
- 走子概率分布（含温度）
- 按概率采样一步棋
- 瞪眼法 Elo 评估
"""

import numpy as np
import onnxruntime as ort
import chess
import os
import sys

# 导入项目根目录下的 prepare_data 中的映射和平面生成函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_data import UCI_TO_IDX, board_to_planes


class MiniMaiaAI:
    def __init__(self, policy_path='models/cond_maia_policy.onnx', elo_path='models/cond_maia_elo.onnx'):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.policy_sess = ort.InferenceSession(os.path.join(base_dir, policy_path))
        self.elo_sess = ort.InferenceSession(os.path.join(base_dir, elo_path))
        self.input_policy = self.policy_sess.get_inputs()[0].name
        self.input_elo = self.elo_sess.get_inputs()[0].name

    def _get_elo_plane(self, target_elo: float) -> np.ndarray:
        """根据目标 Elo 生成与训练时一致的 [-1, 1] 平面"""
        # 训练时 Elo 归一化：elo_norm = (elo - 1100) / (2400 - 1100)
        # 然后映射到 [-1, 1]：plane_value = elo_norm * 2 - 1
        elo_norm = (target_elo - 1100.0) / 1300.0
        elo_val = elo_norm * 2.0 - 1.0
        return np.full((1, 1, 8, 8), elo_val, dtype=np.float32)

    def get_move_probs(self, board: chess.Board, target_elo: float = 1500, temperature: float = 1.0) -> dict:
        """
        返回所有合法走法的概率分布（dict: Move -> prob）。
        temperature = 0 时，返回确定性策略（只选最佳着法）。
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        # 构造输入
        planes = board_to_planes(board)  # (112, 8, 8)
        board_input = np.expand_dims(planes, axis=0)  # (1, 112, 8, 8)
        elo_plane = self._get_elo_plane(target_elo)

        # 策略推理
        logits = self.policy_sess.run(None, {self.input_policy: board_input, 'elo_plane': elo_plane})[0][0]  # (4208,)

        # 构建合法走法掩码
        masked_logits = np.full_like(logits, -1e9)
        for move in legal_moves:
            uci = move.uci()
            idx = UCI_TO_IDX.get(uci)
            if idx is not None:
                masked_logits[idx] = logits[idx]

        # ---- 优化后的温度处理 ----
        if temperature < 0:  # 负温度无意义，自动修正为 1.0
            temperature = 1.0
        elif temperature == 0:  # 确定性策略：直接选 logit 最大的着法
            best_idx = np.argmax(masked_logits)
            move_probs = {}
            for move in legal_moves:
                idx = UCI_TO_IDX.get(move.uci())
                move_probs[move] = 1.0 if idx == best_idx else 0.0
            return move_probs
        else:
            # 温度 > 0：标准 softmax 采样
            exp_logits = np.exp((masked_logits - np.max(masked_logits)) / temperature)
            probs = exp_logits / exp_logits.sum()

            move_probs = {}
            for move in legal_moves:
                idx = UCI_TO_IDX.get(move.uci())
                move_probs[move] = probs[idx] if idx is not None else 0.0

        # 重新归一化（处理浮点误差）
        total = sum(move_probs.values())
        if total > 0:
            for m in move_probs:
                move_probs[m] /= total
        else:
            for m in legal_moves:
                move_probs[m] = 1.0 / len(legal_moves)

        return move_probs

    def sample_move(self, board: chess.Board, target_elo: float = 1500, temperature: float = 1.0) -> chess.Move:
        """根据策略概率和温度采样一步棋"""
        probs = self.get_move_probs(board, target_elo, temperature)
        moves = list(probs.keys())
        p = np.array([probs[m] for m in moves])
        return np.random.choice(moves, p=p)

    def predict_elo(self, board: chess.Board) -> float:
        """瞪眼法：输入走棋前的局面，返回估计 Elo（反归一化后的真实值）"""
        planes = board_to_planes(board)
        board_input = np.expand_dims(planes, axis=0)
        elo = self.elo_sess.run(None, {self.input_elo: board_input})[0][0]
        return float(elo)

    def quit(self):
        # ONNX Runtime 无需显式释放
        pass


if __name__ == "__main__":
    # 简单测试
    b = chess.Board()
    ai = MiniMaiaAI()
    print("Elo 估计:", ai.predict_elo(b))
    print("e2e4 概率:", ai.get_move_probs(b, temperature=1.0).get(chess.Move.from_uci("e2e4"), 0))