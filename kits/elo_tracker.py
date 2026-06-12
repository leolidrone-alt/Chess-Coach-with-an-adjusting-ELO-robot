"""卡尔曼滤波器：融合多观测源，输出平滑 Elo + 速度 v（升级版）"""
import numpy as np
from filterpy.kalman import KalmanFilter

class EloTracker:
    def __init__(self, initial_elo=1500):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([initial_elo, 0.])
        self.initial_elo = initial_elo

        self.kf.F = np.array([[1., 1.],
                              [0., 1.]])
        self.kf.H = np.array([[1., 0.]])

        # 默认过程噪声（稳定追踪）
        self.default_Q = np.array([[0.1, 0.],
                                   [0., 0.01]])
        self.fast_Q = np.array([[1.0, 0.],
                                [0., 0.05]])   # 开局快速收敛

        self.kf.Q = self.fast_Q   # 开局用快速收敛
        self.kf.R = np.array([[500.]])
        self.kf.P = np.array([[250000., 0.],
                              [0., 100.]])

        self.move_count = 0   # 内部步数计数器

    def update(self, observation: float, measurement_noise=None,
               board_complexity=None, cp_loss=None):
        """
        输入一次观测，返回平滑后的 (elo, velocity)。
        measurement_noise: 外部指定的观测噪声（若提供，则覆盖自适应）
        board_complexity: 局面复杂度字典（可选），包含 'piece_count', 'legal_moves'
        cp_loss: 当前步的 centipawn 损失，用于自适应噪声和离群值检测
        """
        self.move_count += 1

        # ---- 1. 自适应观测噪声 ----
        if measurement_noise is None:
            noise = self._adaptive_measurement_noise(board_complexity, cp_loss)
        else:
            noise = measurement_noise
        self.kf.R = np.array([[noise]])

        # ---- 2. 离群值检测 ----
        predicted_elo = self.kf.x[0]
        predicted_std = np.sqrt(self.kf.P[0, 0])
        if abs(observation - predicted_elo) > 3 * predicted_std:
            # 观测值偏离超过3倍标准差，大幅降低信任度
            self.kf.R = np.array([[5000.0]])

        # ---- 3. 开局快速收敛 ----
        if self.move_count <= 5:
            self.kf.Q = self.fast_Q
        else:
            self.kf.Q = self.default_Q

        # 执行预测与更新
        self.kf.predict()
        self.kf.update(np.array([observation]))
        return self.kf.x[0], self.kf.x[1]

    def _adaptive_measurement_noise(self, board_complexity, cp_loss):
        """根据局面复杂度和损失幅度动态计算观测噪声"""
        base_noise = 500.0

        if board_complexity:
            piece_count = board_complexity.get('piece_count', 32)
            legal_moves = board_complexity.get('legal_moves', 30)

            if piece_count <= 12:
                base_noise *= 0.5
            elif piece_count <= 20:
                base_noise *= 0.7

            if legal_moves > 30:
                base_noise *= 1.5

        if cp_loss is not None and cp_loss > 300:
            base_noise *= 2.0

        return max(100.0, min(5000.0, base_noise))

    @property
    def elo(self):
        return self.kf.x[0]

    @property
    def velocity(self):
        return self.kf.x[1]