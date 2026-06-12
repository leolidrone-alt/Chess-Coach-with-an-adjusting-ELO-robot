#!/usr/bin/env python3
"""
跨模型自对弈数据生成器
让 V3 策略头 (cond_maia_policy.onnx) 与 DeepMiniMaia (deep_policy.onnx) 对弈，
生成训练样本并保存为 chunk 文件。
支持断点续采：如果中途中断，再次运行会自动从上次结束的盘数继续。
"""

import chess
import numpy as np
import random
import os
import sys
import glob
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kits.mini_maia_ai import MiniMaiaAI
from prepare_data import board_to_planes, UCI_TO_IDX


class CrossPlayGenerator:
    def __init__(self, output_dir='data/cache/chunks'):
        # 加载两个策略模型
        self.v3_ai = MiniMaiaAI(policy_path='models/cond_maia_policy.onnx',
                                elo_path='models/deep_elo.onnx')
        self.mini_ai = MiniMaiaAI(policy_path='models/deep_policy.onnx',
                                  elo_path='models/deep_elo.onnx')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def play_game(self, elo_v3=1500, elo_mini=1500, temp_v3=0.8, temp_mini=1.2):
        """
        V3 执白，DeepMiniMaia 执黑，进行一盘对弈。
        返回该局的所有训练样本列表 [(planes, elo, move_index), ...]
        """
        board = chess.Board()
        samples = []

        while not board.is_game_over():
            current_elo = elo_v3 if board.turn == chess.WHITE else elo_mini
            current_temp = temp_v3 if board.turn == chess.WHITE else temp_mini
            current_ai = self.v3_ai if board.turn == chess.WHITE else self.mini_ai

            probs = current_ai.get_move_probs(board, target_elo=current_elo, temperature=current_temp)
            if not probs:
                break

            moves = list(probs.keys())
            p = np.array([probs[m] for m in moves])
            chosen = np.random.choice(moves, p=p)

            # 生成训练样本
            planes = board_to_planes(board)
            idx = UCI_TO_IDX.get(chosen.uci())
            if idx is not None:
                samples.append((planes, current_elo, idx))

            board.push(chosen)

            # 防止过长的对局（50回合无吃子或动兵则判和）
            if board.halfmove_clock >= 100:
                break

        return samples

    def generate_dataset(self, num_games=10000, chunk_size=10000):
        """
        生成指定数量的对局，保存为分块 .npz 文件。
        支持断点续采：自动从已有文件的最大编号继续。
        """
        # 1. 扫描已有文件，确定起始编号和已有样本
        pattern = os.path.join(self.output_dir, 'cross_play_*.npz')
        existing_files = sorted(glob.glob(pattern))
        start_game = 1
        all_planes, all_elos, all_indices = [], [], []

        if existing_files:
            # 从文件名中提取最大编号（例如 cross_play_005000.npz -> 5000）
            last_file = existing_files[-1]
            try:
                basename = os.path.basename(last_file)
                # 去掉前缀和后缀，提取数字
                num_str = basename.replace('cross_play_', '').replace('.npz', '')
                last_count = int(num_str)
                start_game = last_count + 1
                print(f"检测到已有 {last_count} 盘对局，从第 {start_game} 盘继续")
            except Exception as e:
                print(f"无法解析已有文件编号，从头开始: {e}")
                start_game = 1
                # 清理掉可能残存的旧文件（可选）
                for f in existing_files:
                    os.remove(f)

        print(f"开始跨模型自对弈：V3 (白) vs DeepMiniMaia (黑)，共 {num_games} 盘")

        # 2. 逐盘对弈
        for i in tqdm(range(start_game, num_games + 1), desc="对弈进度"):
            # 随机设定双方 Elo 和温度，增加多样性
            elo_v = random.randint(1200, 2200)
            elo_m = random.randint(1200, 2200)
            temp_v = max(0.4, 1.5 - elo_v / 1500 + random.uniform(-0.2, 0.2))
            temp_m = max(0.5, 1.8 - elo_m / 1500 + random.uniform(-0.2, 0.2))

            samples = self.play_game(elo_v, elo_m, temp_v, temp_m)
            for p, e, idx in samples:
                all_planes.append(p)
                all_elos.append(e)
                all_indices.append(idx)

            # 3. 定期保存（每 chunk_size 条样本或每 500 盘强制保存）
            if len(all_planes) >= chunk_size or (i % 500 == 0 and i > 0):
                self._save_chunk(all_planes, all_elos, all_indices, i)
                all_planes, all_elos, all_indices = [], [], []

        # 4. 保存剩余样本
        if all_planes:
            self._save_chunk(all_planes, all_elos, all_indices, num_games)

        print(f"\n生成完成！文件保存在 {self.output_dir}")

    def _save_chunk(self, planes, elos, indices, game_count):
        """将累积的样本保存为一个 .npz 文件"""
        if not planes:
            return
        planes_arr = np.array(planes, dtype=np.float32)
        elos_arr = np.array(elos, dtype=np.float32)
        indices_arr = np.array(indices, dtype=np.int64)
        filename = os.path.join(self.output_dir, f'cross_play_{game_count:06d}.npz')
        np.savez_compressed(filename, planes=planes_arr, elos=elos_arr, indices=indices_arr)
        print(f"  保存 {len(planes)} 条样本 -> {os.path.basename(filename)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='跨模型自对弈数据生成器')
    parser.add_argument('--games', type=int, default=10000, help='要生成的对局数量（默认10000）')
    parser.add_argument('--chunk', type=int, default=20000, help='每个 chunk 的样本数（默认20000）')
    args = parser.parse_args()

    gen = CrossPlayGenerator(output_dir='data/cache/chunks')
    gen.generate_dataset(num_games=args.games, chunk_size=args.chunk)