import numpy as np
import os

chunk_dir = "data/cache/chunks"
for fname in sorted(os.listdir(chunk_dir))[:1]:  # 只检查第一个块
    data = np.load(os.path.join(chunk_dir, fname))
    indices = data['indices']
    print(f"索引最小值: {indices.min()}, 最大值: {indices.max()}")
    print(f"索引中负数个数: {(indices < 0).sum()}")
    print(f"索引中超出4199的个数: {(indices >= 4208).sum()}")
    print(f"Elo 最小值: {data['elos'].min()}, 最大值: {data['elos'].max()}")
    break