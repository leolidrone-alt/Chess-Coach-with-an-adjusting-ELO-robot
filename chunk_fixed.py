import numpy as np, os, glob

chunk_dir = "data/cache/value_chunks"
files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npz")))
for f in files:
    try:
        data = np.load(f)
        _ = data['planes'].shape
    except Exception as e:
        print(f"损坏文件: {f} - {e}")