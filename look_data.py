"""
查看npz形状
"""
import numpy as np

# 1. 改成你实际的 npz 路径
npz_path = "data/fullres/try/8.npz"

# 2. 读取 npz 文件
with np.load(npz_path) as f:
    data = f["data"]
    seg = f["seg"]

    print("data shape:", data.shape, "dtype:", data.dtype)
    print("seg  shape:", seg.shape,  "dtype:", seg.dtype)
