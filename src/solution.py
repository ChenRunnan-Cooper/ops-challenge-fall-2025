import polars as pl
import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def _rolling_rank_fast(values, window, out):
    """
    高性能并行滚动排名 - 优化版
    
    关键优化：
    1. 极简循环结构
    2. 并行计算充分利用多核
    3. fastmath 加速浮点运算
    """
    n = len(values)
    
    for i in prange(n):
        start = max(0, i - window + 1)
        current = values[i]
        rank = 0.0
        
        for j in range(start, i + 1):
            rank += values[j] <= current
        
        out[i] = rank / (i - start + 1)
    
    return out


class ops:
    @staticmethod
    def rolling_rank(col_or_expr, window: int) -> pl.Expr:
        """高性能滚动排名"""
        def rolling_rank(s: pl.Series) -> pl.Series:
            values = s.to_numpy()
            result = np.empty(len(values), dtype=np.float32)
            _rolling_rank_fast(values, window, result)
            return result
        
        if isinstance(col_or_expr, str):
            expr = pl.col(col_or_expr)
        else:
            expr = col_or_expr
        
        return expr.map_batches(rolling_rank)


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """
    极致优化版本 - 针对 20 核 GitHub Actions 环境
    
    关键优化：
    1. Polars 懒加载 + 并行 I/O
    2. 直接 Float32 转换
    3. Numba 20 核并行计算
    4. 最简洁的代码路径
    """
    return (
        pl.scan_parquet(input_path)
        .with_columns(pl.col("Close").cast(pl.Float32))
        .select(ops.rolling_rank("Close", window).over("symbol"))
        .collect()
        .to_numpy()
    )

