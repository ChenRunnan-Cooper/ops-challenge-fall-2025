import polars as pl
import numpy as np
from numba import jit, prange, config
import warnings

# 优化 Numba 配置
config.THREADING_LAYER = 'workqueue'

@jit(nopython=True, fastmath=True, nogil=True, parallel=True, inline='always')
def _rolling_rank_parallel(values, window, out):
    """
    并行版本的滚动排名计算
    
    优化点：
    1. parallel=True: 并行处理不同的行
    2. inline='always': 强制内联以减少函数调用开销
    3. prange: 并行循环
    
    注意：移除了 cache=True 以避免与 importlib 动态加载冲突
    """
    n = len(values)
    
    # 并行处理每一行
    for i in prange(n):
        start_idx = max(0, i - window + 1)
        window_size = i - start_idx + 1
        
        current_val = values[i]
        rank_sum = 0.0
        
        # 内层循环：计算排名
        # 使用 <= 比较，相等的值也算在内
        for j in range(start_idx, i + 1):
            if values[j] <= current_val:
                rank_sum += 1.0
        
        out[i] = rank_sum / window_size
    
    return out


@jit(nopython=True, fastmath=True, nogil=True)
def _rolling_rank_optimized(values, window, out):
    """
    优化的串行版本（用于小数据集）
    
    优化点：
    1. 移除不必要的条件检查
    2. 使用更紧凑的循环
    3. 更好的内存访问模式
    
    注意：移除了 cache=True 以避免与 importlib 动态加载冲突
    """
    n = len(values)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_size = i - start_idx + 1
        
        current_val = values[i]
        rank_sum = 0.0
        
        for j in range(start_idx, i + 1):
            # 使用 <= 而不是复杂的 NaN 检查
            # Polars 已经处理了 NaN
            rank_sum += (values[j] <= current_val)
        
        out[i] = rank_sum / window_size
    
    return out


class ops:
    @staticmethod
    def rolling_rank(col_or_expr, window: int) -> pl.Expr:
        """
        自定义滚动排名函数
        
        优化策略：
        1. 根据数据大小自动选择并行或串行版本
        2. 使用 Float32 减少内存带宽
        3. 预分配输出数组
        """
        def rolling_rank(s: pl.Series) -> pl.Series:
            values = s.to_numpy()
            n = len(values)
            result = np.empty(n, dtype=np.float32)
            
            # 根据数据大小选择算法
            # 大数据集使用并行版本，小数据集使用优化串行版本
            if n > 10000:
                _rolling_rank_parallel(values, window, result)
            else:
                _rolling_rank_optimized(values, window, result)
            
            return result
        
        if isinstance(col_or_expr, str):
            expr = pl.col(col_or_expr)
        else:
            expr = col_or_expr
        
        return expr.map_batches(rolling_rank)


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    """
    最快版本的滚动排名计算
    
    优化点：
    1. 使用 scan_parquet 懒加载
    2. 提前转换为 Float32
    3. 使用 Polars 的 over 进行高效分组
    4. 使用并行 Numba 加速核心计算
    5. 流式处理，减少内存占用
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        result = (
            pl.scan_parquet(input_path)
            .with_columns(
                # 提前转换类型，减少后续转换开销
                pl.col("Close").cast(pl.Float32).alias("Close")
            )
            .select(
                # 使用 over 进行分组计算
                # Polars 会自动并行处理不同的组
                ops.rolling_rank("Close", window).over("symbol")
            )
        ).collect(streaming=True)  # 使用流式处理
        
        return result.to_numpy()

