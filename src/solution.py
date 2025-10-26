import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _rolling_percentile_rank(codes, values, window, n_symbols):
    """Compute rolling percentile ranks per symbol code in a single pass."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out

    win = window if window > 1 else 1
    # Circular buffers keep the last `win` closes per symbol.
    buffers = np.empty((n_symbols, win), dtype=np.float32)
    sizes = np.zeros(n_symbols, dtype=np.int32)
    positions = np.zeros(n_symbols, dtype=np.int32)

    for i in range(n):
        code = codes[i]
        val = values[i]

        pos = positions[code]
        buffers[code, pos] = val

        size = sizes[code]
        if size < win:
            size += 1
            sizes[code] = size
        else:
            size = win

        pos += 1
        if pos == win:
            pos = 0
        positions[code] = pos

        valid_len = size
        start = pos - valid_len
        if start < 0:
            start += win

        count = 0
        for k in range(valid_len):
            idx = start + k
            if idx >= win:
                idx -= win
            if buffers[code, idx] <= val:
                count += 1

        out[i] = count / valid_len

    return out


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    close_values = df["Close"].to_numpy(dtype=np.float32, copy=False)
    codes, _ = pd.factorize(df["symbol"], sort=False)
    if (codes < 0).any():
        raise ValueError("symbol column contains missing values")

    codes = codes.astype(np.int64, copy=False)
    n_symbols = int(codes.max()) + 1

    ranks = _rolling_percentile_rank(codes, close_values, int(window), n_symbols)
    return ranks.reshape(-1, 1)