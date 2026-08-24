"""Volume-weighted price profile: a Gaussian KDE of tick_volume by price,
identifying which price levels have seen the most historical activity.

Two entry points:
- `compute_profile` - a single snapshot over the full given df, for live
  display (the df itself already only contains data up to "now").
- `walk_forward_profiles` - a causal, expanding-window walk: at each
  checkpoint bar, the KDE and its peaks are recomputed using ONLY bars up to
  and including that bar (bandwidth included - it's a fresh Silverman's-rule
  estimate from the same causal window, not fixed upfront), so a backtest
  built on these snapshots never sees future data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_GRID_SIZE = 300
DEFAULT_N_PEAKS = 6
DEFAULT_ZONE_HALF_WIDTH_BW = 0.5   # trade zone half-width, in bandwidths
DEFAULT_MIN_SEP_BW = 1.5           # min separation between distinct peaks, in bandwidths


@dataclass
class ProfileResult:
    grid: np.ndarray
    density: np.ndarray        # normalized 0..1
    bandwidth: float
    peaks: list = field(default_factory=list)   # price levels, sorted by density desc


@dataclass
class ProfileSnapshot:
    bar_index: int
    bar_time: "pd.Timestamp"
    bandwidth: float
    peaks: list = field(default_factory=list)   # price levels, sorted by density desc


def _silverman_bandwidth(values: np.ndarray) -> float:
    n = len(values)
    std = float(values.std())
    if std <= 0 or n < 2:
        span = float(values.max() - values.min()) if len(values) else 0.0
        return max(span * 0.05, 1e-6)
    return std * n ** -0.2


def _weighted_kde(mid: np.ndarray, weights: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Vectorized weighted Gaussian KDE, chunked over the grid to bound
    peak memory use on large series."""
    density = np.zeros_like(grid)
    chunk = 50
    for start in range(0, len(grid), chunk):
        g = grid[start:start + chunk]
        diffs = (g[:, None] - mid[None, :]) / bandwidth
        kern = np.exp(-0.5 * diffs ** 2)
        density[start:start + chunk] = (kern * weights[None, :]).sum(axis=1)
    peak = density.max()
    if peak > 0:
        density = density / peak
    return density


def _detect_peaks(grid: np.ndarray, density: np.ndarray, bandwidth: float,
                   n_peaks: int, min_sep_bw: float) -> list:
    """Local maxima of density, kept greedily in descending-density order
    and filtered so no two accepted peaks sit within `min_sep_bw` bandwidths
    of each other - keeps distinct humps, drops near-duplicate shoulders of
    the same one."""
    if len(density) < 3:
        return []
    is_local_max = np.zeros(len(density), dtype=bool)
    is_local_max[1:-1] = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    candidates = sorted(np.where(is_local_max)[0], key=lambda i: -density[i])

    min_sep = bandwidth * min_sep_bw
    accepted = []
    for i in candidates:
        price = float(grid[i])
        if all(abs(price - p) >= min_sep for p in accepted):
            accepted.append(price)
        if len(accepted) >= n_peaks:
            break
    return accepted


def _price_grid(mid: np.ndarray, bandwidth: float, grid_size: int) -> np.ndarray:
    price_min, price_max = float(mid.min()), float(mid.max())
    pad = max((price_max - price_min) * 0.05, bandwidth)
    return np.linspace(price_min - pad, price_max + pad, grid_size)


def compute_profile(df: pd.DataFrame, grid_size: int = DEFAULT_GRID_SIZE,
                     n_peaks: int = DEFAULT_N_PEAKS, min_sep_bw: float = DEFAULT_MIN_SEP_BW) -> "ProfileResult | None":
    """Single current snapshot over the full given df."""
    if len(df) < 2:
        return None
    mid = ((df["high"] + df["low"]) / 2).to_numpy(float)
    vol = df["tick_volume"].to_numpy(float)
    bandwidth = _silverman_bandwidth(mid)
    grid = _price_grid(mid, bandwidth, grid_size)
    density = _weighted_kde(mid, vol, grid, bandwidth)
    peaks = _detect_peaks(grid, density, bandwidth, n_peaks, min_sep_bw)
    return ProfileResult(grid=grid, density=density, bandwidth=bandwidth, peaks=peaks)


def walk_forward_profiles(df: pd.DataFrame, recompute_every: int = 50, warmup_bars: int = 200,
                           grid_size: int = DEFAULT_GRID_SIZE, n_peaks: int = DEFAULT_N_PEAKS,
                           min_sep_bw: float = DEFAULT_MIN_SEP_BW) -> list:
    """Causal, expanding-window snapshots: at each checkpoint bar, the KDE
    and its peaks are computed using ONLY bars up to and including that bar.
    Checkpoints land every `recompute_every` bars, starting once
    `warmup_bars` bars are available. Support/resistance-style levels don't
    need per-bar precision, and recomputing a full weighted KDE at every
    single bar of a large M1 series would be far too slow."""
    n = len(df)
    if n < warmup_bars + recompute_every:
        return []

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    vol = df["tick_volume"].to_numpy(float)
    mid = (high + low) / 2
    time = df["time"].to_numpy()

    snapshots = []
    for end in range(warmup_bars, n, recompute_every):
        mid_so_far = mid[:end + 1]
        vol_so_far = vol[:end + 1]
        bandwidth = _silverman_bandwidth(mid_so_far)
        grid = _price_grid(mid_so_far, bandwidth, grid_size)
        density = _weighted_kde(mid_so_far, vol_so_far, grid, bandwidth)
        peaks = _detect_peaks(grid, density, bandwidth, n_peaks, min_sep_bw)
        snapshots.append(ProfileSnapshot(bar_index=end, bar_time=time[end], bandwidth=bandwidth, peaks=peaks))

    return snapshots
