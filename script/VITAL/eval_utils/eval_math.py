import numpy as np
from typing import Sequence, Union, Optional, Dict, List
import pandas as pd
import ruptures as rpt
from scipy.signal import periodogram, savgol_filter
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL







# ----------------- nicu hr attributes -------------------------------------------
def successive_increases(y):
    """
    Input: y - numeric array/list of measurements
    Output: dictionary containing counts and proportions of successive increases
    """
    # Check length
    N = len(y)
    if N < 5:
        print("Warning: Time series too short")
        return None
    
    # Get binary sequence (1 for increase, 0 for decrease/same)
    y_diff = np.diff(y)
    y_bin = (y_diff > 0).astype(int)
    
    # Initialize output dictionary
    out = {}
    
    # Single increases (length 1)
    out['u'] = np.mean(y_bin == 1)  # proportion of increases
    out['d'] = np.mean(y_bin == 0)  # proportion of decreases/same
    
    # Two consecutive (length 2)
    out['uu'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 1))  # up,up
    out['ud'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 0))  # up,down
    out['du'] = np.mean((y_bin[:-1] == 0) & (y_bin[1:] == 1))  # down,up
    out['dd'] = np.mean((y_bin[:-1] == 0) & (y_bin[1:] == 0))  # down,down
    
    # Three consecutive (length 3)
    out['uuu'] = np.mean((y_bin[:-2] == 1) & (y_bin[1:-1] == 1) & (y_bin[2:] == 1))
    
    # Four consecutive (length 4)
    out['uuuu'] = np.mean((y_bin[:-3] == 1) & (y_bin[1:-2] == 1) & (y_bin[2:-1] == 1) & (y_bin[3:] == 1))
    
    # Five consecutive (length 5)
    out['uuuuu'] = np.mean((y_bin[:-4] == 1) & (y_bin[1:-3] == 1) & (y_bin[2:-2] == 1) & (y_bin[3:-1] == 1) & (y_bin[4:] == 1))
    
    # Calculate entropy for each length
    def calc_entropy(p):
        # Remove zeros and calculate entropy
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    # Entropy calculations
    out['h'] = calc_entropy(np.array([out['u'], out['d']]))  # entropy of length 1
    out['h2'] = calc_entropy(np.array([out['uu'], out['ud'], out['du'], out['dd']]))  # entropy of length 2
    

    umean = out['uuu']*100 * 3 + out['uuuu']*100 * 4 + out['uuuuu']*100 * 5
    return umean

# Example usage:
# y = [1, 2, 1, 3, 2, 4, 5]
# result = successive_increases(y)

def successive_unchanges(y):
    """
    Input: y - numeric array/list of measurements
    Output: dictionary containing counts and proportions of successive unchanges
    """
    # Check length
    N = len(y)
    if N < 5:
        print("Warning: Time series too short")
        return None
    
    # Get binary sequence (1 for no change, 0 otherwise)
    y_diff = np.diff(y)
    y_bin = (y_diff == 0).astype(int)
    
    # Initialize output dictionary
    out = {}
    
    # Single unchanges (length 1)
    out['u'] = np.mean(y_bin == 1)  # proportion of unchanges
    
    # Two consecutive unchanges (length 2)
    n2 = len(y_bin) - 1
    out['uu'] = np.mean((y_bin[:-1] == 1) & (y_bin[1:] == 1))  # unchange,unchange
    
    # Three consecutive (length 3)
    n3 = len(y_bin) - 2
    out['uuu'] = np.mean((y_bin[:-2] == 1) & (y_bin[1:-1] == 1) & (y_bin[2:] == 1))
    
    # Four consecutive (length 4)
    n4 = len(y_bin) - 3
    out['uuuu'] = np.mean((y_bin[:-3] == 1) & (y_bin[1:-2] == 1) & (y_bin[2:-1] == 1) & (y_bin[3:] == 1))
    
    # Five consecutive (length 5)
    n5 = len(y_bin) - 4
    out['uuuuu'] = np.mean((y_bin[:-4] == 1) & (y_bin[1:-3] == 1) & (y_bin[2:-2] == 1) & (y_bin[3:-1] == 1) & (y_bin[4:] == 1))
    
    umean = out['uuu']*100 * 3 + out['uuuu']*100 * 4 + out['uuuuu']*100 * 5
    return umean

# Example usage:
# y = [1, 1, 1, 2, 2, 3, 3, 3]
# result = successive_unchanges(y)


# ----------------- synthetic attributes -------------------------------------------
# ------------------------------------------------------ 1.1 Trend linear slope ----------------------------------------------------------
# def trend_slope(x):
#     """OLS slope beta (trend strength & sign)."""
#     t = np.arange(len(x))
#     return sm.OLS(x, sm.add_constant(t)).fit().params[1]

# single‑window OLS slope (unchanged)
def trend_slope(x):
    t = np.arange(len(x), dtype=float)
    beta = np.polyfit(t, x, 1)      # β1 is slope
    return beta[0]

# ------------------------------------------------------------
# multi‑scale robust slope
# ------------------------------------------------------------
def trend_slope_moving_window(
    x,
    *,
    lengths=(10, 20, 30),     # list/tuple of window sizes
    step=5
) -> float:
    """
    Robust global trend slope from overlapping windows of
    multiple lengths.

    Parameters
    ----------
    x        : array‑like
    lengths  : iterable of int
        Window lengths to use (all ≥3 and ≤ len(x)).
    step     : int
        Hop size between windows.

    Returns
    -------
    float
        Median slope after IQR outlier removal.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < min(lengths, default=3):
        return trend_slope(x)

    slopes = []
    for L in lengths:
        if L < 3 or L > n:
            continue
        for i in range(0, n - L + 1, step):
            s = trend_slope(x[i : i + L])
            if np.isfinite(s):
                slopes.append(s)

    if not slopes:
        return 0.0

    slopes = np.array(slopes)
    q1, q3 = np.percentile(slopes, [25, 75])
    iqr = q3 - q1
    mask = (slopes >= q1 - 1.5 * iqr) & (slopes <= q3 + 1.5 * iqr)
    clean = slopes[mask] if mask.any() else slopes

    return float(np.median(clean))

# ------------------------------------------------------ 1.2 Trend curvature ----------------------------------------------------------
def _central_gradient(y: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Central‑difference first and second derivatives (length = len(y)‑2).
    """
    d1 = (y[2:] - y[:-2]) / (2 * dx)
    d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / (dx ** 2)
    return d1, d2


def calculate_curvature(
    series: Sequence[Union[int, float]],
    *,
    method: str = "quadratic_fit",
    dx: float = 1.0,
    smooth: Optional[int] = None,
) -> float:
    """
    Curvature (signed) of a 1‑D time series.

    Parameters
    ----------
    series : 1‑D iterable
    method : {"quadratic_fit", "second_derivative", "radius"}
    dx     : sampling interval (default 1.0)
    smooth : optional odd window length for Savitzky–Golay pre‑smoothing

    Returns
    -------
    float
        quadratic_fit / second_derivative:
            sign >0 → convex (upward), <0 → concave (downward)
        radius:
            positive magnitude only (multiply by sign if wanted)
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    if y.size < 3 or np.isclose(y.var(), 0):
        return 0.0

    if smooth is not None and smooth >= 5 and smooth % 2 == 1:
        y = savgol_filter(y, window_length=smooth, polyorder=3)

    # -------- 1) robust quadratic‑fit curvature -----------------------------
    if method == "quadratic_fit":
        t = np.linspace(-1.0, 1.0, len(y))             # centred & scaled time
        X = np.column_stack((np.ones_like(t), t, t**2))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(beta[2])                          # signed curvature term

    # -------- 2) median second derivative (scaled) --------------------------
    elif method == "second_derivative":
        _, d2 = _central_gradient(y, dx)
        return float(np.median(d2))

    # -------- 3) differential‑geometry curvature κ = |y''|/(1+y'^2)^{3/2} ---
    elif method == "radius":
        d1, d2 = _central_gradient(y, dx)
        kappa = np.abs(d2) / ((1.0 + d1**2) ** 1.5 + 1e-12)
        return float(np.median(kappa))                 # magnitude only

    else:
        raise ValueError(f"Unknown curvature method '{method}'")


# ---------------------------------------------------------------------------
# Moving‑window wrapper
# ---------------------------------------------------------------------------
def trend_curvature_moving_window(
    x: Sequence[Union[int, float]],
    *,
    length: int = 50,
    step: int = 10,
    method: str = "quadratic_fit",
    dx: float = 1.0,
    smooth: Optional[int] = 10,
) -> float:
    """
    Robust global curvature estimate via overlapping windows + IQR de‑spike.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]

    if len(x) < length:
        return calculate_curvature(x, method=method, dx=dx, smooth=smooth)

    curvs: list[float] = []
    for i in range(0, len(x) - length + 1, step):
        w = x[i : i + length]
        c = calculate_curvature(w, method=method, dx=dx, smooth=smooth)
        if np.isfinite(c):
            curvs.append(c)

    if not curvs:
        return 0.0

    curvs = np.array(curvs)
    q1, q3 = np.percentile(curvs, [25, 75])
    iqr = q3 - q1
    mask = (curvs >= q1 - 1.5 * iqr) & (curvs <= q3 + 1.5 * iqr)
    clean = curvs[mask] if mask.any() else curvs

    return float(np.median(clean))





# ------------------------------------------------------ 2. Seasonality strength ---------------------------------------------------------
# ------------------------------------------------------------------
# helper – estimate fundamental period (integer ≥ 2)
# ------------------------------------------------------------------
def _estimate_period(x, fs=1.0):
    """
    Return integer period m̂ from the strongest spectral peak.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    f, P = periodogram(x, fs=fs, window='hann', detrend='linear')
    f, P = f[1:], P[1:]                     # drop DC
    if len(P) == 0:
        return 1                           # degenerate fallback
    m_hat = int(round(1.0 / f[np.argmax(P)]))
    # keep in a reasonable range
    return max(2, min(m_hat, len(x) // 2))

# ------------------------------------------------------------------
# 1. ACF-based strength  (average of harmonics m,2m,3m,…)
# ------------------------------------------------------------------
def seasonality_strength_acf(series, m=None, fs=1.0, nlags=None):
    """
    Strength = mean positive PACF values at lags k·m, k = 1,2,...
    If m is None it is auto-estimated.
    """
    x = np.asarray(series, dtype=float) - np.mean(series)
    if m is None:
        m = _estimate_period(x, fs)
    if nlags is None:
        nlags = min(4 * m, len(x) - 1)  # Ensure nlags doesn't exceed series length
    r = acf(x, nlags=nlags, fft=True)
    ks = np.arange(1, nlags // m + 1)
    if len(ks) == 0:  # Handle case where nlags < m
        return 0.0
    indices = ks * m
    indices = indices[indices < len(r)]  # Ensure indices are within bounds
    if len(indices) == 0:
        return 0.0
    vals = r[indices]
    vals = vals[vals > 0]  # ignore negative correlations
    return float(vals.mean()) if vals.size else 0.0


# ------------------------------------------------------------------
# 2. STL / X-11 strength (Hyndman & Athanasopoulos)
# ------------------------------------------------------------------
def seasonality_strength_stl(series, m=None, fs=1.0):
    """
    S = 1 – Var(R) / Var(R+S)   where R = remainder, S = seasonal component.
    If m is None it is auto-estimated.
    """
    x = np.asarray(series, dtype=float)
    if m is None:
        m = _estimate_period(x, fs)
    res = STL(x, period=m, robust=True).fit()
    s, r = res.seasonal, res.resid
    denom = np.var(r + s, ddof=1)
    return 0.0 if denom == 0 else 1.0 - np.var(r, ddof=1) / denom

# ------------------------------------------------------------------
# 3. Spectral–power-ratio (periodogram-based) seasonality 
# (Pereira et al., Information, 2022)
# ------------------------------------------------------------------
def seasonality_strength_spectrum(series, fs=1.0, m_hint=None, bw=1):
    """
    Fraction of spectral power in the dominant seasonal peak.
    If m_hint is given (length of season) we integrate a small band
    around that frequency; else we take the highest non-DC peak.
    """
    f, P = periodogram(series, fs=fs, window='hann', detrend='linear',
                       scaling='spectrum')
    # Drop DC
    f, P = f[1:], P[1:]
    if m_hint is not None:
        target = 1.0 / m_hint
        idx = np.argmin(np.abs(f - target))
    else:
        idx = np.argmax(P)
    # integrate ±bw bins
    lo, hi = max(idx-bw, 0), min(idx+bw+1, len(P))
    return P[lo:hi].sum() / P.sum()

# ------------------------------------------------------------------
# 4. spectral entropy
# (Martínez-Granados et al., Entropy, 2020) https://www.mdpi.com/1099-4300/22/1/89?utm_source=chatgpt.com
# ------------------------------------------------------------------
# def seasonality_spectral_entropy(series, fs=1.0):
#     """
#     Strength = 1 – normalized spectral entropy.
#     0: white noise, 1: single pure tone.
#     """
#     _, P = periodogram(series, fs=fs, window='hann', detrend='linear')
#     P = P[1:]                       # drop DC
#     P /= P.sum()
#     ent = -np.sum(P * np.log(P + 1e-12)) / np.log(len(P))
#     return 1.0 - ent                # map to [0,1]

def seasonality_spectral_entropy(
    series,
    *,
    fs: float = 1.0,
    smooth_bw: int = 0,
    lowpass: float = 1.0,
    floor_db: float | None = None,
):
    """
    Seasonality strength = 1 – normalised spectral entropy
    (with optional de‑noising).

    Parameters
    ----------
    smooth_bw : int
        Half‑width (in FFT bins) of a centred box filter applied to |X(f)|^2.
        0 → no smoothing.
    lowpass : float in (0,1]
        Keep frequencies <= lowpass * Nyquist.  e.g. 0.5 keeps only slow half.
    floor_db : float or None
        Subtract this dB floor (w.r.t. max power) before entropy.
        Values below floor are clamped to 0.

    Returns
    -------
    float  in [0,1]   higher ⇒ stronger, cleaner seasonality
    """
    x = np.asarray(series, float) - np.mean(series)
    f, P = periodogram(x, fs=fs, window="hann", detrend="linear")

    # ---- 0. drop DC --------------------------------------------------------
    f, P = f[1:], P[1:]

    if P.size == 0:
        return 0.0

    # ---- 1. optional smoothing (Welch overlap‑add in frequency domain) -----
    if smooth_bw > 0:
        k = 2 * smooth_bw + 1
        P = np.convolve(P, np.ones(k) / k, mode="same")

    # ---- 2. optional low‑pass focus ----------------------------------------
    if 0 < lowpass < 1.0:
        nyq = fs / 2.0
        keep = f <= lowpass * nyq
        P, f = P[keep], f[keep]

    # ---- 3. optional noise‑floor suppression -------------------------------
    if floor_db is not None:
        P_db = 10 * np.log10(P + 1e-24)
        thresh = P_db.max() - abs(floor_db)
        P = np.where(P_db < thresh, 0.0, P)

    # ---- 4. entropy → strength --------------------------------------------
    if P.sum() == 0:
        return 0.0
    P /= P.sum()
    ent = -np.sum(P * np.log(P + 1e-12)) / np.log(len(P))
    return 1.0 - ent

def get_seasonality_scores(series, fs=1.0):
    """
    Calculate all four seasonality strength scores for a time series.
    
    Returns:
        dict: Dictionary containing four seasonality scores:
            - acf: ACF-based strength (0-1)
            - stl: STL-based strength (0-1)
            - spectrum: Spectral power ratio (0-1)
            - entropy: Spectral entropy-based strength (0-1)
    """
    m = _estimate_period(series, fs)
    return {
        'acf': np.round(seasonality_strength_acf(series, m=m, fs=fs), 4),
        'stl': np.round(seasonality_strength_stl(series, m=m, fs=fs), 4),
        'spectrum': np.round(seasonality_strength_spectrum(series, fs=fs, m_hint=m), 4),
        'entropy': np.round(seasonality_spectral_entropy(series, fs=fs, smooth_bw=3, lowpass=0.8), 4)
    }

# ------------------------------------------------------------
# moving‑window seasonality wrapper
# ------------------------------------------------------------
def get_seasonality_scores_moving_window(
    series: Sequence[Union[int, float]],
    *,
    length: int = 100,
    step: int = 5,
    fs: float = 1.0,
) -> Dict[str, float]:
    """
    Robust seasonality strength over a series via overlapping windows.

    Parameters
    ----------
    series : array‑like
    length : int
        Window size (default 50 samples).
    step   : int
        Hop between window starts (default 5 samples).
    fs     : float
        Sampling frequency passed through to the inner scorers.

    Returns
    -------
    dict
        Keys = {"acf", "stl", "spectrum", "entropy"}.
        Each value is the median score after IQR de‑spiking.
    """
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < length:
        return get_seasonality_scores(x, fs=fs)

    # collect per‑window scores
    metrics = {"acf": [], "stl": [], "spectrum": [], "entropy": []}
    for start in range(0, n - length + 1, step):
        win = x[start : start + length]
        s = get_seasonality_scores(win, fs=fs)
        for k in metrics:
            if np.isfinite(s[k]):
                metrics[k].append(s[k])

    # IQR de‑spike + median aggregation
    def robust_median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        v = np.array(vals)
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        keep = (v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)
        v_clean = v[keep] if keep.any() else v
        return float(np.median(v_clean))

    return {k: robust_median(v) for k, v in metrics.items()}



# ------------------------------------------------------ 3. Shift of mean ---------------------------------------------------------------
# --------------------------------------------------------------------
# 1) local‑smoother detrend  (keeps flat‑to‑flat step intact) 
# avoids turning a pure step jump into an artificial linear ramp.
# --------------------------------------------------------------------

# ------------------------------------------------------------------
# moving‑window median‑slope detrend
# ------------------------------------------------------------------
def _detrend(
    x: np.ndarray | Sequence[Union[int, float]],
    *,
    wins: Sequence[int] = (10, 20, 30),
    step: int = 5,
) -> np.ndarray:
    """
    Remove a global linear drift using the *median* of local OLS slopes
    gathered from multiple window sizes.

    Parameters
    ----------
    x    : 1‑D iterable
    wins : iterable of int
        Window lengths to scan (must all be >= 3).
    step : int
        Hop size between window starts.

    Returns
    -------
    residuals  x - (ĉ + â·t)
        where  â = median(all local slopes),
               ĉ = mean(x).
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < min(wins, default=3):
        return x - x.mean()

    slopes = []
    for win in wins:
        if win < 3 or win > n:
            continue
        for s in range(0, n - win + 1, step):
            seg = x[s : s + win]
            t   = np.arange(win, dtype=float)
            cov   = np.mean(t * seg) - t.mean() * seg.mean()
            var_t = np.mean(t**2)     - t.mean()**2
            if var_t > 0:
                slopes.append(cov / var_t)

    if not slopes:
        return x - x.mean()

    a_hat = np.median(slopes)      # robust multi‑scale slope
    c_hat = x.mean()               # intercept = global mean
    t_full = np.arange(n, dtype=float)
    return x - (c_hat + a_hat * t_full)


def shift_cusum(x):
    """
    CUSUM range signed by direction of first large excursion.
    Less sensitive to linear drift thanks to a detrend step.
    """
    x = np.asarray(x, float)
    x_d = _detrend(x)

    cumsum = np.cumsum(x_d - x_d.mean())
    max_idx, min_idx = np.argmax(cumsum), np.argmin(cumsum)
    direction = 1 if max_idx > min_idx else -1
    return float(np.ptp(cumsum) * direction / len(x))


def shift_rolling_mean(x, window=10):
    """
    Max signed jump in rolling mean.
    Linear trend removed first.
    """
    x = _detrend(np.asarray(x, float))
    rm = pd.Series(x).rolling(window).mean()
    diff = rm.diff()
    if diff.isna().all():
        return 0.0
    idx = diff.abs().idxmax()
    return float(diff.iloc[idx])        # already signed


def shift_pelt(x, pen=1):
    """
    First mean shift detected by PELT after detrending.
    """
    x = _detrend(np.asarray(x, float))
    cp = rpt.Pelt(model='l2').fit(x).predict(pen=pen)
    if len(cp) <= 1:
        return 0.0
    i = cp[0]
    shift = x[i:].mean() - x[:i].mean()
    return float(shift)


# ------------------------------------------------------ 4. Step shift (abrupt) ------------------------------------------------------------
def step_cusum(x, threshold=2):
    """Detect steps using CUSUM"""
    cumsum = np.cumsum(x - x.mean())
    max_idx = np.argmax(np.abs(cumsum))
    if np.abs(cumsum[max_idx]) > threshold:
        shift = -(x[max_idx] - x[max_idx-1])
        return abs(shift)*np.sign(shift)
    return 0.0

def step_pelt(
    x
) -> float:
    """
    Detect a *temporary* step-like bump (up-then-down or down-then-up).

    Returns
    -------
    float
        +H  : positive bump (level goes up, stays ≥ min_len samples, then down)
        -H  : negative bump (level goes down, then up)
         0  : no bump, or only a permanent mean shift.
    """
    jump_score_all = []
    for pen in np.logspace(1.0, 2.0, 50):  
        x = np.asarray(x, dtype=float)

        # 1 — change-points with PELT (last cp returned == len(x))
        cps = rpt.Pelt(model="l2").fit(x).predict(pen=pen)
        cps = [0] + cps                                    # prepend start idx

        jump_score = 0 
        # 2 — scan triples of consecutive segments for the bump pattern
        for left, middle, right in zip(cps, cps[1:], cps[2:]):
            plateau_len = right - middle
            if plateau_len < 2:
                continue

            mean_left   = x[left:middle].mean()
            mean_mid    = x[middle:right].mean()
            mean_right  = x[right:].mean()

            jump_1 = mean_mid  - mean_left   # first step edge
            jump_2 = mean_right - mean_mid   # second edge

            # need opposite signs and similar magnitude (within 25 %)
            if np.sign(jump_1) == -np.sign(jump_2) and \
            abs(abs(jump_1) - abs(jump_2)) < 0.25 * abs(jump_1):

                noise_std = (
                    np.sqrt(
                        np.var(x[left:middle], ddof=1) +
                        np.var(x[right:],      ddof=1)
                    ) / 2
                ) or 1.0

                if abs(jump_1) / noise_std >= 0.5:
                    jump_score = float(jump_1)      # positive ⇒ up-then-down, negative ⇒ down-then-up
        jump_score_all.append(jump_score)

    # return median of all jump scores
    return float(np.mean(jump_score_all))



# ------------------------------------------------------ 5. Variability ---------------------------------------------------------------
# How much of the signal's variance looks like unstructured (white) noise?
# ----------------------------------------------------------------------
# 1.  Noise‑to‑total variance ratio  (white‑noise strength)
# ----------------------------------------------------------------------

def white_noise_strength(
    series: Sequence[Union[int, float]],
    *,
    window_smooth: int = 11,
    return_ratio: bool = False,
) -> float:
    """
    Estimate variance of the 'white' component in a 1‑D time‑series.

    Method
    ------
    1. Smooth with a Savitzky–Golay filter (poly‑order 3, window `window_smooth`)
       to approximate the structured part of the signal.
    2. Residuals = raw – smooth  → treated as white noise.
    3. Either return:
       • absolute variance  Var(residuals)            (default)
       • noise‑to‑total ratio Var(residuals) / Var(raw)  (`return_ratio=True`)

    Parameters
    ----------
    series : array‑like
        Input samples (NaNs are dropped).
    window_smooth : int, odd ≥ 5
        S‑G window length; larger → heavier smoothing.
    return_ratio : bool, optional
        If True, return the fraction in [0, 1]; otherwise return
        the absolute sample variance of the residuals.

    Returns
    -------
    float
        Absolute variance (same units² as the input) or ratio.
    """
    y = np.asarray(series, float)
    y = y[~np.isnan(y)]
    if y.size < 5:
        return 0.0

    if window_smooth < 5 or window_smooth % 2 == 0:
        raise ValueError("window_smooth must be an odd integer ≥ 5")

    trend = savgol_filter(y, window_length=window_smooth, polyorder=3)
    resid = y - trend

    var_r = np.var(resid, ddof=1)
    if return_ratio:
        var_y = np.var(y, ddof=1)
        return 0.0 if np.isclose(var_y, 0) else float(np.clip(var_r / var_y, 0.0, 1.0))
    else:
        return float(var_r)

# ----------------------------------------------------------------------
# 2.  Moving‑window robust wrapper  (mirrors curvature helper)
# ----------------------------------------------------------------------
def noise_strength_moving_window(
    x: Sequence[Union[int, float]],
    *,
    length: int = 10,
    step: int = 5,
    window_smooth: int = 7
) -> float:
    """
    Robust global estimate of white‑noise level via overlapping windows.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < length:
        return white_noise_strength(x, window_smooth=window_smooth)

    vals: list[float] = []
    for i in range(0, len(x) - length + 1, step):
        w = x[i : i + length]
        v = white_noise_strength(w, window_smooth=window_smooth)
        if np.isfinite(v):
            vals.append(v)

    if not vals:
        return 0.0

    vals = np.array(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    keep = (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
    clean = vals[keep] if keep.any() else vals

    return float(np.median(clean))


def get_all_stats(x, step = False):
    """Calculate math properties of a given timeseries"""
    res = {
        'trend': np.round(trend_slope_moving_window(x), 4),
        'curvature': np.round(trend_curvature_moving_window(x), 4),
        'seasonality': get_seasonality_scores_moving_window(x)['entropy'],
        'shift': np.round(shift_pelt(x), 4),
        'variability': np.round(noise_strength_moving_window(x), 4),
        'successive_increases': successive_increases(x),
        'successive_unchanges': successive_unchanges(x)
    }
    if step:
        res['step'] = np.round(step_pelt(x), 4) # slow, only use when step is trained on
    return res









# ----------------- Math statistics evaluation on synthetic data --------------------------------
from generation import interpolate_ts_tx
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
from tqdm import tqdm

# Add the tedit_lite and tedit_lite_tx folders to the system path
import sys, os
# Path for tedit_lite
tedit_attr_path = os.path.abspath("../tedit_lite")
if tedit_attr_path not in sys.path:
    sys.path.append(tedit_attr_path)
from tedit_generation import tedit_generate_ts_tx as tedit_generate_ts_tx

# Path for tedit_lite_tx
tedit_tx_path = os.path.abspath("../tedit_lite_tx")
if tedit_tx_path not in sys.path:
    sys.path.append(tedit_tx_path)
from tedit_tx_generation import tedit_generate_ts_tx as tedit_tx_generate_ts_tx

# calculate the math properties of the generated time series (calculate on all time series i.e. df_train.sample(500))
def eval_math_properties(df, model, config_dict, aug_type, w, y_cols = None, 
                        meta = None, # tedit meta
                        configs = None, # teidt configs
                        model_type = None # if None, will be auto-detected
                        ):
    # Auto-detect model type if not specified
    if model_type is None:
        if meta is None:
            model_type = 'vital'
        else:
            if 'level_maps' in meta:
                model_type = 'tedit'
            elif 'attr_emb_dim' in meta:
                model_type = 'tedit_tx'
            else:
                raise ValueError("Could not determine model type from meta dictionary")
    
    model.eval()
    df_augmented = pd.DataFrame()
    if y_cols is None:
        y_cols = config_dict['txt2ts_y_cols']
    
    # Set up parallel processing parameters
    n_cores = min(max(1, int(multiprocessing.cpu_count() * 0.7)), 10)
    
    for y_col in y_cols:
        y_levels = list(df[y_col].unique())
        for i in range(len(y_levels)):
            
            df_level = df[df[y_col] == y_levels[i]].reset_index(drop=False).copy()
            
            # add new text conditions
            for j in range(len(y_levels)):
                if aug_type == "conditional":
                    modified_text = df_level['text'].values[0]
                    for level in y_levels:
                        if level in modified_text:
                            modified_text = modified_text.replace(level, y_levels[j])
                    df_level['text' + str(j)] = modified_text
                elif aug_type == "marginal":
                    df_level['text' + str(j)] = y_levels[j]
                else:
                    raise ValueError("aug_type must be either 'marginal' or 'conditional'")
                
            # Augment the time series with the given text conditions
            text_cols = ['text' + str(j) for j in range(len(y_levels))]
            # mapping text_col to y_level
            col_level_map = dict(zip(['text' + str(j) for j in range(len(y_levels))], y_levels))
    
            # Choose model based on model_type
            if model_type == 'tedit_tx':
                new_level_col_map = {k: v for k, v in col_level_map.items() if k in text_cols}
                ts_hat_ls = tedit_tx_generate_ts_tx(df_level,
                                                   meta,
                                                   config_dict,
                                                   configs,
                                                   y_col,
                                                   new_level_col_map)
            elif model_type == 'tedit':
                new_level_col_map = {k: v for k, v in col_level_map.items() if k in text_cols}
                ts_hat_ls = tedit_generate_ts_tx(df_level,
                                                meta,
                                                config_dict,
                                                configs,
                                                y_col,
                                                new_level_col_map)
            else:  # vital model
                ts_hat_ls = interpolate_ts_tx(df_level, model, config_dict, text_cols, w)

            # Calculate the math properties of the generated time series
            df_prop_all = pd.DataFrame()
            for text_col, pairs in ts_hat_ls.items():
                df_prop = pd.DataFrame(pairs, columns=['aug_text', 'ts_hat'])
                
                n_cores = min(max(1, int(multiprocessing.cpu_count() * 0.7)), 10)
                properties = Parallel(n_jobs=n_cores, backend="threading")(
                    delayed(get_all_stats)(x.detach().cpu().numpy()) for x in tqdm(df_prop['ts_hat'], desc="Calculating math statistics")
                )
                
                # # Process in batches
                # ts_list = [x.detach().cpu().numpy() for x in df_prop['ts_hat']]
                # properties = []
                # # Calculate total number of batches for progress bar
                # n_batches = (len(ts_list) + batch_size - 1) // batch_size
                # parallel = Parallel(n_jobs=n_cores, batch_size=1, prefer='processes')
                # for batch_start in tqdm(range(0, len(ts_list), batch_size), total=n_batches, desc="Processing batches"):
                #     batch_end = min(batch_start + batch_size, len(ts_list))
                #     batch = ts_list[batch_start:batch_end]
                #     properties.extend(parallel(delayed(get_all_stats)(x) for x in batch))
                #     gc.collect() # force garbage collection after each batch

                df_prop['properties'] = properties
                df_prop['text_col'] = text_col
                df_prop['index'] = df_level.index# saved original index
                df_prop_all = pd.concat([df_prop_all, df_prop])
            
            df_prop_all['org_y_col'] = y_col
            df_prop_all['org_y_level'] = y_levels[i]
            df_augmented = pd.concat([df_augmented, df_prop_all])

    return df_augmented


# engineer the math properties from the math properties of the augmented time serie (df_augmented)
def eng_math_diff(df_augmented, base_y_level, augm_y_level, metrics = ['trend', 'curvature', 'seasonality', 'shift', 'variability']):

    df_metrics = None
    for metric in metrics:
        aug_df = df_augmented[
                    (df_augmented['org_y_level'].str.contains(base_y_level, case=False)) & 
                    (df_augmented['aug_text'].str.contains(augm_y_level, case=False))
                ]
        aug_df['aug_'+metric] = aug_df['properties'].apply(lambda x: x[metric])
        bas_df = df_augmented[
                    (df_augmented['org_y_level'].str.contains(base_y_level, case=False)) & 
                    (df_augmented['aug_text'].str.contains(base_y_level, case=False))
                ]
        bas_df['bas_'+metric] = bas_df['properties'].apply(lambda x: x[metric])
        scl = bas_df['bas_'+metric].std()

        # merge the two dataframes on the index column
        aug_df = aug_df.loc[:, ['index', 'aug_'+metric]]
        bas_df = bas_df.loc[:, ['index', 'bas_'+metric]]
        # assert two df has same nrow
        assert len(aug_df) == len(bas_df)
        df = aug_df.merge(bas_df, on = 'index', how = 'left')
        df['diff_'+metric] = df['aug_'+metric] - df['bas_'+metric]
        df['diff_'+metric] = df['diff_'+metric] / scl# / df['diff_'+metric].abs().max()
        df = df.loc[:, ['index', 'diff_'+metric, 'aug_'+metric, 'bas_'+metric]]
        if df_metrics is None:
            df_metrics = df
        else: 
            df_metrics  = df_metrics.merge(df, on = 'index', how = 'left')

    return df_metrics



def eng_math_diff_multiple(df_augmented, base_aug_dict, aug_type='conditional'): # , metrics = ['trend', 'curvature', 'seasonality', 'shift', 'variability']

    metrics = ['trend', 'curvature', 'seasonality', 'shift', 'variability', 'successive_increases', 'successive_unchanges']
    metrics_quest = base_aug_dict.keys()
    metrics = [metric for metric in metrics if metric in metrics_quest]
    base_aug_dict = {k:v for k,v in base_aug_dict.items() if k in metrics}
    
    df_augmented = df_augmented[df_augmented['aug_type'] == aug_type]
    
    df_ls = []
    for aug_matrix, pairs in base_aug_dict.items():
        n_cols = 4  # Always use 4 columns
        n_pairs = len(pairs)
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        # Create figure with 2 rows per pair: one for boxplots, one for histograms
        fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(n_cols*5, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(2, n_cols)  # Make it 2D array if only one row
        
        for idx, (base, aug) in enumerate(pairs):
            row = idx // n_cols
            col = idx % n_cols
            
            df = eng_math_diff(df_augmented, base, aug, metrics = metrics)
            df['base'] = base
            df['aug'] = aug
            df['aug_matrix'] = aug_matrix
            df_ls.append(df)

            # --- Boxplot in first row ---
            diff_df = pd.DataFrame()
            for metric in metrics:
                diff_df[metric] = df['diff_'+metric]
            
            box = axes[2*row, col].boxplot(
                [diff_df[col] for col in diff_df.columns],
                labels=diff_df.columns,
                showfliers=False,
                patch_artist=False  # No fill, just lines
            )
            for i, metric in enumerate(diff_df.columns):
                color = 'black' if metric == aug_matrix else 'grey'
                # Set box color
                for line in [
                    box['boxes'][i],
                    box['whiskers'][2*i], box['whiskers'][2*i+1],
                    box['caps'][2*i], box['caps'][2*i+1],
                    box['medians'][i]
                ]:
                    line.set_color(color)
                    line.set_linewidth(1)
            #axes[2*row, col].set_ylim(-5, 5)
            axes[2*row, col].axhline(0, color='black', linewidth=0.5)
            axes[2*row, col].set_title(f'{base}\n→\n{aug}', pad=20)
            if col == 0:  # Only add ylabel to first subplot
                axes[2*row, col].set_ylabel('Difference')
            
            # --- Histogram in second row ---
            aug_stat = df['aug_'+aug_matrix].values.tolist()
            bas_stat = df['bas_'+aug_matrix].values.tolist()
            axes[2*row+1, col].hist(aug_stat, bins=20, alpha=0.5, label='Augmented')
            axes[2*row+1, col].hist(bas_stat, bins=20, alpha=0.5, label='Base')
            axes[2*row+1, col].set_xlabel(aug_matrix)
            if col == 0:
                axes[2*row+1, col].set_ylabel('Count')
            axes[2*row+1, col].legend()
        
        # Hide any empty subplots
        total_plots = n_rows * n_cols
        for idx in range(n_pairs, total_plots):
            row = idx // n_cols
            col = idx % n_cols
            axes[2*row, col].axis('off')
            axes[2*row+1, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    res_df = pd.concat(df_ls)
    return res_df








