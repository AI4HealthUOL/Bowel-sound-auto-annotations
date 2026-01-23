from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_band_filter(sr: int, f_lo: float = 30.0, f_hi: float = 4000.0, order: int = 4):
    """
    Design a Butterworth band filter in second-order-sections (SOS) form.

    Notes:
    - SOS is numerically stable.
    - We apply it with zero-phase filtering (sosfiltfilt) to avoid phase distortion.
    """
    nyq = 0.5 * sr
    freqs = []
    btype = None

    if f_lo is not None and f_lo > 0:
        freqs.append(f_lo / nyq)
    if f_hi is not None and f_hi < nyq:
        freqs.append(f_hi / nyq)

    if len(freqs) == 2:
        btype = "bandpass"
    elif len(freqs) == 1:
        # High-pass OR low-pass
        if f_lo is not None and f_lo > 0 and (f_hi is None or f_hi >= nyq):
            btype = "highpass"
        else:
            btype = "lowpass"
    else:
        # No filtering requested
        return None

    return butter(order, freqs, btype=btype, output="sos")


def apply_pre_filter(y: np.ndarray, sr: int, mode: str = "default") -> np.ndarray:
    """
    Optional pre-filtering step to reduce irrelevant frequency content.

    Modes:
      - none    : no filtering
      - default : 30–4000 Hz
      - heart   : 20–400 Hz
      - lung    : 100–2000 Hz
    """
    if mode == "none":
        return y

    if mode == "heart":
        f_lo, f_hi = 20.0, 400.0
    elif mode == "lung":
        f_lo, f_hi = 100.0, 2000.0
    else:
        f_lo, f_hi = 30.0, 4000.0

    sos = design_band_filter(sr, f_lo=f_lo, f_hi=f_hi, order=4)
    if sos is None:
        return y

    y_filt = sosfiltfilt(sos, y)
    return y_filt.astype(y.dtype, copy=False)
