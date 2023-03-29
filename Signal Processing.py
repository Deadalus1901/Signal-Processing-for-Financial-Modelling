# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:27:29 2023

@author: Uday Goel
"""

from math import pi

from scipy.signal import butter, filtfilt
import numpy as np
import talib

import matplotlib.pyplot as plt

from openbb_terminal.sdk import openbb

data = openbb.forex.load(
    from_symbol="EUR", 
    to_symbol="USD", 
    start_date="2016-01-01", 
    end_date="2021-12-31"
)

prices = (
    data["Adj Close"]
    .to_frame()
    .rename(
        columns={
            "Adj Close": "close"
        }
    )
)

prices["log_return"] = (
    prices.close
    .apply(np.log)
    .diff(1)
)
# Hilbert Transform - Dominant Cycle Phase
prices["phase"] = talib.HT_DCPHASE(prices.close)

# Convert into a wave using a cycle model
prices["signal"] = np.sin(prices.phase + pi / 4)

# Use the Hilbert Transform - Dominant Cycle Period
prices["period"] = talib.HT_DCPERIOD(prices.close)

def butter_bandpass(data, period, delta=0.5, fs=5):
    nyq = 0.5 * fs

    # Low cutoff frequency
    low = 1.0 / (period * (1 + delta))
    low /= nyq

    # High cutoff frequency
    high = 1.0 / (period * (1 - delta))
    high /= nyq

    b, a = butter(2, [low, high], btype="band")

    return filtfilt(b, a, data)

def roll_apply(e):
    close = prices.close.loc[e.index]
    period = prices.period.loc[e.index][-1]
    out = butter_bandpass(close, period)
    return out[-1]

prices["filtered"] = (
    prices.dropna()
    .rolling(window=30)
    .apply(lambda series: roll_apply(series), raw=False)
    .iloc[:, 0]
)

prices["amplitude"] = (
    prices.
    filtered
    .rolling(window=30)
    .apply(
        lambda series: series.max() - series.min()
    )
)
prices["ema_amplitude"] = (
    talib
    .EMA(
        prices.amplitude,
        timeperiod=30
    )
)
signal_thresh = 0.75
amp_thresh = 0.004  # 40 pips

prices["position"] = 0
prices.loc[
    (prices.signal >= signal_thresh), 
    (prices.amplitude > amp_thresh), "position"
] = -1
prices.loc[
    (prices.signal <= -signal_thresh), 
    (prices.amplitude > amp_thresh), "position"
] = 1
fig, axes = plt.subplots(
    nrows=3,
    figsize=(15, 10),
    sharex=True
)

prices.ema_amplitude.plot(
    ax=axes[0],
    title="amp"
)
axes[0].axhline(
    amp_thresh,
    lw=1,
    c="r"
)
prices.signal.plot(
    ax=axes[1],
    title="signal"
)
axes[1].axhline(
    signal_thresh,
    c="r"
)
axes[1].axhline(
    -signal_thresh,
    c="r"
)
prices.position.plot(
    ax=axes[2],
    title="position"
)
fig.tight_layout()