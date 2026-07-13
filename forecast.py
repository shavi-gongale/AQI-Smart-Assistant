"""
forecast.py
===========
7-day AQI forecast.

Day 1 is ALWAYS exactly equal to the anchor AQI:
    - if `base_aqi` is supplied, that is the anchor
    - otherwise the model is run on `features` and that becomes the anchor

Days 2-7 evolve with small random perturbations of the input features
plus a smoothed trend, so the result is plausible but reproducible.
"""

import numpy as np
from predict_aqi import predict_aqi, get_default_features


def forecast_next_7_days(features=None, base_aqi=None, days=7):
    """
    Produce a `days`-long forecast.

    Parameters
    ----------
    features : list[float] | None
        13 feature values used for the ML model.  Falls back to defaults.
    base_aqi : float | None
        If given, this value is used as Day 1 of the forecast — guaranteed
        equal to whatever the caller predicted before.
    days : int
        Forecast length (default 7).

    Returns
    -------
    list[float]   length == days
    """
    if features is None:
        features = get_default_features()
    base_features = np.array(features, dtype=float)

    # ── Day 1 anchor ─────────────────────────────────────────────────────────
    if base_aqi is None:
        anchor = predict_aqi(base_features.tolist())
        if anchor is None:
            anchor = 100.0
    else:
        anchor = float(base_aqi)

    anchor = round(max(0.0, anchor), 2)
    forecast = [anchor]
    history  = [anchor]

    # Use a deterministic RNG seeded from the anchor so the same anchor
    # always gives the same forecast (useful when navigating pages).
    rng = np.random.default_rng(int(round(anchor)) or 1)

    # ── Days 2..N ────────────────────────────────────────────────────────────
    for day in range(1, days):
        noise        = rng.normal(0, 0.03, len(base_features))
        new_features = base_features * (1 + noise)

        pred = predict_aqi(new_features.tolist())
        if pred is None:
            pred = anchor

        history.append(pred)

        # Smooth with a 3-day moving average once we have enough history.
        if len(history) > 2:
            pred = (history[-1] + history[-2] + history[-3]) / 3

        # Slow drift so the chart is not flat.
        pred += day * 2

        forecast.append(round(max(0.0, pred), 2))
        base_features = new_features

    # Belt-and-braces: never let Day 1 drift due to rounding.
    forecast[0] = anchor
    return forecast


def get_forecast_for_day(features, day_offset, base_aqi=None):
    full = forecast_next_7_days(features, base_aqi=base_aqi)
    if 1 <= day_offset <= len(full):
        return full[day_offset - 1]
    return None
