import json
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from train import demand_series

N_SAMPLES = 100


def candidate_configs(n_observations: int, season: int) -> list[dict]:
    configs = [{"trend": None}, {"trend": "add", "damped_trend": True}]
    if n_observations >= 2 * season:
        configs += [
            {"trend": None, "seasonal": "add", "seasonal_periods": season},
            {"trend": "add", "damped_trend": True, "seasonal": "add", "seasonal_periods": season},
        ]
    return configs


def fit_best_ets(series: pd.Series, season: int):
    """Fit the candidate ETS models and return the one with the lowest AIC, or None if none fits."""
    y = series.interpolate(limit_direction="both").to_numpy(dtype=float)
    best = None
    for config in candidate_configs(len(y), season):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = ETSModel(y, error="add", **config).fit(disp=False)
        except Exception:
            continue
        if np.isfinite(result.aic) and (best is None or result.aic < best.aic):
            best = result
    return best


def sample_paths(series: pd.Series, horizon: int, settings: dict, rng: np.random.Generator) -> np.ndarray:
    """Array of shape (horizon, N_SAMPLES) of non-negative simulated consumption."""
    observed = series.dropna()
    fitted = fit_best_ets(series, settings["season"]) if len(observed) >= settings["min_observations"] else None
    if fitted is None:
        mean = float(observed.tail(6).mean()) if len(observed) else 0.0
        return rng.poisson(max(mean, 0.0), (horizon, N_SAMPLES)).astype(float)
    paths = fitted.simulate(nsimulations=horizon, repetitions=N_SAMPLES, anchor="end", rng=rng)
    return np.maximum(np.asarray(paths, dtype=float).reshape(horizon, N_SAMPLES), 0.0)


def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
    with open(model_filename) as f:
        settings = json.load(f)
    rng = np.random.default_rng(0)
    history = demand_series(pd.read_csv(historic_data_filename))
    future_df = pd.read_csv(future_data_filename)
    future_df["period"] = future_df["time_period"].astype(str).map(lambda p: pd.Period(p, freq="M"))

    rows = []
    for loc, group in future_df.groupby("location", sort=False):
        series = history.get(loc, pd.Series(dtype=float))
        last = series.index.max() if len(series) else group["period"].min() - 1
        steps = [(period - last).n for period in group["period"]]
        paths = sample_paths(series, max(steps), settings, rng)
        for (_, row), step in zip(group.iterrows(), steps, strict=True):
            row_data = {"time_period": row["time_period"], "location": loc}
            row_data.update({f"sample_{i}": s for i, s in enumerate(paths[step - 1])})
            rows.append(row_data)

    pd.DataFrame(rows).to_csv(output_filename, index=False)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
