import json
import sys

import pandas as pd

SEASON = 12
MIN_OBSERVATIONS = 6


def demand_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Monthly consumption per location, indexed by period.

    Missing entries and stock-out months (zero stock on hand) are NaN: consumption in a
    stock-out month understates demand, so it is treated as not observed.
    """
    df = df.copy()
    df["period"] = df["time_period"].astype(str).map(lambda p: pd.Period(p, freq="M"))
    stock_out = df["stock_on_hand"].fillna(1) <= 0
    df["demand"] = df["consumption"].where(~stock_out)
    return {loc: g.set_index("period")["demand"].sort_index() for loc, g in df.groupby("location")}


def train(training_data_filename: str, model_path: str):
    # Models are fitted on the historic data at prediction time, so each forecast uses the latest observations
    pd.read_csv(training_data_filename)
    with open(model_path, "w") as f:
        json.dump({"season": SEASON, "min_observations": MIN_OBSERVATIONS}, f)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
