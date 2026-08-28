"""Leakage-aware ATM withdrawal forecasting benchmark.

This script complements the exploratory notebooks in this repository. It uses
an expanding-window cross-validation scheme for model selection and keeps the
last 15% of dates as an untouched final test period.

The source workbook is intentionally not committed. See data/README.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


TARGET = "WITHDRWLS"
ID_COL = "CASHP_ID"
DATE_COL = "DATE"
ZERO_RUN_THRESHOLD = 14


OFFICIAL_HOLIDAYS = pd.to_datetime(
    [
        "2006-01-01", "2006-01-10", "2006-01-11", "2006-01-12", "2006-01-13",
        "2006-04-23", "2006-05-19", "2006-08-30", "2006-10-23", "2006-10-24",
        "2006-10-25", "2006-10-29", "2006-12-31", "2007-01-01", "2007-01-02",
        "2007-01-03", "2007-04-23", "2007-05-19", "2007-08-30", "2007-10-12",
        "2007-10-13", "2007-10-14", "2007-10-29", "2007-12-20", "2007-12-21",
        "2007-12-22", "2007-12-23", "2007-12-31", "2008-01-01", "2008-04-23",
        "2008-05-19", "2008-08-30", "2008-09-30", "2008-10-01", "2008-10-02",
        "2008-10-29", "2008-12-08", "2008-12-09", "2008-12-10", "2008-12-11",
        "2008-12-31",
    ]
)

HALF_DAYS = pd.to_datetime(
    [
        "2006-01-09", "2006-10-22", "2006-12-30", "2007-10-11",
        "2007-12-19", "2008-09-29", "2008-12-07",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the source Excel workbook")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denominator, 1e-9)) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denominator * 100) if denominator else float("nan")


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=float)
    predicted = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "smape": smape(actual, predicted),
        "wape": wape(actual, predicted),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def mark_long_zero_runs(group: pd.DataFrame) -> pd.Series:
    is_zero = group[TARGET].eq(0)
    run_id = is_zero.ne(is_zero.shift(fill_value=False)).cumsum()
    run_length = is_zero.groupby(run_id).transform("size")
    return is_zero & run_length.ge(ZERO_RUN_THRESHOLD)


def build_features(data_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(data_path, sheet_name="ATM", engine="openpyxl")
    required = {ID_COL, DATE_COL, TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[ID_COL, DATE_COL, TARGET]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    df["is_active"] = df.groupby(ID_COL)[TARGET].transform(lambda s: s.gt(0).cummax())
    df["long_zero_run"] = (
        df.groupby(ID_COL, group_keys=False)
        .apply(mark_long_zero_runs, include_groups=False)
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day_of_month"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["day_of_year"] = df[DATE_COL].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["is_official_holiday"] = df[DATE_COL].isin(OFFICIAL_HOLIDAYS).astype(int)
    df["is_half_day"] = df[DATE_COL].isin(HALF_DAYS).astype(int)

    grouped_target = df.groupby(ID_COL)[TARGET]
    for lag in (1, 2, 3, 7, 14, 28):
        df[f"lag_{lag}"] = grouped_target.shift(lag)
    shifted = grouped_target.shift(1)
    for window in (7, 14, 28):
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df[ID_COL]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df[ID_COL]).rolling(window).std().reset_index(level=0, drop=True)
        )

    atm_dummies = pd.get_dummies(df[ID_COL], prefix="atm", dtype=int)
    df = pd.concat([df, atm_dummies], axis=1)

    feature_columns = [
        "day_of_month", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "doy_sin", "doy_cos", "is_official_holiday", "is_half_day",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
        "rolling_std_14", "rolling_mean_28", "rolling_std_28",
        *atm_dummies.columns.tolist(),
    ]
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df, feature_columns


def fit_predict(model_name: str, params: dict[str, Any], x_train: pd.DataFrame,
                y_train: pd.Series, x_eval: pd.DataFrame, n_jobs: int) -> np.ndarray:
    if model_name == "Ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    elif model_name == "XGBoost":
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=n_jobs,
            tree_method="hist",
            **params,
        )
    else:
        raise ValueError(model_name)

    model.fit(x_train, np.log1p(y_train))
    return np.expm1(model.predict(x_eval))


def tune_model(model_name: str, candidates: list[dict[str, Any]], dev: pd.DataFrame,
               feature_columns: list[str], dev_dates: np.ndarray, n_jobs: int) -> tuple[dict[str, Any], float]:
    splitter = TimeSeriesSplit(n_splits=3)
    ranked: list[tuple[float, dict[str, Any]]] = []

    for params in candidates:
        fold_scores: list[float] = []
        for train_index, validation_index in splitter.split(dev_dates):
            train_dates = dev_dates[train_index]
            validation_dates = dev_dates[validation_index]
            train = dev[dev[DATE_COL].isin(train_dates)]
            validation = dev[dev[DATE_COL].isin(validation_dates)]
            predictions = fit_predict(
                model_name, params, train[feature_columns], train[TARGET],
                validation[feature_columns], n_jobs,
            )
            fold_scores.append(evaluate(validation[TARGET], predictions)["smape"])
        ranked.append((float(np.mean(fold_scores)), params))

    ranked.sort(key=lambda item: item[0])
    return ranked[0][1], ranked[0][0]


def run_variant(df: pd.DataFrame, feature_columns: list[str], variant: str,
                test_start: pd.Timestamp, n_jobs: int) -> list[dict[str, Any]]:
    eligible = df["is_active"].copy()
    if variant == "zero_handled":
        eligible &= ~df["long_zero_run"]
    data = df[eligible].copy()

    dev = data[data[DATE_COL] < test_start]
    test = data[data[DATE_COL] >= test_start]
    dev_dates = np.sort(dev[DATE_COL].unique())

    results: list[dict[str, Any]] = []
    for label, column in (
        ("Naive (lag 1)", "lag_1"),
        ("Seasonal naive (lag 7)", "lag_7"),
    ):
        baseline = np.clip(test[column].to_numpy(dtype=float), 0, None)
        results.append({
            "variant": variant,
            "model": label,
            "cv_smape": np.nan,
            **evaluate(test[TARGET], baseline),
            "test_rows": len(test),
            "parameters": "{}",
        })

    grids = {
        "Ridge": [{"alpha": value} for value in (0.1, 1.0, 10.0, 100.0)],
        "XGBoost": [
            {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.03,
             "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
             "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0},
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
             "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 5.0},
        ],
    }

    for model_name, candidates in grids.items():
        best_params, cv_smape = tune_model(
            model_name, candidates, dev, feature_columns, dev_dates, n_jobs,
        )
        predictions = fit_predict(
            model_name, best_params, dev[feature_columns], dev[TARGET],
            test[feature_columns], n_jobs,
        )
        results.append({
            "variant": variant,
            "model": model_name,
            "cv_smape": cv_smape,
            **evaluate(test[TARGET], predictions),
            "test_rows": len(test),
            "parameters": json.dumps(best_params, sort_keys=True),
        })
    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, feature_columns = build_features(args.data)
    unique_dates = np.sort(df[DATE_COL].unique())
    test_start = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.85)])

    rows = []
    for variant in ("raw_active", "zero_handled"):
        rows.extend(run_variant(df, feature_columns, variant, test_start, args.n_jobs))

    results = pd.DataFrame(rows).sort_values(["variant", "smape"]).reset_index(drop=True)
    results.to_csv(args.output_dir / "time_safe_benchmark.csv", index=False)

    metadata = {
        "source_file": args.data.name,
        "atm_count": int(df[ID_COL].nunique()),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "final_test_start": str(test_start.date()),
        "zero_run_threshold_days": ZERO_RUN_THRESHOLD,
        "validation": "3-fold expanding-window CV on the development period; final 15% of dates untouched",
    }
    (args.output_dir / "time_safe_benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )

    plot = results.pivot(index="model", columns="variant", values="smape")
    ax = plot.plot(kind="bar", figsize=(9, 5), color=["#8da0cb", "#66c2a5"])
    ax.set_title("Final-test sMAPE by model and zero-handling variant")
    ax.set_ylabel("sMAPE (%) — lower is better")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "time_safe_benchmark.png", dpi=180)
    plt.close()

    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

