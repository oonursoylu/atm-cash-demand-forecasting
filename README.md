# ATM Cash Withdrawal Forecasting

Daily cash-demand forecasting for 31 ATMs, with a focus on calendar effects,
lagged demand, zero-withdrawal periods and model comparison.

The repository now separates two layers of work:

- **Leakage-aware benchmark:** a reproducible expanding-window evaluation with
  an untouched final test period (`src/time_safe_benchmark.py`).
- **Exploratory experiments:** Linear Regression, XGBoost, LSTM and N-BEATS
  notebooks covering feature sets, sequence lengths, clustering and zero
  handling.

## Why this project matters

ATM replenishment has asymmetric operational costs. Excess cash creates idle
capital, insurance and transport costs; insufficient cash creates service
failures and emergency replenishment. Forecast quality therefore affects both
customer availability and cash logistics.

## Verified benchmark result

The leakage-aware benchmark uses three expanding-window validation folds for
model selection and reserves the final 15% of dates as an untouched test set.
All lag and rolling features use past observations only.

| Data variant | Model | Final-test sMAPE | WAPE | MAE |
|---|---:|---:|---:|---:|
| Zero-handled | XGBoost | **42.60%** | **32.12%** | **15,438.71** |
| Raw active period | XGBoost | 42.94% | 32.33% | 15,539.38 |
| Raw active period | Naive lag-1 | 56.40% | 52.11% | 25,045.70 |
| Raw active period | Ridge | 60.77% | 52.55% | 25,254.32 |

On the untouched test period, the zero-handled XGBoost model improves sMAPE by
13.80 percentage points and WAPE by 19.99 percentage points versus the lag-1
naive benchmark. In this final period, long zero runs do not occur in the
targets, so zero handling mainly changes the development data and produces only
a small improvement over the raw-active XGBoost model.

The full table and chart are in:

- `results/time_safe_benchmark.csv`
- `results/time_safe_benchmark.png`
- `results/time_safe_benchmark_metadata.json`

## Exploratory experiment scope

The archived notebooks investigate:

- **Calendar features:** weekday, day of month, month, official holidays and half-days
- **Temporal features:** lags of 1, 2, 3, 7, 14 and 28 days; rolling means and standard deviations
- **Models:** Linear Regression, XGBoost, LSTM and N-BEATS
- **Sequence lengths:** 7, 14, 28 and 84 days, depending on the experiment
- **ATM segmentation:** global models and k-means demand-level clusters
- **Zero handling:** raw active-period data versus removal of long zero runs

The archived workbook contains the earlier exploratory comparison. Its best
recorded weighted sMAPE is 27.84% for XGBoost with combined calendar and temporal
features. That figure is **not presented as final-test performance**: the earlier
notebooks used the same holdout for tuning and reporting, and their row-wise
sequence construction does not fully isolate ATM series. Use the leakage-aware
benchmark above for defensible performance claims.

## Repository structure

```text
.
├── data/
│   └── README.md
├── notebooks/
│   ├── preliminary_data_analysis.ipynb
│   ├── models_with_zero_handling.ipynb
│   └── models_without_zero_handling.ipynb
├── results/
│   ├── README.md
│   ├── standardized_models_results.xlsx
│   ├── time_safe_benchmark.csv
│   ├── time_safe_benchmark.png
│   └── time_safe_benchmark_metadata.json
├── src/
│   └── time_safe_benchmark.py
├── environment.yml
└── LICENSE
```

## Data

The source workbook is not distributed in the repository. The analysis expects
an `ATM` sheet with:

- `CASHP_ID`
- `DATE`
- `WITHDRWLS`

Place a legally obtained copy at
`data/2024-12-09_ATM_Branch_Data.xlsx`. See `data/README.md` for details.

## Setup

```bash
conda env create -f environment.yml
conda activate atm-forecasting
```

Run the leakage-aware benchmark from the repository root:

```bash
python src/time_safe_benchmark.py \
  --data data/2024-12-09_ATM_Branch_Data.xlsx \
  --output-dir results
```

The notebooks are computationally heavier because they include repeated Optuna
searches and neural models. Reduce the number of trials for a quick smoke test.

## Evaluation design

The current benchmark:

1. sorts every ATM series by date;
2. creates lags and rolling statistics using past values only;
3. keeps the last 15% of dates untouched;
4. tunes Ridge and XGBoost over three expanding-window folds;
5. reports sMAPE, WAPE and MAE on the final test period;
6. compares raw active-period data with a 14-day long-zero-run rule.

## Limitations and next steps

- The dataset covers one city and one historical period; external validity is limited.
- Deep-learning and clustering experiments remain exploratory until rerun under
  the same series-aware, untouched-test protocol.
- Probabilistic forecasts and prediction intervals are not yet included.
- A replenishment-cost evaluation would connect forecast error more directly to
  the operational decision.

## Licence

MIT. See `LICENSE`.


