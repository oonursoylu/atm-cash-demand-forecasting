# Results

## Leakage-aware benchmark

`time_safe_benchmark.csv` is produced by `src/time_safe_benchmark.py` using:

- three expanding-window validation folds on the development period;
- the final 15% of dates as an untouched test period;
- series-aware lag and rolling features;
- non-negative predictions evaluated with sMAPE, WAPE and MAE.

The strongest verified final-test result is the zero-handled XGBoost model:

| Model | sMAPE | WAPE | MAE |
|---|---:|---:|---:|
| XGBoost, zero-handled | **42.60%** | **32.12%** | **15,438.71** |
| XGBoost, raw active period | 42.94% | 32.33% | 15,539.38 |
| Naive lag-1, raw active period | 56.40% | 52.11% | 25,045.70 |

## Archived exploratory workbook

`standardized_models_results.xlsx` contains 28 sheets covering Linear
Regression, XGBoost, LSTM and N-BEATS across feature, sequence-length,
clustering and zero-handling variants.

Its best recorded weighted sMAPE is 27.84% for XGBoost with combined calendar
and temporal features. Treat this as an exploratory validation result, not an
untouched-test estimate: the archived notebooks tune and report on the same
holdout, and their sequence construction does not fully isolate ATM series.


