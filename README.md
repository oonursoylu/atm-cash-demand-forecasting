# ATM Cash Demand Forecasting

Developed by Onur Soylu and Berk Barış Ersoy as a project for the Applications
of AI course.

This project investigates daily cash-demand forecasting for 31 ATMs. It
compares calendar and lag-based features, different treatments of extended
zero-withdrawal periods, and forecasting models ranging from linear regression
and XGBoost to LSTM and N-BEATS.

## Business context

ATM replenishment involves a trade-off. Excess cash increases idle-capital,
insurance and transport costs, while insufficient cash can cause service
failures and emergency replenishments. Forecasts therefore need to support both
cash availability and efficient replenishment planning.

## Project scope

The original course work explores:

- calendar, holiday, lag and rolling-window features;
- raw active-period data and removal of extended zero-withdrawal periods;
- Linear Regression, XGBoost, LSTM and N-BEATS models;
- global models and demand-based ATM clusters;
- several sequence lengths for the neural-network experiments.

After the course, a reproducible benchmark was added to evaluate the main
forecasting approach more rigorously. It uses expanding-window validation for
model selection and keeps the final 15% of dates untouched until the final
evaluation.

## Re-evaluation results

All lag and rolling features in the added benchmark use past observations only.
The best result within this benchmark is the zero-handled XGBoost model.

| Data variant | Model | Final-test sMAPE | WAPE | MAE |
|---|---:|---:|---:|---:|
| Zero-handled | XGBoost | **42.60%** | **32.12%** | **15,438.71** |
| Raw active period | XGBoost | 42.94% | 32.33% | 15,539.38 |
| Raw active period | Naive lag-1 | 56.40% | 52.11% | 25,045.70 |
| Raw active period | Ridge | 60.77% | 52.55% | 25,254.32 |

Compared with the lag-1 baseline, the selected model reduces final-test sMAPE
by 13.80 percentage points and WAPE by 19.99 percentage points. The difference
between the two XGBoost variants is small because the final test targets do not
contain extended zero runs.

The archived course experiments report a lower weighted sMAPE of 27.84% for an
XGBoost configuration. That value is useful for comparing the exploratory
configurations, but it is not treated as an independent test estimate because
the same holdout influenced both model selection and reporting. The 42.60%
result is therefore the more defensible performance estimate, not a claim about
the best result achievable on the dataset.

Because ATM demand includes many zero and low-demand observations, sMAPE can be
volatile. WAPE is included as a complementary measure of aggregate forecast
error.

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

The source workbook is not distributed with the repository. The analysis
expects an `ATM` sheet containing:

- `CASHP_ID`
- `DATE`
- `WITHDRWLS`

Place a legally obtained copy at
`data/2024-12-09_ATM_Branch_Data.xlsx`. See `data/README.md` for details.

## Setup and execution

```bash
conda env create -f environment.yml
conda activate atm-forecasting
```

Run the reproducible benchmark from the repository root:

```bash
python src/time_safe_benchmark.py \
  --data data/2024-12-09_ATM_Branch_Data.xlsx \
  --output-dir results
```

The exploratory notebooks include repeated Optuna searches and neural-network
training, so they are substantially more computationally expensive than the
benchmark script.

## Limitations and possible extensions

- The dataset covers one city and one historical period.
- Neural-network and clustering configurations have not yet been re-evaluated
  under the same untouched-test protocol.
- Intermittent-demand methods such as Croston, SBA and TSB would provide useful
  additional baselines.
- Probabilistic forecasts and a replenishment-cost evaluation would connect
  forecast accuracy more directly to the operational decision.

## Licence

MIT. See `LICENSE`.
