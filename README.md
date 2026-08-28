# ATM Cash Demand Forecasting

Applications of AI course project at Otto von Guericke University Magdeburg,
Summer Semester 2025. Developed by Onur Soylu and Berk Barış Ersoy.

This project investigates daily cash-withdrawal forecasting for 31 ATMs using
machine-learning and deep-learning models. The experiments compare calendar and
temporal features, several input-window lengths, demand-based ATM clusters and
two treatments of extended zero-withdrawal periods.

## Business context

ATM replenishment involves a trade-off. Excess cash increases idle-capital,
insurance and transport costs, while insufficient cash can cause service
failures and emergency replenishments. More accurate forecasts can support both
cash availability and efficient replenishment planning.

## Methods

The course experiments cover:

- calendar and holiday features;
- lagged demand, rolling means and rolling standard deviations;
- Linear Regression, XGBoost, LSTM and N-BEATS;
- global models and demand-based ATM clusters;
- experiments with and without extended zero-withdrawal periods.

## Course results

The archived workbook in `results/standardized_models_results.xlsx` contains 28
result tables covering the tested model and feature configurations. The best
recorded weighted sMAPE is **27.84%**, achieved by XGBoost with combined
calendar and temporal features, an 84-day input window and a single global
model (`n_clusters = 1`).

| Model | Features | Input window | Clusters | Weighted sMAPE |
|---|---|---:|---:|---:|
| XGBoost | Calendar and temporal | 84 days | 1 | **27.84%** |

These figures are the results recorded for the original course experiments.
The workbook is retained so that the model comparisons can be inspected
directly.

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

The source workbook is not distributed with the repository. The notebooks
expect an `ATM` sheet containing:

- `CASHP_ID`
- `DATE`
- `WITHDRWLS`

Place a legally obtained copy at:

```text
data/2024-12-09_ATM_Branch_Data.xlsx
```

The path can also be supplied through the `ATM_DATA_PATH` environment variable.

## Running the project in VS Code

1. Open the repository folder in VS Code.
2. Create the Conda environment from the integrated terminal:

   ```bash
   conda env create -f environment.yml
   conda activate atm-forecasting
   ```

3. In VS Code, select the `atm-forecasting` Python interpreter and use the same
   environment as the Jupyter notebook kernel.
4. Add the source workbook to the `data` directory.
5. Start with `notebooks/preliminary_data_analysis.ipynb`, followed by either of
   the modelling notebooks.

The modelling notebooks contain repeated Optuna searches and neural-network
training, so a full run can take substantial time.

## Additional evaluation

`src/time_safe_benchmark.py` is a lightweight, time-ordered benchmark added
after the course. It uses expanding-window validation and an untouched final
period. Its results are kept as a supplementary reproducibility check and do
not replace the archived course results above.

Run it from the repository root with:

```bash
python src/time_safe_benchmark.py \
  --data data/2024-12-09_ATM_Branch_Data.xlsx \
  --output-dir results
```

## Licence

MIT. See `LICENSE`.
