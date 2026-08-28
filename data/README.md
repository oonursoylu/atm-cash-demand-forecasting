# Data directory

The source workbook is intentionally excluded from version control.

Expected local path:

```text
data/2024-12-09_ATM_Branch_Data.xlsx
```

The `ATM` sheet must contain these columns:

| Column | Meaning |
|---|---|
| `CASHP_ID` | ATM identifier |
| `DATE` | Daily observation date |
| `WITHDRWLS` | Daily cash withdrawals |

The verified local dataset contains 31 ATMs and 786 dates per ATM, spanning
1 January 2006 to 25 February 2008. The modelling script drops the first 28
dates per ATM after creating lagged and rolling features.

Do not commit the workbook unless you have confirmed that redistribution is
permitted.


