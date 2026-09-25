# ML Pricing

A machine learning project for empirical asset pricing based on Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*.

The goal is to predict monthly stock returns using 94 lagged firm characteristics and compare several ML models based on the Sharpe ratio of a long-short portfolio formed from their predictions.

## Models

The project currently includes:

- OLS
- Elastic Net
- PCA regression
- Random Forest
- Gradient Boosting
- Feed-forward neural network in PyTorch

## Project Structure

```text
ML_Pricing/
├── main.py
├── src/
│   ├── preprocess.py
│   ├── expandingwindow.py
│   ├── linear_models.py
│   ├── treemodels.py
│   └── neuralnet.py
├── data/
│   ├── datashare.csv
│   └── readme.txt
├── ML_Pricing_Paper.pdf
├── ML_Pricing_Scope.pdf
└── requirements.txt
```

## Data

The project uses the firm characteristics dataset from Gu, Kelly & Xiu. Each row represents a stock-month and contains 94 lagged firm characteristics along with identifiers such as `DATE`, `permno`, and `sic2`.

The dataset is too large to include in the repository. Download `datashare.csv` from Dacheng Xiu's website and place it in:

```text
data/datashare.csv
```

## Pipeline

The preprocessing pipeline:

1. Builds a forward monthly return target.
2. Fills missing values using the monthly cross-sectional median.
3. Adds indicators for imputed values.
4. Rank-normalizes characteristics to `[-1, 1]`.

Models are trained and evaluated using expanding-window time-series splits so that future data is never used during training.

## Evaluation

For each month, stocks are ranked by predicted return.

The strategy goes long the top decile and short the bottom decile. Model performance is compared using the annualized Sharpe ratio of the resulting long-short returns.

## Running

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

`main.py` runs the preprocessing, expanding-window split, model training, and portfolio evaluation pipeline.

## Reference

Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. The Review of Financial Studies, 33(5), 2223-2273.
