# ML Pricing

Machine learning models for empirical asset pricing, based on the framework of
[Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*](https://dachxiu.chicagobooth.edu/download/ML.pdf).

The project predicts monthly stock returns from 94 lagged firm characteristics, then evaluates
each model by the annualized Sharpe ratio of a decile long–short portfolio built from its
predictions.

## Project structure

```
ML_Pricing/
├── main.py                 # End-to-end demo: preprocess → split → fit → evaluate
├── src/
│   ├── preprocess.py       # DataPipeline: load, impute, normalize
│   ├── expandingwindow.py  # ExpandingWindow: time-series train/val/test splits
│   ├── linear_models.py    # OLS, ElasticNetModel, PCAModel
│   ├── treemodels.py       # RandomForestModel, GradientBoostedModel
│   └── neuralnet.py        # NeuralNetModel (PyTorch feed-forward net)
├── data/
│   ├── datashare.csv       # Firm characteristics dataset (not committed, see below)
│   └── readme.txt          # Dataset description from the original authors
├── ML_Pricing_Paper.pdf    # Reference paper
├── ML_Pricing_Scope.pdf    # Project scope
└── requirements.txt
```

## Data

The dataset is the firm characteristics panel from Gu, Kelly & Xiu, available on
[Dacheng Xiu's website](https://dachxiu.chicagobooth.edu/). Each row is one stock–month and
contains:

- `DATE` — end of month (YYYYMMDD)
- `permno` — CRSP permanent company identifier
- 94 lagged firm characteristics (momentum, size, value, volatility, ...)
- `sic2` — two-digit industry code

Publication lags are already applied by the authors, so characteristics at month *t* can be used
to predict the return at month *t + 1* without look-ahead bias. The file is large and therefore
gitignored — download it and place it at `data/datashare.csv`. See `data/readme.txt` for full
details and citation requirements.

## Methodology

**Preprocessing** (`src/preprocess.py`)

1. **Target construction** — the next month's `mom1m` (one-month momentum, i.e. last month's
   return) is shifted back one period per stock to serve as the forward return target `ret`.
2. **Imputation** — missing characteristics are filled with the cross-sectional monthly median,
   following the paper. A binary indicator matrix records which values were imputed and is
   appended to the feature set so models can learn from missingness itself.
3. **Normalization** — each characteristic is rank-transformed within each month to the
   interval \[-1, 1\], making features comparable across time and robust to outliers.

**Backtesting** (`src/expandingwindow.py`)

An expanding-window scheme splits the panel chronologically: an initial training window, followed
by validation and test years. After each round the training window absorbs the validation and
test periods and the process repeats, mimicking how a model would be re-estimated in production.

**Models**

| Model | File | Notes |
|---|---|---|
| OLS | `src/linear_models.py` | Least-squares baseline via `np.linalg.lstsq` |
| Elastic Net | `src/linear_models.py` | L1/L2-penalized regression (scikit-learn) |
| PCA regression | `src/linear_models.py` | OLS on the top principal components |
| Random Forest | `src/treemodels.py` | Bagged decision trees (scikit-learn) |
| Gradient Boosting | `src/treemodels.py` | `HistGradientBoostingRegressor` |
| Neural Network | `src/neuralnet.py` | Single-hidden-layer ReLU network (PyTorch) |

**Evaluation**

Each month, stocks are sorted by predicted return. The strategy goes long the top decile and
short the bottom decile; the annualized Sharpe ratio of the resulting spread measures how well
the model ranks stocks cross-sectionally.

## Getting started

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datashare.csv (see Data section) into data/

# 4. Run the demo
python main.py
```

`main.py` runs the full pipeline on the first 25,000 rows, fits OLS and the neural network on an
expanding-window split, and prints the long–short Sharpe ratio for each:

```
OLS Sharpe: ~0.5
NN  Sharpe: ~1.5
```

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `torch`, `matplotlib`, and `seaborn`
(see `requirements.txt`).

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning.
  *The Review of Financial Studies*, 33(5), 2223–2273.
- Gu, S., Kelly, B., & Xiu, D. (2021). Autoencoder Asset Pricing Models.
  *Journal of Econometrics*, 222(1), 429–450.
