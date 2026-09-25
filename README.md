# ML Pricing

Predicts monthly stock returns using 94 lagged firm characteristics. Compares linear models, tree models, and a PyTorch neural network using expanding-window training and the Sharpe ratio of long-short portfolios.

Based on Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*.

## Models

OLS, Elastic Net, PCA regression, Random Forest, Gradient Boosting, and a feed-forward neural network.

## Run

Requires Python 3.10+. Download `datashare.csv` from Dacheng Xiu's website and place it in `data/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The pipeline preprocesses the data, trains models using expanding time windows, and evaluates portfolios that buy the top 10% and short the bottom 10% of stocks by predicted return.
