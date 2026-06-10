# ML Pricing

## Short Summary & Results
This project is a complete data processing pipeline harnessing over 4M rows of financial data. The goal is to compare how different models (GBTs, NNs, linear models) fair up against each other in context of correctly predicting a stocks monthly returns. The outcomes of the models predictions are measured using Sharpe ratios, which is basically a return/risk ratio. 

Neural networks were shown to be the 'best' in this environment, vastly outperforming their linear model counterparts with a ~3x Sharpe score (1.5:0.5, roughly). This project is important beyond just model testing on this particular data; you can insert really any generic data, and with a couple adjustments on the backtesting engine, you can assess how these models
fair up against each other with some benchmark too.

## How To Run
Running this pipeline is simple. You must load the data (source provided in scope) and plug it into a root/data directory. Then, you can simply load up the main Python testing folder and see the pipeline in action (in a venv with all requirements installed).
