# Apple Stock Price Prediction Using LSTM

A deep learning project that forecasts Apple (AAPL) stock closing prices using a Long Short-Term Memory (LSTM) neural network built with Keras. The notebook walks through loading historical price data, scaling it, training an LSTM, tuning its hyperparameters, and comparing the predictions with the actual prices.

> **Disclaimer:** This project is for learning and demonstration only. It is **not** financial advice, and the model should not be used to make real trading or investment decisions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model](#model)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Possible Improvements](#possible-improvements)
- [Acknowledgements](#acknowledgements)

## Overview

| | |
|---|---|
| **Task** | Time-series regression: predict the daily closing price (`Close`) of AAPL |
| **Model** | LSTM neural network (Keras `Sequential` with `LSTM` and `Dense` layers) |
| **Scaling** | `MinMaxScaler` from scikit-learn |
| **Notebook** | `hyperparameter.ipynb` |

Stock prices are sequential: today's price depends on the days before it. LSTMs are a type of recurrent neural network designed to learn patterns across sequences, which makes them a common choice for time-series forecasting.

## Dataset

The data is stored in `apple_stock_prices.csv` and contains daily trading records for Apple (ticker `AAPL`).

- **Rows:** 3,019
- **Columns:** 7
- **Date range:** 2006-01-03 to 2017-12-29

| Column | Description |
|---|---|
| `Date` | Trading date |
| `Open` | Price at market open |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |
| `Close` | Price at market close (**the value being predicted**) |
| `Volume` | Number of shares traded |
| `Name` | Ticker symbol (`AAPL`) |

Sample of the data:

| Date | Open | High | Low | Close | Volume | Name |
|---|---:|---:|---:|---:|---:|---|
| 2006-01-03 | 10.34 | 10.68 | 10.32 | 10.68 | 201,853,036 | AAPL |
| 2006-01-04 | 10.73 | 10.85 | 10.64 | 10.71 | 155,225,609 | AAPL |
| 2006-01-05 | 10.69 | 10.70 | 10.54 | 10.63 | 112,396,081 | AAPL |
| ... | ... | ... | ... | ... | ... | ... |
| 2017-12-28 | 171.00 | 171.85 | 170.48 | 171.08 | 16,480,187 | AAPL |
| 2017-12-29 | 170.52 | 170.59 | 169.22 | 169.23 | 25,999,922 | AAPL |

Data source: _add the source of the dataset here (for example, the Kaggle dataset link)._

## Approach

1. **Load the data** from `apple_stock_prices.csv` with pandas.
2. **Select the closing price** as the series to forecast.
3. **Scale the values** to the range 0 to 1 with `MinMaxScaler`, which helps the LSTM train faster and more stably.
4. **Split into training and validation sets.** The first 2,416 rows (about 80%) are used for training, and the remaining 603 rows (about 20%, roughly mid-2015 to the end of 2017) are held out for validation. The split is chronological, so the model never sees future data during training.
5. **Create input sequences.** Each training sample is a window of past closing prices (_add window size, e.g. 60 days_) with the next day's price as the target.
6. **Build and train the LSTM** and tune its hyperparameters.
7. **Predict and inverse-transform.** Predictions are converted back from the scaled range into dollar prices and compared with the actual closing prices.

## Model

The network is built with Keras using a `Sequential` model with `LSTM` and `Dense` layers.

| Setting | Value |
|---|---|
| LSTM layers | _add number of layers and units_ |
| Dense layers | _add layers and units_ |
| Optimizer | _add (e.g. Adam)_ |
| Loss | _add (e.g. mean squared error)_ |
| Epochs | _add_ |
| Batch size | _add_ |
| Window size | _add_ |

### Hyperparameter tuning

`hyperparameter.ipynb` explores different hyperparameter settings. The values tried and the best configuration found:

| Hyperparameter | Values tried | Best |
|---|---|---|
| _e.g. LSTM units_ | _add_ | _add_ |
| _e.g. batch size_ | _add_ | _add_ |
| _e.g. epochs_ | _add_ | _add_ |

## Results

| Metric | Validation |
|---|---|
| RMSE | _add value_ |
| MAE | _add value_ |

### Predicted vs. actual price

![Actual vs Predicted AAPL closing price](images/predictions.png)

What the plot shows:

- The predicted line (orange) **follows the overall trend and the day-to-day movements** of the actual closing price (blue) closely across the validation period.
- The model reproduces both the dip in the early part of the period and the strong rally toward the end.
- In the later, steeply rising part of the period, the predictions tend to sit **slightly below** the actual prices. This is typical for LSTMs forecasting an upward trend that reaches price levels higher than most of what the model saw in training.
- Like most price-forecasting models, the predictions largely follow the previous days' movement, so they can lag behind sudden changes.

## Project Structure

```
Apple-Stock-Price-Prediction-Using-LSTM/
├── README.md
├── apple_stock_prices.csv     # historical AAPL daily prices
├── hyperparameter.ipynb       # main notebook: data prep, LSTM, tuning, plots
├── requirements.txt           # (optional) Python dependencies
└── images/
    └── predictions.png        # plot used in this README
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/kidnextdoor57/Apple-Stock-Price-Prediction-Using-LSTM.git
   cd Apple-Stock-Price-Prediction-Using-LSTM
   ```

2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
   ```

3. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook hyperparameter.ipynb
   ```

4. Run all cells from top to bottom. Keep `apple_stock_prices.csv` in the same folder as the notebook, since it is loaded with a relative path:

   ```python
   df = pd.read_csv('apple_stock_prices.csv')
   ```

### Troubleshooting

- When importing Keras you may see a `FutureWarning` about `np.object`. It comes from a library dependency, is harmless, and does not affect the results.
- Results can vary slightly between runs because neural network training involves randomness. Set random seeds for reproducible results.

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow (includes Keras)
- jupyter

## Possible Improvements

- Compare the LSTM against a simple **baseline** (such as predicting that tomorrow's price equals today's) to confirm the model adds real value.
- Use additional features such as `Open`, `High`, `Low`, `Volume`, or technical indicators (moving averages, RSI).
- Try early stopping, dropout, and regularization to reduce overfitting.
- Evaluate with walk-forward (rolling) validation instead of a single train/validation split.
- Forecast daily returns instead of raw prices, which is often easier for a model to learn.
- Compare against other architectures such as GRU, a bidirectional LSTM, or a Transformer, and against classical models like ARIMA.
- Extend the forecast to multiple days ahead.

## Acknowledgements

- Historical stock data: _add the dataset source_.
- Built with [TensorFlow/Keras](https://keras.io/), [scikit-learn](https://scikit-learn.org/), and [pandas](https://pandas.pydata.org/).
