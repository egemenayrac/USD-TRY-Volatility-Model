# Classification of USD/TRY Volatility Regimes: A Machine-Learning Approach

**Egemen Ayraç — Boğaziçi University**

---

## Abstract

This study classifies high- and low-volatility regimes of the USD/TRY exchange rate using
supervised machine learning. The problem is framed as a binary classification task: for each
month I compute the annualized volatility of daily returns (the standard deviation of daily
returns scaled by √252 and expressed in percent) and label the month *high-volatility* when this
value is at least 10% and *low-volatility* otherwise. Using daily USD/TRY data from 2006 to 2024
aggregated to 222 monthly observations, I engineer price-based features (lagged and rolling
volatility, return moments, technical indicators) together with macro-financial variables (VIX,
CPI, Brent oil, current account, FX and gold reserves, and an economic-policy-uncertainty index),
and predict next month's regime.

I compare a naive persistence benchmark, Logistic Regression, Balanced Random Forest, XGBoost and
LightGBM. Models are tuned with time-series cross-validation, and the decision threshold is
selected on out-of-fold validation data only, never on the test set. On an untouched 48-month
hold-out, LightGBM reaches **0.854 accuracy, 0.833 macro-F1 and 0.899 ROC-AUC**, clearly beating
both the persistence benchmark (0.792 accuracy) and the linear baseline. An expanding-window
walk-forward backtest confirms the result (0.833 accuracy, 0.903 ROC-AUC). The findings show that
monthly USD/TRY volatility regimes are predictable out-of-sample from price-based and
macro-financial features, and that gradient-boosted trees add genuine value over a persistence
rule.

---

## 1. Introduction

Exchange rates are central to trade competitiveness, inflation, capital flows and the transmission
of monetary policy. For emerging markets, currency volatility raises the cost of external debt and
undermines price stability and investor confidence. The Turkish lira (TRY) has been among the most
volatile currencies of the past two decades, with episodes of rapid depreciation and sharp
corrections triggered by macroeconomic shocks, political developments and shifts in global risk
appetite. Anticipating transitions in exchange-rate behaviour is therefore valuable to market
participants, policymakers and researchers.

Classical foreign-exchange models—ARIMA, GARCH and Markov-switching frameworks—rely on assumptions
of linearity, stationarity or fixed regime persistence and adapt poorly to abrupt, nonlinear
structural breaks. Machine-learning methods relax these assumptions and can learn nonlinear
interactions directly from data. Rather than forecasting the exchange-rate *level*, I classify the
*volatility regime* of the coming month as high or low, based on statistical properties of the
recent return distribution and on macro-financial context.

The approach is designed to be interpretable, reproducible and honestly evaluated: hyperparameters
are chosen by time-series cross-validation, the decision threshold is fixed on validation data, and
the model is tested once on a four-year hold-out and again through walk-forward backtesting. A naive
persistence rule serves as the benchmark every model must beat.

---

## 2. A Brief Review of the Literature

Exchange-rate modelling has traditionally relied on econometric tools such as ARIMA, GARCH and
Markov-switching models (Engel & Hamilton, 1990; Meese & Rogoff, 1983; Hamilton, 1989), which
assume linear relationships and often struggle with the non-stationary, nonlinear behaviour of
emerging-market currencies. Tree-based machine-learning methods—Random Forests (Breiman, 2001),
XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke et al., 2017)—have gained traction for their
flexibility, non-parametric nature and robustness to noisy financial data, and have been applied to
regime switching and volatility prediction (Krauss et al., 2017; Gu et al., 2020). A recurring theme
is the value of feature engineering from price data—lagged returns, rolling statistics and
volatility measures—for classification accuracy (Bao et al., 2017). This study follows that line,
combining price-based signals with macro-financial context for the USD/TRY pair.

---

## 3. Methods and the Model

The pipeline has four stages: data preparation, feature engineering, model selection, and the
classification/evaluation process.

### 3.1 Data Preparation

I collected **daily** USD/TRY (selling) quotes from the CBRT covering **January 2006 to December
2024** (6,940 calendar days; the sample begins in 2006 to limit the influence of the February-2001
crisis and the transition to a free float). Non-trading days were flagged and removed, and daily
simple returns were computed. The series was then aggregated to a monthly frequency.

Monthly volatility is the standard deviation of that month's daily returns, annualized as
`vol × √252 × 100`. A month is labelled **high-volatility** when annualized volatility ≥ 10% and
**low-volatility** otherwise. This threshold sits near the sample median, so the two classes are
close to balanced (50.5% high), which is appropriate for a regime-discrimination task.
Macro-financial series—Turkey's annual CPI (TÜİK); current account, gold reserves and FX reserves
(CBRT); Brent crude oil; the CBOE VIX; and the Economic Country-Specific Uncertainty (ECSU) index
for Türkiye (Kılıç & Ballı, 2024)—were merged on month-end dates and forward-filled where needed.
After computing lagged features and dropping rows with missing values, **222 monthly observations**
remain.

### 3.2 Feature Engineering

I extract **38 features** in two groups.

**Price-derived (technical):**
- annualized monthly volatility and its 1-, 2- and 3-month lags;
- rolling volatility mean (3m, 6m), rolling volatility standard deviation (3m), and an
  exponentially-weighted volatility (span 3);
- volatility momentum (month-over-month change) and the current regime's run length
  (`high_vol_streak`);
- monthly mean, skewness and kurtosis of daily returns; maximum and minimum daily return and their
  range; mean absolute return; lag-1 autocorrelation of daily returns;
- longest consecutive up/down daily-return streaks within the month;
- end-of-month technical indicators: RSI(14), the gap to the 50-day moving average (MA-gap), and the
  MACD histogram.

**Macro-financial:** VIX level, its 1-month lag, month-over-month change and 3-month mean; annual CPI
and its change; Brent oil level and change; current account; gold reserves; FX reserves and their
change; and the ECSU index.

For the Logistic Regression baseline, features are standardized with z-scores inside the training
pipeline; tree-based models are scale-invariant and use the raw features. All transformations are
fit on training data only.

### 3.3 Model Selection

I evaluate five predictors:
1. **Persistence (naive benchmark)** — predicts next month's regime equal to the current month's.
   Because volatility clusters, this is a strong, honest baseline that any useful model must beat.
2. **Logistic Regression** — a regularized linear baseline (class-balanced, z-scored features).
3. **Balanced Random Forest** — a bagged tree ensemble with class balancing.
4. **XGBoost** — gradient-boosted trees.
5. **LightGBM** — gradient-boosted trees, selected for strong performance on tabular data.

Tree models use `scale_pos_weight`/class balancing to handle the mild class imbalance.
Implementations use scikit-learn, imbalanced-learn, xgboost and lightgbm, all seeded for
reproducibility.

### 3.4 Classification and Evaluation Process

The target is the **next month's** regime, so the task is genuinely predictive. I hold out the
**last 48 months** as a test set that is never seen during training or tuning. On the remaining 174
months I tune hyperparameters with a 5-fold `TimeSeriesSplit` grid search (scoring class-1 F1). The
decision threshold is then selected to maximize macro-F1 on **pooled out-of-fold validation
predictions**, not on the test set, which keeps the test evaluation unbiased. Finally I evaluate
once on the hold-out and additionally run an **expanding-window walk-forward backtest**, retraining
each month across the 48-month window. Reported metrics are accuracy, precision, recall, F1 (per
class and macro), ROC-AUC and PR-AUC.

---

## 4. The Data

- **USD/TRY exchange rate** — daily CBRT selling quotes, 2006–2024; basis for the regime label and
  all price-derived features.
- **CPI (TÜİK)** — annual inflation, capturing domestic price pressure.
- **CBRT series** — current account (external position), gold reserves and FX reserves (reserve
  adequacy / defence capacity).
- **Brent crude oil** — global factor for Turkey's trade and energy-import bill.
- **VIX (CBOE)** — global risk aversion and investor sentiment.
- **ECSU index (Kılıç & Ballı, 2024)** — newspaper-based economic-policy-uncertainty measure for
  Türkiye, built in the spirit of Baker et al. (2016).

All series are aligned to month-end and resampled to monthly frequency.

---

## 5. Empirical Results

**Table 1. Hold-out performance (last 48 months; threshold fixed on validation data).**

| Model | Accuracy | Macro-F1 | High-vol F1 | Precision (high) | Recall (high) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Persistence (naive) | 0.792 | 0.758 | 0.667 | 0.667 | 0.667 | – | – |
| Logistic Regression | 0.542 | 0.529 | 0.450 | 0.360 | 0.600 | 0.634 | 0.470 |
| Balanced Random Forest | 0.812 | 0.798 | 0.743 | 0.650 | 0.867 | 0.873 | 0.708 |
| XGBoost | 0.833 | 0.798 | 0.714 | 0.769 | 0.667 | 0.885 | 0.704 |
| **LightGBM** | **0.854** | **0.833** | **0.774** | 0.750 | 0.800 | **0.899** | 0.725 |
| LightGBM (walk-forward) | 0.833 | 0.812 | 0.750 | 0.706 | 0.800 | 0.903 | 0.758 |

LightGBM is the best single model, with the highest accuracy, macro-F1 and ROC-AUC, and a balanced
precision/recall on the high-volatility class. Its best configuration is `learning_rate = 0.03`,
`max_depth = 3`, `n_estimators = 300`, `num_leaves = 7`, `scale_pos_weight ≈ 1.19`. The hold-out
confusion matrix is

```
          pred low   pred high
true low      29          4
true high      3         12
```

**Feature importance (LightGBM gain).** The most influential features are the gap to the 50-day
moving average (MA-gap), Brent oil, the month-over-month change in CPI, the previous month's maximum
daily return, RSI(14), gold reserves, the 3-month rolling volatility dispersion, the monthly mean
return, the MACD histogram, the current regime run length, the down-streak, and the lag-1 return
autocorrelation. Regime shifts are thus driven by a blend of trend/deviation signals, recent extreme
returns, short-horizon volatility persistence, and macro context (oil and inflation dynamics).

*Figures produced by the pipeline:* `results/regime_series.png`, `results/model_comparison.png`,
`results/confusion_lgbm.png`, `results/roc_pr.png`, `results/feature_importance.png`.

---

## 6. Key Findings

- **Monthly USD/TRY volatility regimes are predictable out-of-sample.** LightGBM reaches 0.854
  accuracy and 0.899 ROC-AUC on a four-year hold-out it never saw during training or tuning.
- **The model adds value beyond the obvious.** It clearly beats the persistence benchmark
  (0.854 vs 0.792 accuracy; 0.774 vs 0.667 high-vol F1), so it captures more than the trivial
  autocorrelation of volatility.
- **The result is robust, not a lucky split.** An expanding-window walk-forward backtest that
  retrains every month reaches essentially the same level (0.833 accuracy, 0.903 ROC-AUC).
- **Nonlinearity matters.** The linear Logistic Regression baseline is weak (0.542 accuracy),
  confirming that gradient-boosted trees are better suited to this decision boundary.
- **Drivers are intuitive and price-led.** Trend/deviation indicators, recent extreme daily returns
  and short-horizon volatility persistence dominate, with oil and inflation dynamics providing
  macro context.
- **The evaluation is leak-free.** Tuning the threshold on validation data (not the test set) means
  the reported numbers reflect genuine out-of-sample performance.

---

## 7. Limitations and Future Work

The 10% volatility cut is fixed; time-varying thresholds based on rolling quantiles of the
volatility distribution would keep the labels meaningful through crisis periods when "normal"
volatility itself rises. Additional inputs—the CBRT one-week repo rate and the BIST100 index—and
higher-frequency (weekly/daily) targets are natural extensions. Sentiment features from central-bank
communications or financial news, captured with FinBERT-style embeddings, could add qualitative
signal, and sequence architectures (LSTM, TCN) or probability calibration could further refine the
predictions.

---

## 8. Conclusion

I classified USD/TRY volatility regimes as a binary prediction problem using features engineered from
daily prices and a compact set of macro-financial indicators. With time-series cross-validation, a
leak-free decision threshold and a walk-forward backtest, LightGBM predicted next-month regimes with
about 0.85 accuracy and 0.90 ROC-AUC out-of-sample, beating both a persistence benchmark and a linear
baseline. While the study does not exhaust the complexity of exchange-rate behaviour, it shows that
disciplined, interpretable machine learning can complement traditional econometric tools for
monitoring FX-market risk in emerging economies.

---

## References

1. Kılıç, İ., & Ballı, F. (2024). *Measuring economic country-specific uncertainty in Türkiye.* Empirical Economics, 67, 1649–1689.
2. Meese, R., & Rogoff, K. (1983). *Empirical exchange rate models of the seventies.* Journal of International Economics, 14(1), 3–24.
3. Engel, C., & Hamilton, J. D. (1990). *Long swings in the dollar.* American Economic Review, 80(4), 689–713.
4. Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series.* Econometrica, 57(2), 357–384.
5. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5–32.
6. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD '16, 785–794.
7. Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.
8. Krauss, C., Do, X. A., & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500.* EJOR, 259(2), 689–702.
9. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical asset pricing via machine learning.* Review of Financial Studies, 33(5), 2223–2273.
10. Bao, W., Yue, J., & Rao, Y. (2017). *A deep learning framework for financial time series.* PLoS ONE, 12(7), e0180944.
11. Baker, S. R., Bloom, N., & Davis, S. J. (2016). *Measuring economic policy uncertainty.* Quarterly Journal of Economics, 131(4), 1593–1636.
12. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
