# Classification of USD/TRY Volatility Regimes: A Machine-Learning Approach
## Detailed Technical Report

**Egemen Ayraç — Boğaziçi University**

---

## Contents

1. Abstract
2. Introduction and Motivation
3. Related Work
4. Data
5. Labelling: Defining Volatility Regimes
6. Feature Engineering
7. Models
8. Evaluation Design (and why it is leak-free)
9. Results
10. Discussion
11. Limitations
12. Future Work
13. Reproducibility
14. References

---

## 1. Abstract

This report studies whether the volatility *regime* of the USD/TRY exchange rate one month
ahead can be predicted from historical price behaviour and macro-financial context. Each month
is labelled *high-* or *low-volatility* according to the annualized standard deviation of its
daily returns, and the task is cast as binary classification of the *next* month's regime. Using
daily CBRT data from 2006 to 2024 (222 monthly observations after feature construction), I
engineer 38 features and compare a naive persistence benchmark, Logistic Regression, Balanced
Random Forest, XGBoost and LightGBM. The evaluation is deliberately conservative: hyperparameters
are tuned with time-series cross-validation, the classification threshold is fixed on out-of-fold
validation data, and the model is then tested once on an untouched 48-month hold-out and again
through expanding-window walk-forward backtesting. LightGBM achieves 0.854 accuracy, 0.833
macro-F1 and 0.899 ROC-AUC out-of-sample, clearly beating the persistence benchmark (0.792
accuracy) and the linear baseline. The work demonstrates that disciplined, interpretable ML can
predict emerging-market FX volatility regimes with useful accuracy, and—just as importantly—how
to evaluate such a model without fooling oneself.

---

## 2. Introduction and Motivation

Exchange rates sit at the centre of trade competitiveness, inflation, capital flows and monetary
transmission. For emerging markets the stakes are higher: currency volatility raises the cost of
external borrowing, complicates corporate hedging and can erode confidence quickly. The Turkish
lira has been one of the most volatile major emerging-market currencies of the last two decades,
alternating between calm stretches and sharp depreciation episodes driven by macro shocks,
political events and global risk sentiment.

Forecasting the *level* of an exchange rate is notoriously hard; the seminal Meese–Rogoff result
(1983) showed that structural models rarely beat a random walk out-of-sample. A more tractable and
arguably more useful question for risk management is the *regime*: will the coming month be turbulent
or calm? A reliable high-volatility early warning is directly actionable for hedging decisions,
reserve management and position sizing.

This report frames that question as binary classification and answers it with a transparent,
reproducible machine-learning pipeline. Three design principles guide the work:

- **Predictive, not descriptive.** The target is always *next* month's regime, so nothing in the
  feature set peeks into the period being predicted.
- **Honest evaluation.** The threshold and hyperparameters are chosen without ever touching the
  test set, and results are cross-checked with a walk-forward backtest.
- **Beat a real baseline.** Because volatility is autocorrelated, a naive "tomorrow looks like
  today" rule is already strong. The model is only interesting if it beats that rule.

---

## 3. Related Work

Classical FX volatility modelling rests on GARCH-type conditional-variance models and on
Markov-switching frameworks (Hamilton, 1989; Engel & Hamilton, 1990), which capture volatility
clustering and discrete regimes but assume specific parametric dynamics and adapt slowly to abrupt
structural breaks. Meese & Rogoff (1983) established the difficulty of out-of-sample exchange-rate
prediction with structural models.

Machine learning relaxes parametric assumptions. Tree ensembles—Random Forests (Breiman, 2001),
XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017)—handle nonlinearities, mixed-scale
inputs and noisy financial data well, and have been applied to financial classification, regime
detection and statistical arbitrage (Krauss et al., 2017; Gu et al., 2020). A consistent finding in
this literature is that careful feature engineering from price data—lagged returns, rolling
statistics, volatility proxies—drives much of the predictive performance (Bao et al., 2017). This
report follows that tradition and adds a compact macro-financial block (global risk, inflation,
reserves, oil, policy uncertainty) tailored to an emerging-market currency.

---

## 4. Data

**USD/TRY exchange rate.** Daily selling quotes from the Central Bank of the Republic of Turkey
(CBRT), January 2006 to December 2024, 6,940 calendar rows. The sample starts in 2006 to avoid the
distortion of the 2001 crisis and the subsequent transition to a floating regime. Non-trading days
(NaN quotes) are flagged via an `is_holiday` indicator and removed before computing returns; the
first observation is back-filled.

**Macro-financial series**, all merged at month-end and forward-filled where necessary:

| Series | Source | Role |
|---|---|---|
| Annual CPI | TÜİK | domestic inflation pressure |
| Current account balance | CBRT | external position |
| Gold reserves | CBRT | reserve assets |
| FX reserves | CBRT | currency-defence capacity |
| Brent crude oil | commodity data | trade / energy-import channel |
| VIX | CBOE | global risk aversion |
| ECSU index | Kılıç & Ballı (2024) | Türkiye-specific economic-policy uncertainty |

After daily→monthly aggregation, feature construction and dropping rows with missing lagged values,
**222 monthly observations** remain.

---

## 5. Labelling: Defining Volatility Regimes

For each calendar month I compute the standard deviation of that month's daily simple returns and
annualize it:

```
ann_vol_% = std(daily returns within month) × √252 × 100
```

A month is labelled **high-volatility (1)** if `ann_vol_% ≥ 10`, else **low-volatility (0)**. The
10% cut sits close to the sample median, so the classes are nearly balanced (50.5% high). This is a
deliberate choice: a near-balanced regime-discrimination label is more informative for learning than
a rare-event label dominated by one class, and it keeps standard metrics (accuracy, F1) meaningful.

The prediction target is the **shifted** label `high_vol_next = high_vol.shift(-1)`: given everything
known at the end of month *t*, predict the regime of month *t+1*.

---

## 6. Feature Engineering

The pipeline produces **38 features** in two blocks. Everything is computed from information available
at the end of the current month, so there is no look-ahead leakage into the target.

### 6.1 Price-derived (technical)

- **Volatility & its memory:** current annualized monthly volatility; its 1-, 2- and 3-month lags;
  3- and 6-month rolling means; 3-month rolling standard deviation; an exponentially-weighted
  volatility (span 3); month-over-month volatility momentum. *Rationale:* volatility clusters, so its
  own recent history is the single most informative signal.
- **Regime memory:** the run length of the current regime (`high_vol_streak`). *Rationale:* the longer
  a regime persists, the more it conditions the transition probability.
- **Return distribution shape:** monthly mean daily return; skewness; kurtosis; maximum and minimum
  daily return; their range; mean absolute return; lag-1 autocorrelation of daily returns.
  *Rationale:* fat tails, asymmetry and large single-day moves often precede regime shifts.
- **Intramonth dynamics:** longest consecutive up- and down-return streaks.
- **End-of-month technical indicators:** RSI(14), the gap between price and its 50-day moving average
  (MA-gap, in %), and the MACD histogram. *Rationale:* trend stretch and momentum capture conditions
  that tend to break into volatility.

### 6.2 Macro-financial

VIX level, its 1-month lag, month-over-month change and 3-month mean; annual CPI and its change;
Brent oil level and change; current account; gold reserves; FX reserves and their change; and the
ECSU index. Levels capture state; changes capture shocks.

### 6.3 Scaling

For the Logistic Regression baseline, features are standardized with z-scores **inside** a pipeline
fit on training data only, so the scaler never sees validation or test statistics. Tree models are
scale-invariant and use raw features.

---

## 7. Models

| Model | Why included |
|---|---|
| **Persistence (naive)** | Predicts next regime = current regime. The benchmark every model must beat, since volatility clustering makes it strong on its own. |
| **Logistic Regression** | Regularized linear baseline (class-balanced, z-scored). Tests whether a linear boundary suffices. |
| **Balanced Random Forest** | Bagged trees with built-in class balancing; robust, low-variance. |
| **XGBoost** | Gradient-boosted trees, strong tabular learner. |
| **LightGBM** | Gradient-boosted trees optimized for speed and tabular accuracy; the primary model. |

Tree models address the mild class imbalance through class weighting / `scale_pos_weight`. All models
are seeded (`random_state = 42`).

---

## 8. Evaluation Design (and why it is leak-free)

This section is the methodological core, because in financial classification the *evaluation* is
where most optimistic bias creeps in.

**8.1 Temporal hold-out.** The last **48 months** are reserved as a test set and never used during
feature selection, hyperparameter tuning or threshold selection. The remaining 174 months form the
training set.

**8.2 Hyperparameter tuning.** On the training set, a 5-fold `TimeSeriesSplit` grid search (scoring
class-1 F1) selects hyperparameters. `TimeSeriesSplit` only ever trains on the past and validates on
the future, so no future information leaks backwards.

**8.3 Threshold selection — the key point.** A classifier outputs a probability; turning it into a
high/low decision requires a threshold. A tempting but wrong approach is to scan thresholds on the
test set and report the best one — this leaks the test labels into a model decision and inflates the
reported score. Instead, I collect **out-of-fold** validation probabilities across the time-series
folds of the training set, pick the threshold that maximizes macro-F1 there, and then **freeze** it.
The test set sees exactly one, pre-committed threshold.

**8.4 Single test evaluation + walk-forward.** The frozen model and threshold are evaluated once on
the 48-month hold-out. As an independent robustness check, an **expanding-window walk-forward
backtest** retrains the model month by month across the same window and predicts each month one step
ahead. Agreement between the two confirms the result is not an artefact of a single split.

**8.5 Metrics.** Accuracy, precision, recall and F1 (per class and macro-averaged), plus ROC-AUC and
PR-AUC, which are threshold-independent and more informative under class imbalance.

---

## 9. Results

### 9.1 Headline comparison (48-month hold-out, frozen threshold)

| Model | Accuracy | Macro-F1 | High-vol F1 | Precision (high) | Recall (high) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Persistence (naive) | 0.792 | 0.758 | 0.667 | 0.667 | 0.667 | – | – |
| Logistic Regression | 0.542 | 0.529 | 0.450 | 0.360 | 0.600 | 0.634 | 0.470 |
| Balanced Random Forest | 0.812 | 0.798 | 0.743 | 0.650 | 0.867 | 0.873 | 0.708 |
| XGBoost | 0.833 | 0.798 | 0.714 | 0.769 | 0.667 | 0.885 | 0.704 |
| **LightGBM** | **0.854** | **0.833** | **0.774** | 0.750 | 0.800 | **0.899** | 0.725 |
| LightGBM (walk-forward) | 0.833 | 0.812 | 0.750 | 0.706 | 0.800 | 0.903 | 0.758 |

Best LightGBM configuration: `learning_rate = 0.03`, `max_depth = 3`, `n_estimators = 300`,
`num_leaves = 7`, `scale_pos_weight ≈ 1.19`.

### 9.2 LightGBM confusion matrix (hold-out)

```
          pred low   pred high
true low      29          4
true high      3         12
```

Of 33 calm months, 29 are correctly identified; of 15 turbulent months, 12 are caught. The errors
are roughly symmetric, which the balanced precision/recall on the high-volatility class reflects.

### 9.3 Feature importance (LightGBM gain)

Top contributors: MA-gap (price vs 50-day average), Brent oil, month-over-month CPI change, the
previous month's maximum daily return, RSI(14), gold reserves, 3-month rolling volatility dispersion,
monthly mean return, MACD histogram, current regime run length, down-streak, and lag-1 return
autocorrelation. The signal is led by trend/deviation indicators, recent extreme returns and
short-horizon volatility persistence, with oil and inflation dynamics providing macro context.

### 9.4 Figures

- `results/regime_series.png` — annualized volatility with the 10% threshold and shaded high-vol months.
- `results/model_comparison.png` — accuracy / macro-F1 / high-vol F1 across models.
- `results/confusion_lgbm.png` — LightGBM hold-out confusion matrix.
- `results/roc_pr.png` — ROC and precision-recall curves.
- `results/feature_importance.png` — top-15 LightGBM gains.

---

## 10. Discussion

Three results carry the report.

**The model adds genuine value.** LightGBM beats the persistence benchmark on every metric
(0.854 vs 0.792 accuracy; 0.774 vs 0.667 high-vol F1). Because persistence already exploits volatility
clustering, the gap is evidence that the model captures additional structure—momentum stretch, recent
tail behaviour and macro context—rather than re-learning autocorrelation.

**The result is robust.** The walk-forward backtest, which retrains every month and is the closest
analogue to live deployment, lands at essentially the same level (0.833 accuracy, 0.903 ROC-AUC). The
single-split hold-out is therefore not a lucky cut.

**Nonlinearity matters.** Logistic Regression collapses to 0.542 accuracy—barely better than chance and
well below persistence—while tree ensembles cluster at 0.81–0.85. The decision boundary between regimes
is nonlinear and interaction-heavy, exactly the setting where gradient-boosted trees excel.

A ROC-AUC near 0.90 means the model ranks a randomly chosen turbulent month above a randomly chosen calm
month about 90% of the time, which is a strong, decision-useful separation for an early-warning tool.

---

## 11. Limitations

- **Fixed threshold for the label.** The 10% cut is static; in a structurally higher-volatility era it
  may misclassify what is locally "normal."
- **Monthly resolution.** Monthly aggregation smooths intramonth dynamics and limits sample size
  (222 observations), which constrains model complexity.
- **Macro coverage.** Some plausible drivers (policy rate, equity-market sentiment) are not in the
  current dataset; results reflect the included variables.
- **Regime shocks.** Purely price-and-macro features cannot anticipate idiosyncratic geopolitical
  shocks that move FX markets discontinuously.

---

## 12. Future Work

- **Dynamic regime thresholds** based on rolling quantiles of the volatility distribution, so labels
  stay meaningful through crises.
- **Additional inputs:** CBRT policy rate, BIST100 equity sentiment, and higher-frequency (weekly/daily)
  targets.
- **Text/sentiment features** from central-bank communications and financial news via FinBERT-style
  embeddings, testing whether shifts in tone precede turbulence.
- **Sequence models** (LSTM/TCN) and probability calibration for sharper, better-calibrated forecasts.

---

## 13. Reproducibility

```bash
pip install -r requirements.txt
python improved_pipeline.py
```

The script runs end-to-end from `data/USD_TRY.xlsx` (daily prices) and `data/monthly.csv` (aligned
macro), prints the full results table and writes metrics and figures to `results/`. All randomness is
seeded (`random_state = 42`); the time-series split, threshold rule and walk-forward loop are
deterministic, so the numbers above reproduce exactly.

---

## 14. References

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
