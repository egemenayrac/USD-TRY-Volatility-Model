"""
Improved USD/TRY volatility-regime classification pipeline.

Key fixes over the original notebook:
  1. NO test-set threshold tuning. Decision threshold is chosen from out-of-fold
     (time-series CV) probabilities on the TRAIN set only, then frozen.
  2. Adds the features the paper claims but never implemented: lagged volatility
     (1m/2m/3m), rolling vol stats, EWMA vol, volatility momentum, return
     autocorrelation, macro changes/lags.
  3. Adds honest baselines: a naive persistence rule + a real Logistic Regression.
  4. Adds walk-forward (expanding-window) out-of-sample backtest.
  5. Reports accuracy, precision, recall, F1 (class-1 & macro), ROC-AUC, PR-AUC.
All randomness is seeded. Run top-to-bottom: `python3 improved_pipeline.py`.
"""

import warnings, json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_predict
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)
np.seterr(all="ignore")

# ============================================================
# 1) DAILY DATA  ->  monthly price-derived features
# ============================================================
df = (pd.read_excel("data/USD_TRY.xlsx")
        .rename(columns={"Tarih": "date", "TP DK USD S YTL": "price"}))
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(",", ".", regex=False),
                            errors="coerce")
df["is_holiday"] = df["price"].isna().astype(int)
df.loc[df.index[0], "price"] = df.loc[df.index[1], "price"]
df["price"] = df["price"].ffill()
df = df.set_index("date").sort_index()
df = df[df["is_holiday"] == 0].copy()
df["daily_ret"] = df["price"].pct_change()

# technicals
df["SMA50"] = df["price"].rolling(50).mean()
df["MA_GAP50"] = (df["price"] / df["SMA50"] - 1) * 100
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)
df["RSI14"] = rsi(df["price"])
ema_f = df["price"].ewm(span=12, adjust=False).mean()
ema_s = df["price"].ewm(span=26, adjust=False).mean()
macd = ema_f - ema_s
df["MACD_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

def longest_streak(series, positive=True):
    mx = st = 0
    for v in series.dropna():
        if (v > 0) if positive else (v < 0):
            st += 1
        else:
            mx, st = max(mx, st), 0
    return max(mx, st)

def autocorr1(x):
    x = x.dropna()
    return x.autocorr(lag=1) if len(x) > 3 else np.nan

g = pd.Grouper(freq="ME")
M = pd.DataFrame(index=df.resample("ME").last().index)
M["vol_1m"]        = df["daily_ret"].groupby(g).std()
M["ann_vol_1m_%"]  = M["vol_1m"] * np.sqrt(252) * 100
M["mean_ret_1m"]   = df["daily_ret"].groupby(g).mean()
M["skew_1m"]       = df["daily_ret"].groupby(g).apply(skew)
M["kurt_1m"]       = df["daily_ret"].groupby(g).apply(kurtosis)
M["max_ret_1m"]    = df["daily_ret"].groupby(g).max()
M["min_ret_1m"]    = df["daily_ret"].groupby(g).min()
M["range_1m"]      = M["max_ret_1m"] - M["min_ret_1m"]          # rolling max-min range
M["absret_1m"]     = df["daily_ret"].abs().groupby(g).mean()    # mean abs return (vol proxy)
M["autocorr_1m"]   = df["daily_ret"].groupby(g).apply(autocorr1)
M["streak_up_1m"]  = df["daily_ret"].groupby(g).apply(lambda x: longest_streak(x, True))
M["streak_down_1m"]= df["daily_ret"].groupby(g).apply(lambda x: longest_streak(x, False))
for c in ["RSI14", "MA_GAP50", "MACD_hist"]:
    M[c + "_last"] = df[c].resample("ME").last()

# label on annualized monthly vol (>=10% -> high), exactly as original
M["high_vol"] = (M["ann_vol_1m_%"] >= 10).astype(int)

# ============================================================
# 2) MACRO merge (from the already-aligned monthly.csv)
# ============================================================
macro = pd.read_csv("data/monthly.csv", index_col="date", parse_dates=True)
macro_cols = ["ECSU", "vix_close", "Annual_CPI", "Oil",
              "Current_Account", "Gold_Reserve", "Forex_Reserve"]
M = M.join(macro[macro_cols])

# ============================================================
# 3) ENGINEERED lag / rolling / momentum features  (the paper's claimed set)
# ============================================================
# volatility clustering: lags of annualized vol
for k in (1, 2, 3):
    M[f"ann_vol_lag{k}"] = M["ann_vol_1m_%"].shift(k)
M["vol_roll3_mean"] = M["ann_vol_1m_%"].rolling(3).mean()
M["vol_roll3_std"]  = M["ann_vol_1m_%"].rolling(3).std()
M["vol_roll6_mean"] = M["ann_vol_1m_%"].rolling(6).mean()
M["vol_ewm3"]       = M["ann_vol_1m_%"].ewm(span=3, adjust=False).mean()
M["vol_momentum"]   = M["ann_vol_1m_%"] - M["ann_vol_1m_%"].shift(1)
M["high_vol_streak"]= M["high_vol"].groupby((M["high_vol"] != M["high_vol"].shift()).cumsum()).cumcount() + 1
# macro dynamics
M["vix_lag1"]   = M["vix_close"].shift(1)
M["vix_chg"]    = M["vix_close"].pct_change()
M["vix_roll3"]  = M["vix_close"].rolling(3).mean()
M["cpi_chg"]    = M["Annual_CPI"].diff()
M["oil_chg"]    = M["Oil"].pct_change()
M["forex_chg"]  = M["Forex_Reserve"].pct_change()
M["gold_chg"]   = M["Gold_Reserve"].pct_change()

M = M.drop(columns=["vol_1m"])
M = M.dropna().copy()

# ============================================================
# 4) TARGET = next-month regime
# ============================================================
M["high_vol_next"] = M["high_vol"].shift(-1)
M = M.dropna(subset=["high_vol_next"])
y = M["high_vol_next"].astype(int)
X = M.drop(columns=["high_vol_next"])          # keep high_vol & ann_vol as predictors
print(f"Samples: {len(X)} | features: {X.shape[1]} | positives: {int(y.sum())} ({y.mean():.1%})")

# ============================================================
# 5) TEMPORAL SPLIT  (last 48 months = untouched test)
# ============================================================
HOLD = 48
X_tr, y_tr = X.iloc[:-HOLD], y.iloc[:-HOLD]
X_te, y_te = X.iloc[-HOLD:], y.iloc[-HOLD:]
print(f"Train {len(X_tr)}  |  Test {len(X_te)} (pos in test: {int(y_te.sum())})")

cv = TimeSeriesSplit(n_splits=5)

def pick_threshold_oof(estimator, Xtr, ytr):
    """Threshold maximizing macro-F1 on pooled time-series validation folds (no test leak).
    TimeSeriesSplit is not a full partition, so we loop folds manually instead of
    using cross_val_predict."""
    yv, pv = [], []
    for tr_idx, va_idx in cv.split(Xtr):
        est = clone(estimator).fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
        pv.append(est.predict_proba(Xtr.iloc[va_idx])[:, 1])
        yv.append(ytr.iloc[va_idx].values)
    yv = np.concatenate(yv); pv = np.nan_to_num(np.concatenate(pv))
    ths = np.linspace(0.15, 0.85, 71)
    f1s = [f1_score(yv, pv > t, average="macro") for t in ths]
    return float(ths[int(np.argmax(f1s))])

def evaluate(name, estimator, grid=None):
    est = estimator
    if grid:
        gs = GridSearchCV(estimator, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr); est = gs.best_estimator_
        best = {k: round(v, 4) if isinstance(v, float) else v for k, v in gs.best_params_.items()}
    else:
        est.fit(X_tr, y_tr); best = {}
    thr = pick_threshold_oof(clone(est), X_tr, y_tr)
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_te)[:, 1]
    pred = (proba > thr).astype(int)
    row = dict(model=name, threshold=round(thr, 3),
               accuracy=round(accuracy_score(y_te, pred), 3),
               macro_f1=round(f1_score(y_te, pred, average="macro"), 3),
               f1_high=round(f1_score(y_te, pred, pos_label=1, zero_division=0), 3),
               precision_high=round(precision_score(y_te, pred, pos_label=1, zero_division=0), 3),
               recall_high=round(recall_score(y_te, pred, pos_label=1, zero_division=0), 3),
               roc_auc=round(roc_auc_score(y_te, proba), 3),
               pr_auc=round(average_precision_score(y_te, proba), 3),
               best_params=best)
    return row, est, proba, pred

# ============================================================
# 6) BASELINES
# ============================================================
results = []
# 6a) persistence: next regime = current regime
pred_p = X_te["high_vol"].astype(int).values
results.append(dict(model="Persistence (naive)", threshold="-",
    accuracy=round(accuracy_score(y_te, pred_p), 3),
    macro_f1=round(f1_score(y_te, pred_p, average="macro"), 3),
    f1_high=round(f1_score(y_te, pred_p, pos_label=1, zero_division=0), 3),
    precision_high=round(precision_score(y_te, pred_p, pos_label=1, zero_division=0), 3),
    recall_high=round(recall_score(y_te, pred_p, pos_label=1, zero_division=0), 3),
    roc_auc="-", pr_auc="-", best_params={}))

# 6b) Logistic Regression (scaled) — real baseline the paper claims
logit = Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=5000, solver="liblinear",
                                            class_weight="balanced", random_state=RNG))])
r, *_ = evaluate("Logistic Regression", logit,
                 grid={"lr__C": [0.01, 0.05, 0.1, 0.5, 1.0]})
results.append(r)

# ============================================================
# 7) TREE MODELS
# ============================================================
ratio = (y_tr == 0).sum() / (y_tr == 1).sum()

r_brf, *_ = evaluate("Balanced Random Forest",
    BalancedRandomForestClassifier(random_state=RNG, n_jobs=-1,
        sampling_strategy="all", replacement=True, bootstrap=False),
    grid={"n_estimators": [200, 400], "max_depth": [3, 4, 5],
          "min_samples_leaf": [3, 5]})
results.append(r_brf)

r_lgb, lgb_est, lgb_proba, lgb_pred = evaluate("LightGBM",
    LGBMClassifier(objective="binary", subsample=0.8, colsample_bytree=0.7,
                   random_state=RNG, n_jobs=-1, verbose=-1),
    grid={"n_estimators": [300, 500], "learning_rate": [0.03, 0.05],
          "num_leaves": [7, 15], "max_depth": [3, 4],
          "scale_pos_weight": [ratio, ratio * 1.5]})
results.append(r_lgb)

r_xgb, *_ = evaluate("XGBoost",
    XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=RNG, n_jobs=-1),
    grid={"n_estimators": [300, 500], "learning_rate": [0.03, 0.05],
          "max_depth": [2, 3], "subsample": [0.8], "colsample_bytree": [0.7],
          "scale_pos_weight": [ratio, ratio * 1.5]})
results.append(r_xgb)

# ============================================================
# 8) WALK-FORWARD backtest (expanding window over last 48 months)
# ============================================================
def walk_forward(make_est, thr=0.5):
    preds, probas, ys = [], [], []
    for i in range(len(X) - HOLD, len(X)):
        est = make_est()
        est.fit(X.iloc[:i], y.iloc[:i])
        p = est.predict_proba(X.iloc[[i]])[:, 1][0]
        probas.append(p); preds.append(int(p > thr)); ys.append(int(y.iloc[i]))
    ys = np.array(ys); preds = np.array(preds); probas = np.array(probas)
    return dict(model="LightGBM (walk-forward)", threshold=thr,
        accuracy=round(accuracy_score(ys, preds), 3),
        macro_f1=round(f1_score(ys, preds, average="macro"), 3),
        f1_high=round(f1_score(ys, preds, pos_label=1, zero_division=0), 3),
        precision_high=round(precision_score(ys, preds, pos_label=1, zero_division=0), 3),
        recall_high=round(recall_score(ys, preds, pos_label=1, zero_division=0), 3),
        roc_auc=round(roc_auc_score(ys, probas), 3),
        pr_auc=round(average_precision_score(ys, probas), 3), best_params={})

wf_params = lgb_est.get_params()
results.append(walk_forward(lambda: LGBMClassifier(**wf_params), thr=r_lgb["threshold"]))

# ============================================================
# 9) REPORT
# ============================================================
res_df = pd.DataFrame(results)[["model", "threshold", "accuracy", "macro_f1",
    "f1_high", "precision_high", "recall_high", "roc_auc", "pr_auc"]]
pd.set_option("display.width", 160)
print("\n================= HOLD-OUT RESULTS (last 48 months, leak-free threshold) =================")
print(res_df.to_string(index=False))

print("\nBest LightGBM params:", r_lgb["best_params"])
print("\nLightGBM hold-out confusion matrix (rows=true [low,high]):")
print(confusion_matrix(y_te, lgb_pred))
print("\nLightGBM classification report:")
print(classification_report(y_te, lgb_pred, digits=3, zero_division=0))

# feature importances
imp = pd.Series(lgb_est.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top-12 LightGBM feature importances (gain):")
print(imp.head(12).to_string())

res_df.to_csv("results/improved_results.csv", index=False)
imp.to_csv("results/feature_importance.csv")
print("\nSaved -> results/improved_results.csv , results/feature_importance.csv")

# ============================================================
# 10) FIGURES for the paper
# ============================================================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay

# 10a) regime-shaded volatility series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(M.index, M["ann_vol_1m_%"], lw=1, color="#1f77b4")
ax.axhline(10, color="red", ls="--", lw=1, label="threshold = 10%")
ax.fill_between(M.index, 0, M["ann_vol_1m_%"].max(), where=M["high_vol"] == 1,
                color="orange", alpha=0.12, label="high-vol regime")
ax.set_title("USD/TRY Monthly Annualized Volatility with High/Low Regimes")
ax.set_ylabel("Annualized volatility (%)"); ax.set_xlabel("Date"); ax.legend()
fig.tight_layout(); fig.savefig("results/regime_series.png", dpi=130); plt.close(fig)

# 10b) confusion matrix
fig, ax = plt.subplots(figsize=(4.2, 4))
ConfusionMatrixDisplay(confusion_matrix(y_te, lgb_pred),
                       display_labels=["low", "high"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("LightGBM — hold-out confusion"); fig.tight_layout()
fig.savefig("results/confusion_lgbm.png", dpi=130); plt.close(fig)

# 10c) ROC + PR curves
fpr, tpr, _ = roc_curve(y_te, lgb_proba)
prec, rec, _ = precision_recall_curve(y_te, lgb_proba)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(fpr, tpr, lw=2, label=f"AUC={r_lgb['roc_auc']}"); axes[0].plot([0, 1], [0, 1], "k--", lw=.8)
axes[0].set_title("ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
axes[1].plot(rec, prec, lw=2, color="purple", label=f"AP={r_lgb['pr_auc']}")
axes[1].set_title("Precision-Recall"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
fig.tight_layout(); fig.savefig("results/roc_pr.png", dpi=130); plt.close(fig)

# 10d) feature importance
fig, ax = plt.subplots(figsize=(7, 5))
imp.head(15)[::-1].plot.barh(ax=ax, color="#2ca02c")
ax.set_title("LightGBM feature importance (gain) — top 15"); ax.set_xlabel("gain")
fig.tight_layout(); fig.savefig("results/feature_importance.png", dpi=130); plt.close(fig)

# 10e) model comparison bars
cmp = res_df.set_index("model")[["accuracy", "macro_f1", "f1_high"]]
fig, ax = plt.subplots(figsize=(9, 4.5))
cmp.plot.bar(ax=ax); ax.set_ylim(0.4, 0.95); ax.set_ylabel("score")
ax.set_title("Model comparison on 48-month hold-out (leak-free)")
ax.legend(loc="lower right"); plt.xticks(rotation=20, ha="right")
fig.tight_layout(); fig.savefig("results/model_comparison.png", dpi=130); plt.close(fig)
print("Saved 5 figures -> results/*.png")
