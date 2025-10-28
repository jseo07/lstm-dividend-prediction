#!/usr/bin/env python3
"""
ETF Dividend Pipeline (EVENT-BASED): Fetch, Prep, Predict (next 5 dividends), Export for Tableau

Matches notebook logic:
- Train/predict on dividend **events only** (rows where Dividends > 0)
- Features per event t:
    Dividends_t
    Avg_Close_Prev_1_Month_t  (mean Close of the **previous** calendar month, aligned to event t)
- Sequences are the last `lookback` events (default 12).
- Prediction loop runs in **scaled space**, inverse transforms dividend using
  a row [yhat_scaled, last_avg_close_scaled] (exactly like your notebook trick).

Outputs
-------
data/etf_dividends_history.csv
    ["ticker","date","Dividends","Avg_Close_Prev_1_Month","source"]
data/etf_dividend_predictions.csv
    ["ticker","pred_index","pred_date_est","pred_dividend"]
data/etf_dividend_predictions_events.csv
    same as above but filtered to >0 predictions (event months)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please `pip install yfinance` before running this script.") from e

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None
    warnings.warn("TensorFlow not available. Predictions will be skipped.")

try:
    import joblib
except Exception:
    joblib = None
    warnings.warn("joblib not available. Will skip scaler loading and use raw values.")

DEFAULT_TICKERS = ["SPY", "VTI", "QQQ", "IWM", "EFA", "EEM", "AGG", "SCHD", "VYM", "XLK"]
EVENT_THRESHOLD = 1e-3  # predictions < this are treated as zero (non-event) in events CSV


# ---------- FETCH & FEATURE ENGINEERING (matches your notebook) ----------

def request_data(symbol: str) -> pd.DataFrame:
    """Notebook's request_data: full history (daily)."""
    t = yf.Ticker(symbol)
    df = t.history(period="max", interval="1d", auto_adjust=False).reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def filter_dividend(data: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook's filter_dividend:
    - keep rows with Dividends > 0
    - keep ['Date','Dividends']
    - convert Date to monthly period string index (we'll convert back to datetime)
    """
    df = data[data["Dividends"] > 0][["Date", "Dividends"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.to_period("M").astype(str)
    df = df.set_index("Date")
    return df


def add_stock_prices(dividend_data: pd.DataFrame, stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook's add_stock_prices (corrected to 1 month lookback, as in your code):
    For each dividend event date (month t), compute the average Close of the
    **previous** month (t-1) over calendar days.
    """
    dividend_data = dividend_data.reset_index()
    stock_data = stock_data.copy()
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.tz_localize(None)
    dividend_data["Date"] = pd.to_datetime(dividend_data["Date"]).dt.tz_localize(None)

    avg_prices = []
    for div_date in dividend_data["Date"]:
        # previous calendar month window: [first_day_prev_month, last_day_prev_month]
        start_date = (div_date - pd.DateOffset(months=1)).replace(day=1)
        end_date = div_date - pd.Timedelta(days=1)  # last day before the event month starts
        mask = (stock_data["Date"] >= start_date) & (stock_data["Date"] <= end_date)
        avg_close = stock_data.loc[mask, "Close"].mean()
        avg_prices.append(avg_close)

    out = dividend_data.copy()
    out["Avg_Close_Prev_1_Month"] = avg_prices
    return out[["Date", "Dividends", "Avg_Close_Prev_1_Month"]]


def fetch_features_events(ticker: str) -> pd.DataFrame:
    """
    End-to-end: request data -> filter_dividend -> add_stock_prices.
    Returns ONLY dividend events with columns:
      ["ticker","date","Dividends","Avg_Close_Prev_1_Month","source"]
    """
    raw = request_data(ticker)
    div = filter_dividend(raw)
    stock = request_data(ticker)
    feats = add_stock_prices(div, stock)

    feats = feats.dropna(subset=["Avg_Close_Prev_1_Month"]).copy()
    feats["ticker"] = ticker
    feats["source"] = "yfinance"
    feats = feats.rename(columns={"Date": "date"})
    # ensure tz-naive
    feats["date"] = pd.to_datetime(feats["date"]).dt.tz_localize(None)
    # sort by real date
    feats = feats.sort_values("date").reset_index(drop=True)
    return feats[["ticker", "date", "Dividends", "Avg_Close_Prev_1_Month", "source"]]


# ---------- ARTIFACTS ----------

def load_artifacts(artifacts_dir: Path):
    model = None
    scaler = None

    mp = artifacts_dir / "model.h5"
    if load_model and mp.exists():
        try:
            model = load_model(mp)
        except Exception as e:
            warnings.warn(f"Failed to load model.h5: {e}")

    sp = artifacts_dir / "scaler.joblib"
    if joblib and sp.exists():
        try:
            scaler = joblib.load(sp)
        except Exception as e:
            warnings.warn(f"Failed to load scaler.joblib: {e}")

    return model, scaler


def assert_scaler_feature_names(scaler):
    """Ensure scaler was fitted on the exact feature names/order."""
    if scaler is None:
        return
    if hasattr(scaler, "feature_names_in_"):
        expected = ["Dividends", "Avg_Close_Prev_1_Month"]
        names = list(scaler.feature_names_in_)
        if names != expected:
            raise ValueError(
                f"Scaler feature order mismatch. Got {names}, expected {expected}. "
                "Refit the scaler with the correct column order."
            )


# ---------- PREDICTION LOOP (EVENT-BASED, scaled-space like your notebook) ----------

def predict_next_n_events(model, scaler, event_rows: np.ndarray, n_steps: int, lookback: int) -> list[float]:
    """
    event_rows: ndarray shape (N, 2) in RAW scale with columns:
        [Dividends, Avg_Close_Prev_1_Month]
    - Take last `lookback` rows -> scale -> feed to model.
    - Keep input in **scaled space**.
    - After predicting yhat_scaled, inverse-transform the dividend using a row
      [yhat_scaled, last_avg_close_scaled].
    - Append the new scaled step [yhat_scaled, last_avg_close_scaled] to the window.
    """
    assert event_rows.ndim == 2 and event_rows.shape[1] == 2
    if len(event_rows) < lookback:
        return []

    # prepare initial window in scaled space
    last_seq_raw = event_rows[-lookback:]  # (lookback, 2)
    if scaler is not None:
        last_seq_scaled = scaler.transform(
            pd.DataFrame(last_seq_raw, columns=["Dividends", "Avg_Close_Prev_1_Month"])
        )
    else:
        last_seq_scaled = last_seq_raw

    x_scaled = last_seq_scaled.reshape(1, lookback, 2)
    preds = []

    for _ in range(n_steps):
        yhat_scaled = model.predict(x_scaled, verbose=0).reshape(-1)[0]

        # use the most recent avg_close from the ORIGINAL raw series (persistence)
        last_avg_close = float(event_rows[-1, 1])

        # we need the SCALED value for the avg_close feature to build the row for inverse_transform
        if scaler is not None:
            dummy = scaler.transform(
                pd.DataFrame([[0.0, last_avg_close]], columns=["Dividends", "Avg_Close_Prev_1_Month"])
            )
            last_avg_close_scaled = float(dummy[0, 1])

            row_scaled = np.array([[yhat_scaled, last_avg_close_scaled]])
            row_inv = scaler.inverse_transform(row_scaled)  # ndarray
            yhat = float(row_inv[0, 0])  # <- fixed: no `.values`
        else:
            yhat = float(yhat_scaled)

        # clip negatives (dividends can't be < 0)
        yhat = max(0.0, yhat)
        preds.append(yhat)

        # advance the scaled window: append [yhat_scaled, last_avg_close_scaled]
        if scaler is not None:
            append_step = np.array([[yhat_scaled, last_avg_close_scaled]])
        else:
            append_step = np.array([[yhat, last_avg_close]])

        x_scaled = np.concatenate([x_scaled[:, 1:, :], append_step.reshape(1, 1, 2)], axis=1)

    return preds



# ---------- PAY-DATE ESTIMATOR (EVENT-BASED) ----------

def estimate_next_event_dates(event_df: pd.DataFrame, n_steps: int = 5) -> list[pd.Timestamp]:
    """
    Use gaps between actual dividend **event dates** to estimate cadence (in months).
    Project from the **last event month** by the modal month gap. Return month-end dates.
    """
    df = event_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")

    if df.empty:
        return []

    # months (periods) for non-zero events (all rows are events already)
    periods = df["date"].dt.to_period("M")
    if len(periods) < 2:
        base = pd.Period(df["date"].max(), freq="M")
        return [(base + i + 1).to_timestamp("M") for i in range(n_steps)]

    month_ids = periods.dt.year.values * 12 + periods.dt.month.values
    diffs = np.diff(month_ids)
    if diffs.size == 0:
        period = 3
    else:
        vals, counts = np.unique(diffs, return_counts=True)
        period = int(vals[np.argmax(counts)]) if vals.size else 3
        if period <= 0:
            period = 3

    last_p = periods.iloc[-1]
    return [(last_p + period * (i + 1)).to_timestamp("M") for i in range(n_steps)]


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser(description="ETF Dividend Prediction Pipeline (event-based, notebook-aligned)")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS, help="List of ETF tickers.")
    parser.add_argument("--artifacts_dir", default="model_artifacts",
                        help="Directory with model.h5, scaler.joblib.")
    parser.add_argument("--steps", type=int, default=5, help="Number of future dividend events to predict.")
    parser.add_argument("--lookback", type=int, default=12, help="Number of past dividend events per sequence.")
    parser.add_argument("--outdir", default="data", help="Output directory for CSVs.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir)
    model, scaler = load_artifacts(artifacts_dir)
    if model is None:
        warnings.warn("No model loaded. The script will export history but skip predictions.")
    # ensure scaler feature order matches
    assert_scaler_feature_names(scaler)

    all_hist = []
    all_preds = []

    for tk in args.tickers:
        print(f"[{tk}] Fetching event features...")
        ev = fetch_features_events(tk)
        if ev.empty:
            print(f"[{tk}] No dividend events found.")
            continue

        # Save history (event rows only, as trained)
        all_hist.append(ev[["ticker", "date", "Dividends", "Avg_Close_Prev_1_Month", "source"]])

        if model is not None:
            series = ev[["Dividends", "Avg_Close_Prev_1_Month"]].values.astype(float)
            if len(series) < args.lookback:
                warnings.warn(f"[{tk}] Not enough events for lookback={args.lookback}. Skipping predictions.")
                continue

            preds = predict_next_n_events(model, scaler, series, n_steps=args.steps, lookback=args.lookback)
            pred_dates = estimate_next_event_dates(ev[["date", "Dividends"]], n_steps=args.steps)

            # guard: align lengths
            if len(preds) != len(pred_dates):
                m = min(len(preds), len(pred_dates))
                preds = preds[:m]
                pred_dates = pred_dates[:m]

            pred_df = pd.DataFrame({
                "ticker": tk,
                "pred_index": list(range(1, len(preds) + 1)),
                "pred_date_est": pd.to_datetime(pred_dates),
                "pred_dividend": preds,
            })
            all_preds.append(pred_df)

    # Write outputs
    if all_hist:
        hist_out = pd.concat(all_hist, ignore_index=True)
        hist_out = hist_out.sort_values(["ticker", "date"])

    if all_preds:
        preds_out = pd.concat(all_preds, ignore_index=True)
        preds_out = preds_out.sort_values(["ticker", "pred_index"])

        # --- Make unified table for Tableau ---
        # Add Type flag to history
        hist_labeled = hist_out.copy()
        hist_labeled["Type"] = "History"
        hist_labeled = hist_labeled.rename(columns={
            "Dividends": "Dividends",
            "Avg_Close_Prev_1_Month": "Avg_Close_Prev_1_Month"
        })

        # Rename predictions to align with history schema
        preds_labeled = preds_out.rename(columns={
            "pred_date_est": "date",
            "pred_dividend": "Dividends"
        }).copy()
        preds_labeled["Avg_Close_Prev_1_Month"] = None   # no forecast for this
        preds_labeled["source"] = "model"
        preds_labeled["Type"] = "Prediction"

        # Stack them
        combined = pd.concat([hist_labeled, preds_labeled], ignore_index=True)
        combined = combined[["ticker", "date", "Dividends", "Avg_Close_Prev_1_Month", "Type", "source"]]

        # Export single CSV
        combined.to_csv(outdir / "etf_dividends_all.csv", index=False)
        print(f"Wrote {outdir / 'etf_dividends_all.csv'}")

    else:
        print("No predictions generated (no model loaded or insufficient history).")


if __name__ == "__main__":
    main()
