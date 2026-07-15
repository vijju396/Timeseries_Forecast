from services.data_service import preprocessed_path


def build_preprocessed_files(cleaned_df=None):
    import pandas as pd

    if cleaned_df is None:
        cleaned_df = pd.read_csv(preprocessed_path("cleaned_training_input.csv"))

    df = cleaned_df.copy()
    df["date"] = pd.to_datetime(df["Date"])
    df["target"] = pd.to_numeric(df["Click Count"], errors="coerce")
    df = df.dropna(subset=["date", "target"]).sort_values("date")
    df = df.groupby("date", as_index=False)["target"].sum()
    df = _add_exogenous(df[["date", "target"]])

    holdout_size = _holdout_size(len(df))
    train = df.iloc[:-holdout_size].copy() if holdout_size else df.copy()
    predictions = df.iloc[-holdout_size:].copy() if holdout_size else df.tail(0).copy()
    predictions["date_var"] = predictions["date"]

    ts_path = preprocessed_path("ts_preprocessed.csv")
    train_path = preprocessed_path("train.csv")
    predictions_path = preprocessed_path("predictions.csv")
    df.to_csv(ts_path, index=False)
    train.to_csv(train_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    return {
        "ts_preprocessed": str(ts_path),
        "train": str(train_path),
        "predictions": str(predictions_path),
        "rows": len(df),
        "holdout_rows": len(predictions),
    }


def add_exogenous_features(df):
    return _add_exogenous(df.copy())


def _add_exogenous(df):
    dates = df["date"]
    df["week_exog"] = dates.dt.isocalendar().week.astype(int)
    df["dayOfWeek_exog"] = dates.dt.dayofweek.astype(int)
    df["month_exog"] = dates.dt.month.astype(int)
    return df


def _holdout_size(length):
    if length < 10:
        return max(1, length // 3)
    return min(28, max(5, length // 5))


def recommend_imputation(role, semantic_type="unknown", max_gap=3):
    semantic = str(semantic_type).lower()
    if role == "target" and any(token in semantic for token in ("demand", "count", "event", "transaction")):
        return {"method": "zero", "confidence": "high", "max_gap": 0, "reason": "missing observations represent no activity for a count-like target."}
    if role == "target" and any(token in semantic for token in ("balance", "stock", "inventory", "state")):
        return {"method": "forward_fill", "confidence": "medium", "max_gap": max_gap, "reason": "state values persist until changed."}
    if role == "exogenous" and any(token in semantic for token in ("sensor", "temperature", "rate", "continuous")):
        return {"method": "time", "confidence": "medium", "max_gap": max_gap, "reason": "short gaps in continuous measurements can be time-interpolated."}
    return {"method": "leave_missing", "confidence": "low", "max_gap": 0, "reason": "semantics do not justify inventing values."}


def impute_training_partition(frame, columns, method, train_end, max_gap=3, seasonal_period=None):
    """Fit and apply an imputer without allowing validation rows to train it."""
    import pandas as pd

    result = frame.copy()
    train_end = max(0, min(int(train_end), len(result)))
    train = result.iloc[:train_end]
    audit = {"method": method, "columns": {}, "future_dependent": method == "backward_fill"}
    for column in columns:
        if column not in result:
            continue
        values = pd.to_numeric(train[column], errors="coerce")
        missing_before = int(result[column].isna().sum())
        if method == "mean":
            fill_value = values.mean()
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].fillna(fill_value)
        elif method == "median":
            fill_value = values.median()
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].fillna(fill_value)
        elif method == "zero":
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].fillna(0)
        elif method == "forward_fill":
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].ffill(limit=max_gap)
        elif method == "backward_fill":
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].bfill(limit=max_gap)
        elif method in {"linear", "time"}:
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].interpolate(limit=max_gap, limit_direction="both")
        elif method == "seasonal" and seasonal_period:
            result.loc[:train_end - 1, column] = result.loc[:train_end - 1, column].fillna(result.loc[:train_end - 1, column].shift(seasonal_period))
        elif method == "drop":
            result = result.dropna(subset=[column])
        elif method == "reject" and result[column].iloc[:train_end].isna().any():
            raise ValueError(f"Missing values in {column} require an explicit imputation policy.")
        filled = missing_before - int(result[column].isna().sum())
        audit["columns"][column] = {"values_imputed": max(0, filled), "training_statistic": float(fill_value) if method in {"mean", "median"} and pd.notna(fill_value) else None}
    return result, audit
