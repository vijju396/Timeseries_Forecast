"""Regression coverage for constant-series and collinear-exogenous VAR failures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import training_service


def _panel():
    periods = 210
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    index = np.arange(periods, dtype=float)
    first = 20 + np.sin(index / 5) * 2 + index * 0.03
    second = 15 + np.cos(index / 9) * 1.5 + index * 0.02
    constant = np.full(periods, 4.0)
    day_of_week = dates.dayofweek.astype(float)
    return pd.DataFrame({
        "date": dates,
        "target": first + second + constant,
        "endogenous_1": first,
        "endogenous_2": second,
        "endogenous_3": constant,
        "week_exog": dates.isocalendar().week.astype(float).to_numpy(),
        "dayOfWeek_exog": day_of_week,
        "month_exog": dates.month.astype(float),
        "exogenous_constant": 1.0,
        "exogenous_duplicate_day": day_of_week,
    })


def main():
    frame = _panel()
    train = frame.iloc[:180].copy()
    output = frame.iloc[180:].copy()

    endog, constant_total = training_service._prepare_var_endog(train)
    assert list(endog.columns) == ["endogenous_1", "endogenous_2"]
    assert constant_total == 4.0

    exog_train, exog_output = training_service._exog_pair(train, output)
    stable_train, stable_output = training_service._prepare_var_exog(exog_train, exog_output)
    assert "exogenous_constant" not in stable_train.columns
    assert stable_train.shape[1] == np.linalg.matrix_rank(stable_train.to_numpy())
    assert stable_output.shape == (len(output), stable_train.shape[1])

    plain = np.asarray(training_service._fit_var(train, output), dtype=float)
    with_exog = np.asarray(training_service._fit_var_exog(train, output), dtype=float)
    assert plain.shape == with_exog.shape == (len(output),)
    assert np.isfinite(plain).all() and np.isfinite(with_exog).all()
    assert (plain > constant_total).all() and (with_exog > constant_total).all()
    for fitter in (training_service._fit_var, training_service._fit_var_exog):
        evaluation = training_service._rolling_origin_evaluation(fitter, frame)
        assert evaluation["metrics"]["mape"] is not None
        assert len(evaluation["predictions"]) > 0
    print("VAR stability regression passed")


if __name__ == "__main__":
    main()
