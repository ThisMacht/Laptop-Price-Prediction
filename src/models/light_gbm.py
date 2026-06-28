# ============================================================
# LightGBM Experiment Function Implementations
# ============================================================
# Designed to plug directly into notebook 07b_modeling_numeric_copy.ipynb
# - Targets: "target_price" hoặc "log_target_price"
# - Inverse transform: np.expm1 (log1p) hoặc np.exp (log)
# - CLIP_NEGATIVE_PREDICTIONS = True (consistent với notebook config)
# ============================================================

from __future__ import annotations

import warnings
from typing import Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


# ─────────────────────────────────────────────
# 1. Build LightGBM model
# ─────────────────────────────────────────────
def build_lightgbm_model(
    params: dict | None = None,
    random_state: int = 42,
    verbose: bool | int = False,
) -> lgb.LGBMRegressor:
    """
    Tạo LGBMRegressor với params mặc định hoặc params tự truyền vào.

    Default params được chọn consistent với các model khác trong notebook
    (n_estimators=500, learning_rate=0.05, max_depth=6) để so sánh công bằng.

    Parameters
    ----------
    params : dict or None
        Hyperparameters muốn override.
        Nếu None, dùng bộ params mặc định.
    random_state : int
        Seed để đảm bảo reproducibility.
    verbose : bool or int
        Điều khiển log training. False = tắt hoàn toàn (-1).

    Returns
    -------
    model : LGBMRegressor
        Model chưa train.
    """
    verbosity = -1 if verbose is False else (1 if verbose is True else int(verbose))

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,          # 2^(max_depth) - 1, consistent với depth=6
        "objective": "regression",
        "metric": "rmse",
        "random_state": random_state,
        "verbosity": verbosity,
        "n_jobs": -1,
    }

    if params is not None:
        merged = {**default_params, **params}
        if "random_state" not in params:
            merged["random_state"] = random_state
    else:
        merged = default_params

    return lgb.LGBMRegressor(**merged)


# ─────────────────────────────────────────────
# 2. Prepare train/test data from existing split
# ─────────────────────────────────────────────
def make_train_test_data(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    train_idx,
    test_idx,
    reset_index: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tạo X_train, X_test, y_train, y_test từ X/y và split index có sẵn.

    Dùng khi đã có train_idx / test_idx từ train_test_split một lần duy nhất,
    muốn áp lại đúng split đó cho nhiều phiên bản feature X khác nhau.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix đầy đủ (cả train lẫn test).
    y : pd.Series or np.ndarray
        Target tương ứng với X.
    train_idx : array-like
        Index (positional hoặc label) của tập train.
    test_idx : array-like
        Index (positional hoặc label) của tập test.
    reset_index : bool
        Nếu True, reset integer index sau khi tách.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    X_train = X.iloc[train_idx] if isinstance(train_idx[0], (int, np.integer)) else X.loc[train_idx]
    X_test  = X.iloc[test_idx]  if isinstance(test_idx[0],  (int, np.integer)) else X.loc[test_idx]
    y_train = y.iloc[train_idx] if isinstance(train_idx[0], (int, np.integer)) else y.loc[train_idx]
    y_test  = y.iloc[test_idx]  if isinstance(test_idx[0],  (int, np.integer)) else y.loc[test_idx]

    if reset_index:
        X_train = X_train.reset_index(drop=True)
        X_test  = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. Train one LightGBM model
# ─────────────────────────────────────────────
def train_lightgbm_model(
    model: lgb.LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    sample_weight: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
) -> lgb.LGBMRegressor:
    """
    Train một LGBMRegressor.

    LightGBM nhận eval_set và callbacks qua fit() — không dùng Pool như CatBoost.

    Parameters
    ----------
    model : LGBMRegressor
        Model đã khởi tạo (từ build_lightgbm_model).
    X_train : pd.DataFrame
        Feature train.
    y_train : pd.Series
        Target train (target_price hoặc log_target_price).
    X_valid : pd.DataFrame or None
        Feature validation dùng cho eval_set (early stopping).
    y_valid : pd.Series or None
        Target validation tương ứng.
    sample_weight : array-like or None
        Trọng số mẫu — ví dụ: tăng weight cho Premium segment.
    early_stopping_rounds : int or None
        Số vòng không cải thiện trước khi dừng.
        Cần có X_valid/y_valid để có hiệu lực.

    Returns
    -------
    model : LGBMRegressor
        Model sau khi đã fit.
    """
    fit_kwargs: dict = {}

    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]

        if early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),   # tắt log verbose trong callback
            ]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ─────────────────────────────────────────────
# 4. Inverse prediction back to price scale
# ─────────────────────────────────────────────
def inverse_prediction_to_price(
    y_pred: np.ndarray,
    target_name: str,
    log_transform: str = "log1p",
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Chuyển prediction về thang giá gốc.

    - Nếu target_name == "target_price": không làm gì, chỉ clip nếu cần.
    - Nếu target_name == "log_target_price": inverse theo log_transform.

    Parameters
    ----------
    y_pred : array-like
        Prediction thô từ model (log scale nếu dùng log_target_price).
    target_name : str
        "target_price" hoặc "log_target_price".
    log_transform : str
        "log1p"  → inverse = np.expm1  (khuyến nghị, consistent với notebook)
        "log"    → inverse = np.exp
    clip_negative : bool
        Clip giá trị âm về 0 sau inverse (mặc định True).

    Returns
    -------
    y_pred_price : np.ndarray
        Prediction trên thang giá thật (USD / VND / ...).
    """
    y_pred = np.asarray(y_pred, dtype=float)

    if target_name == "log_target_price":
        if log_transform == "log1p":
            y_pred_price = np.expm1(y_pred)
        elif log_transform == "log":
            y_pred_price = np.exp(y_pred)
        else:
            raise ValueError(f"log_transform phải là 'log1p' hoặc 'log', nhận: {log_transform!r}")
    elif target_name == "target_price":
        y_pred_price = y_pred.copy()
    else:
        raise ValueError(f"target_name không hợp lệ: {target_name!r}")

    if clip_negative:
        y_pred_price = np.maximum(y_pred_price, 0.0)

    return y_pred_price


# ─────────────────────────────────────────────
# 5. Compute regression metrics
# ─────────────────────────────────────────────
def compute_regression_metrics(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
) -> dict:
    """
    Tính các regression metrics trên thang giá thật.

    Tất cả metrics đều đo trên price scale (consistent với notebook —
    mọi comparison đều phải về cùng đơn vị).

    Parameters
    ----------
    y_true_price : array-like
        Giá thật (target_price).
    y_pred_price : array-like
        Giá dự đoán đã inverse nếu cần.

    Returns
    -------
    metrics : dict
        mae, rmse, r2, mape, median_ape (%)
    """
    y_true = np.asarray(y_true_price, dtype=float)
    y_pred = np.asarray(y_pred_price, dtype=float)

    # MAPE từ sklearn trả về dạng fraction (0.1 = 10%); nhân 100 cho dễ đọc
    mape_pct = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Median APE: tính thủ công để có median (sklearn không cung cấp)
    ape = np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true)) * 100
    median_ape_pct = float(np.median(ape))

    return {
        "mae":            float(mean_absolute_error(y_true, y_pred)),
        "rmse":           float(root_mean_squared_error(y_true, y_pred)),
        "r2":             float(r2_score(y_true, y_pred)),
        "mape_pct":       float(mape_pct),
        "median_ape_pct": median_ape_pct,
    }


# ─────────────────────────────────────────────
# 6. Compute segment-level metrics
# ─────────────────────────────────────────────
def compute_segment_metrics(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    segment_labels: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Tính metrics theo từng price segment.

    Giúp phát hiện model có bị yếu ở nhóm Premium / High hay không,
    vì aggregate RMSE đôi khi che khuất lỗi ở tail.

    Parameters
    ----------
    y_true_price : array-like
        Giá thật trên test set.
    y_pred_price : array-like
        Giá dự đoán trên test set (đã inverse).
    segment_labels : pd.Series or array-like
        Segment label: "Low", "Medium", "High", "Premium" (hoặc tương đương).

    Returns
    -------
    segment_metrics : pd.DataFrame
        Bảng metrics theo từng segment, sắp xếp theo segment.
    """
    y_true = np.asarray(y_true_price, dtype=float)
    y_pred = np.asarray(y_pred_price, dtype=float)
    labels = np.asarray(segment_labels)

    rows = []
    for seg in np.unique(labels):
        mask = labels == seg
        if mask.sum() == 0:
            continue
        m = compute_regression_metrics(y_true[mask], y_pred[mask])
        m["segment"] = seg
        m["n_samples"] = int(mask.sum())
        rows.append(m)

    segment_df = pd.DataFrame(rows)

    # Sắp xếp theo thứ tự giá tăng dần nếu có thể
    order = ["Low", "Medium", "High", "Premium"]
    present = [s for s in order if s in segment_df["segment"].values]
    others  = [s for s in segment_df["segment"].values if s not in order]
    sort_order = present + sorted(others)
    segment_df["_sort"] = segment_df["segment"].map(
        {s: i for i, s in enumerate(sort_order)}
    ).fillna(len(sort_order))
    segment_df = segment_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    cols_front = ["segment", "n_samples"]
    cols_rest  = [c for c in segment_df.columns if c not in cols_front]
    return segment_df[cols_front + cols_rest]


# ─────────────────────────────────────────────
# 7. Build prediction dataframe
# ─────────────────────────────────────────────
def build_prediction_frame(
    experiment_name: str,
    target_name: str,
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    segment_labels=None,
    X_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Tạo DataFrame chứa prediction + error columns để debug.

    Columns cố định:
        experiment_name, target_name,
        y_true_price, y_pred_price,
        error, abs_error,
        pct_error, abs_pct_error,
        segment (nếu có)

    Parameters
    ----------
    experiment_name : str
        Tên experiment (ví dụ: "LightGBM_log1p_v1").
    target_name : str
        "target_price" hoặc "log_target_price".
    y_true_price : array-like
        Giá thật.
    y_pred_price : array-like
        Giá dự đoán đã inverse.
    segment_labels : array-like or None
        Segment của mỗi sample trong test set.
    X_test : pd.DataFrame or None
        Nếu muốn nối thêm features để debug từng row.

    Returns
    -------
    pred_df : pd.DataFrame
    """
    y_true = np.asarray(y_true_price, dtype=float)
    y_pred = np.asarray(y_pred_price, dtype=float)

    error     = y_pred - y_true
    abs_error = np.abs(error)
    # Tránh div-by-zero khi giá thật == 0
    denom = np.where(y_true == 0, 1e-8, y_true)
    pct_error     = error / denom * 100
    abs_pct_error = np.abs(pct_error)

    pred_df = pd.DataFrame({
        "experiment_name": experiment_name,
        "target_name":     target_name,
        "y_true_price":    y_true,
        "y_pred_price":    y_pred,
        "error":           error,
        "abs_error":       abs_error,
        "pct_error":       pct_error,
        "abs_pct_error":   abs_pct_error,
    })

    if segment_labels is not None:
        pred_df["segment"] = np.asarray(segment_labels)

    if X_test is not None:
        X_reset = X_test.reset_index(drop=True)
        pred_df = pd.concat([pred_df, X_reset], axis=1)

    return pred_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 8. Plot true vs predicted price
# ─────────────────────────────────────────────
def plot_true_vs_predicted(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    title: str | None = None,
    segment_labels=None,
    save_path=None,
) -> None:
    """
    Scatter plot giá thật vs giá dự đoán.

    - Đường đỏ đứt: perfect prediction (y = x).
    - Màu theo segment nếu truyền vào (giúp thấy ngay nhóm nào bị lệch).
    - Có thể save figure ra file nếu truyền save_path.

    Parameters
    ----------
    y_true_price : array-like
        Giá thật.
    y_pred_price : array-like
        Giá dự đoán.
    title : str or None
        Tiêu đề biểu đồ.
    segment_labels : array-like or None
        Nếu có, tô màu điểm theo segment.
    save_path : str or Path or None
        Path để lưu figure (PNG/PDF). Nếu None thì chỉ show.

    Returns
    -------
    None
    """
    y_true = np.asarray(y_true_price, dtype=float)
    y_pred = np.asarray(y_pred_price, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))

    if segment_labels is not None:
        plot_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "segment": np.asarray(segment_labels),
        })
        sns.scatterplot(
            data=plot_df, x="y_true", y="y_pred",
            hue="segment", alpha=0.55, edgecolor=None, ax=ax,
        )
    else:
        ax.scatter(y_true, y_pred, alpha=0.55, edgecolors="none", s=30)

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.plot([vmin, vmax], [vmin, vmax], color="red", linestyle="--", linewidth=1.5,
            label="Perfect prediction")

    ax.set_xlabel("Actual Price", fontsize=12)
    ax.set_ylabel("Predicted Price", fontsize=12)
    ax.set_title(title or "Actual vs Predicted Price", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# Convenience: run a full experiment in one call
# ─────────────────────────────────────────────

def run_lightgbm_experiment(
    experiment_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_test_price: pd.Series,
    target_name: str = "log_target_price",
    log_transform: str = "log1p",
    model_params: dict | None = None,
    segment_labels_test=None,
    early_stopping_rounds: int | None = 50,
    sample_weight=None,
    plot: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Chạy một experiment LightGBM end-to-end.

    Workflow:
        build model
        train model
        predict raw target
        inverse prediction về price scale nếu train bằng log target
        compute overall metrics
        compute segment metrics nếu có
        build prediction dataframe
        plot actual vs predicted nếu cần

    Parameters
    ----------
    experiment_name : str
        Tên experiment, ví dụ:
        - "lightgbm_log_baseline"
        - "lightgbm_log_onehot_warranty"
        - "lightgbm_log_drop_low_importance"

    X_train : pd.DataFrame
        Feature train.

    y_train : pd.Series
        Target train.
        Nếu target_name = "log_target_price" thì đây là log_target_price train.

    X_test : pd.DataFrame
        Feature test.

    y_test : pd.Series
        Target test trên cùng scale với y_train.
        Nếu train bằng log_target_price thì y_test cũng phải là log_target_price.
        Biến này dùng cho eval_set / early stopping.

    y_test_price : pd.Series
        Giá thật trên price scale.
        Biến này dùng để tính metrics cuối cùng.

    target_name : str
        "target_price" hoặc "log_target_price".

    log_transform : str
        "log1p" hoặc "log".
        Dùng để inverse prediction nếu target_name = "log_target_price".

    model_params : dict or None
        Hyperparameters cho LightGBM.

    segment_labels_test : array-like or None
        Segment của test set, ví dụ Low / Medium / High / Premium.

    early_stopping_rounds : int or None
        Nếu None thì không dùng early stopping.
        Nếu khác None thì dùng X_test, y_test làm eval_set.

    sample_weight : array-like or None
        Sample weight cho train set.

    plot : bool
        Có vẽ scatter actual vs predicted hay không.

    verbose : bool or int
        Log training LightGBM.

    Returns
    -------
    result : dict
        Dictionary gồm:
        - model
        - metrics
        - segment_metrics
        - pred_df
        - raw_pred
        - y_pred_price
        - feature_names
    """

    # ------------------------------------------------------------
    # 1. Validate input shapes
    # ------------------------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train length mismatch: "
            f"{len(X_train)} vs {len(y_train)}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test length mismatch: "
            f"{len(X_test)} vs {len(y_test)}"
        )

    if len(X_test) != len(y_test_price):
        raise ValueError(
            f"X_test and y_test_price length mismatch: "
            f"{len(X_test)} vs {len(y_test_price)}"
        )

    if segment_labels_test is not None and len(X_test) != len(segment_labels_test):
        raise ValueError(
            f"X_test and segment_labels_test length mismatch: "
            f"{len(X_test)} vs {len(segment_labels_test)}"
        )

    # ------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------

    model = build_lightgbm_model(
        params=model_params,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 3. Train model
    # ------------------------------------------------------------

    model = train_lightgbm_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_test if early_stopping_rounds is not None else None,
        y_valid=y_test  if early_stopping_rounds is not None else None,
        sample_weight=sample_weight,
        early_stopping_rounds=early_stopping_rounds,
    )

    # ------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------

    raw_pred = model.predict(X_test)

    y_pred_price = inverse_prediction_to_price(
        y_pred=raw_pred,
        target_name=target_name,
        log_transform=log_transform,
        clip_negative=True,
    )

    # ------------------------------------------------------------
    # 5. Overall metrics on original price scale
    # ------------------------------------------------------------

    metrics = compute_regression_metrics(
        y_true_price=y_test_price,
        y_pred_price=y_pred_price,
    )

    metrics["experiment_name"] = experiment_name
    metrics["target_name"] = target_name
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    metrics["n_features"] = X_train.shape[1]

    try:
        metrics["best_iteration"] = model.best_iteration_
    except Exception:
        metrics["best_iteration"] = None

    # ------------------------------------------------------------
    # 6. Segment metrics
    # ------------------------------------------------------------

    segment_metrics = None

    if segment_labels_test is not None:
        segment_metrics = compute_segment_metrics(
            y_true_price=y_test_price,
            y_pred_price=y_pred_price,
            segment_labels=segment_labels_test,
        )

        segment_metrics.insert(0, "experiment_name", experiment_name)
        segment_metrics.insert(1, "target_name", target_name)

    # ------------------------------------------------------------
    # 7. Prediction dataframe
    # ------------------------------------------------------------

    pred_df = build_prediction_frame(
        experiment_name=experiment_name,
        target_name=target_name,
        y_true_price=y_test_price,
        y_pred_price=y_pred_price,
        segment_labels=segment_labels_test,
        X_test=None,
    )

    # ------------------------------------------------------------
    # 8. Plot
    # ------------------------------------------------------------

    if plot:
        plot_true_vs_predicted(
            y_true_price=y_test_price,
            y_pred_price=y_pred_price,
            title=f"{experiment_name} — Actual vs Predicted",
            segment_labels=segment_labels_test,
        )

    # ------------------------------------------------------------
    # 9. Return result
    # ------------------------------------------------------------

    return {
        "experiment_name": experiment_name,
        "target_name":     target_name,
        "model":           model,
        "metrics":         metrics,
        "segment_metrics": segment_metrics,
        "pred_df":         pred_df,
        "raw_pred":        raw_pred,
        "y_pred_price":    y_pred_price,
        "feature_names":   X_train.columns.tolist(),
    }