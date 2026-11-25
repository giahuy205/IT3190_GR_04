import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "station_demand_processed.csv")
STATION_META_PATH = os.path.join(BASE_DIR, "data", "processed", "stations_meta.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_JSON_PATH = os.path.join(BASE_DIR, "results.json")

RNG = 42


def evaluate(name, y_true, y_pred, train_time=0.0):
    p = np.clip(y_pred, 0, None)
    return {
        "model": name,
        "MAE": round(float(mean_absolute_error(y_true, p)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, p))), 4),
        "R2": round(float(r2_score(y_true, p)), 4),
        "train_time_sec": round(float(train_time), 2),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"[*] Đang đọc dataset từ: {PROCESSED_DATA_PATH}...")
    data = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["hour_ts"])
    stations = pd.read_csv(STATION_META_PATH)

    # Đọc stats nếu đã có từ tiền xử lý
    stats_path = os.path.join(BASE_DIR, "data", "processed", "data_stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    # 1. Chia Train / Test theo chuỗi thời gian (70% train, 30% test)
    data = data.sort_values(by="hour_ts").reset_index(drop=True)
    split_idx = int(len(data) * 0.70)
    split_date = data["hour_ts"].iloc[split_idx]
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()

    # Target encoding trạm: trung bình nhu cầu của trạm tính trên tập train (tránh rò rỉ dữ liệu)
    st_mean = train.groupby("station_id")["demand"].mean().rename("station_mean")
    gmean = train["demand"].mean()
    for d in (train, test):
        d["station_mean"] = d["station_id"].map(st_mean).fillna(gmean)

    stats["train_samples"] = len(train)
    stats["test_samples"] = len(test)
    stats["split_date"] = str(split_date)
    print(f"[+] Chia tập dữ liệu: Train={len(train):,} dòng | Test={len(test):,} dòng (Điểm cắt: {split_date})")

    # 2. Định nghĩa danh sách đặc trưng
    FEATS = [
        "hour", "dayofweek", "month", "is_weekend", "is_holiday", "season", "is_rush",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "lat", "lon", "station_mean",
        "demand_lag_1h", "demand_lag_2h", "demand_lag_24h", "rolling_mean_3h"
    ]

    X_tr, y_tr = train[FEATS], train["demand"]
    X_te, y_te = test[FEATS], test["demand"]

    # Thiết kế ma trận đặc trưng riêng cho Linear Regression (One-Hot để bắt tính phi tuyến)
    def lr_design(d):
        oh = pd.get_dummies(d[["hour", "dayofweek", "season"]].astype(str), drop_first=True).astype(float)
        base = d[["is_weekend", "is_holiday", "is_rush", "lat", "lon", "station_mean", "demand_lag_1h", "demand_lag_24h"]].reset_index(drop=True)
        return pd.concat([base, oh.reset_index(drop=True)], axis=1)

    Xl_tr = lr_design(train)
    Xl_te = lr_design(test).reindex(columns=Xl_tr.columns, fill_value=0)

    # 3. Huấn luyện và đánh giá các mô hình
    results = []

    # Model 1: Baseline ngây thơ (Trung bình lịch sử theo trạm, giờ, cuối tuần)
    print("\n[*] Đang đánh giá: Baseline (Trung bình lịch sử trạm x giờ x cuối tuần)...")
    hist = train.groupby(["station_id", "hour", "is_weekend"])["demand"].mean().rename("hist")
    p_hist = test.join(hist, on=["station_id", "hour", "is_weekend"])["hist"].fillna(gmean)
    results.append(evaluate("Baseline: TB lịch sử (trạm, giờ, cuối tuần)", y_te, p_hist, 0.0))

    # Model 2: Linear Regression
    print("[*] Đang huấn luyện: Linear Regression...")
    t0 = time.time()
    lr = LinearRegression().fit(Xl_tr, y_tr)
    t_lr = time.time() - t0
    results.append(evaluate("Linear Regression", y_te, lr.predict(Xl_te), t_lr))

    # Model 3: Decision Tree Regressor
    print("[*] Đang huấn luyện: Decision Tree Regressor (max_depth=12)...")
    t0 = time.time()
    dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=50, random_state=RNG).fit(X_tr, y_tr)
    t_dt = time.time() - t0
    results.append(evaluate("Decision Tree (max_depth=12)", y_te, dt.predict(X_te), t_dt))

    # Model 4: Random Forest Regressor
    print("[*] Đang huấn luyện: Random Forest Regressor (100 cây)...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=20, max_samples=0.3, n_jobs=-1, random_state=RNG).fit(X_tr, y_tr)
    t_rf = time.time() - t0
    p_rf = rf.predict(X_te)
    results.append(evaluate("Random Forest (100 cây)", y_te, p_rf, t_rf))

    # Model 5: XGBoost Regressor
    print("[*] Đang huấn luyện: XGBoost Regressor...")
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1).fit(X_tr, y_tr)
    t_xgb = time.time() - t0
    p_xgb = xgb.predict(X_te)
    results.append(evaluate("XGBoost Regressor (Tối ưu)", y_te, p_xgb, t_xgb))

    stats["results"] = results
    stats["rf_importance"] = dict(sorted(zip(FEATS, rf.feature_importances_.round(4)), key=lambda x: -x[1]))
    stats["xgb_importance"] = dict(sorted(zip(FEATS, xgb.feature_importances_.round(4)), key=lambda x: -x[1]))

    # Phân tích sai số theo từng trạm
    test["pred_best"] = np.clip(p_xgb, 0, None)
    test["pred_rf"] = np.clip(p_rf, 0, None)
    per_st = test.groupby("station_id").apply(
        lambda g: pd.Series({
            "mean_demand": g["demand"].mean(),
            "MAE_XGB": mean_absolute_error(g["demand"], g["pred_best"]),
            "MAE_RF": mean_absolute_error(g["demand"], g["pred_rf"]),
        }),
        include_groups=False,
    )
    per_st = per_st.join(stations.set_index("station_id")["name"]).sort_values("mean_demand", ascending=False)
    stats["top_stations"] = per_st.head(10).round(3).reset_index().to_dict("records")

    # In bảng kết quả
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("BẢNG KẾT QUẢ ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST:")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print("=" * 70)

    # 4. Vẽ các hình EDA và Hình Dự đoán (fig1_eda.png & fig2_prediction.png)
    print("\n[*] Đang sinh các hình vẽ fig1_eda.png và fig2_prediction.png...")
    plt.rcParams.update({"font.size": 9})

    # fig1_eda.png (3 biểu đồ con: a, b, c)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    hp = data.groupby(["is_weekend", "hour"])["demand"].mean().unstack(0)
    axes[0].plot(hp.index, hp[0], label="Ngày thường", color="#1f77b4", lw=2)
    axes[0].plot(hp.index, hp[1], label="Cuối tuần", color="#ff7f0e", lw=2)
    axes[0].set_xlabel("Giờ trong ngày")
    axes[0].set_ylabel("Lượt thuê TB / trạm / giờ")
    axes[0].legend()
    axes[0].set_title("(a) Theo giờ")

    daily = data.groupby(data["hour_ts"].dt.date)["demand"].sum()
    axes[1].plot(pd.to_datetime(daily.index), daily.values, lw=0.8, color="#2ca02c")
    axes[1].set_title("(b) Tổng lượt thuê theo ngày")
    axes[1].tick_params(axis="x", rotation=30)

    top = (
        data.groupby("station_id")["demand"]
        .mean()
        .nlargest(15)
        .sort_values()
        .rename("mean_demand")
        .to_frame()
        .join(stations.set_index("station_id")["name"])
    )
    axes[2].barh(top["name"].astype(str).str.slice(0, 24), top["mean_demand"], color="#9467bd")
    axes[2].set_title("(c) 15 trạm đông nhất (TB/giờ)")
    axes[2].tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_eda.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # fig2_prediction.png (Dự đoán theo giờ tại trạm đông nhất trong 1 tuần)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    st0 = per_st.index[0]
    st0_name = per_st.loc[st0, "name"]
    # Chọn 1 tuần mẫu trong test set
    week_start = test["hour_ts"].min() + pd.Timedelta(days=7)
    week_end = week_start + pd.Timedelta(days=7)
    w = test[(test["station_id"] == st0) & (test["hour_ts"] >= week_start) & (test["hour_ts"] < week_end)]

    ax.plot(w["hour_ts"], w["demand"], label="Thực tế", color="#333333", lw=1.2)
    ax.plot(w["hour_ts"], w["pred_best"], label="XGBoost Dự đoán", color="#d62728", lw=1.2, linestyle="--")
    ax.plot(w["hour_ts"], w["pred_rf"], label="Random Forest Dự đoán", color="#1f77b4", lw=1.0, alpha=0.7)
    ax.set_title(f"Dự đoán nhu cầu theo giờ tại trạm {st0_name} ({week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')})")
    ax.set_ylabel("Số lượt thuê (xe/giờ)")
    ax.legend()
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_prediction.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # 5. Lưu kết quả số và đóng gói mô hình tốt nhất
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"[+] Đã lưu kết quả thực nghiệm chi tiết ra: {RESULTS_JSON_PATH}")

    # Đóng gói XGBoost làm best_model
    model_save_path = os.path.join(MODELS_DIR, "best_model.pkl")
    payload = {
        "model": xgb,
        "model_name": "XGBoost Regressor (Tối ưu)",
        "feature_cols": FEATS,
        "station_mean_map": st_mean.to_dict(),
        "global_mean": gmean,
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[+] Đã đóng gói mô hình tốt nhất tại: {model_save_path}")

    # Xuất Markdown bảng so sánh thủ công
    header = "| " + " | ".join(res_df.columns) + " |\n"
    sep = "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(res_df.columns))]) + " |\n"
    rows = ""
    for _, r in res_df.iterrows():
        rows += f"| **{r['model']}** | {r['MAE']} | {r['RMSE']} | {r['R2']} | {r['train_time_sec']}s |\n"
    md_table = header + sep + rows
    md_out = os.path.join(MODELS_DIR, "model_comparison_results.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# KẾT QUẢ THỰC NGHIỆM SO SÁNH CÁC MÔ HÌNH MACHINE LEARNING\n\n")
        f.write(md_table)
        f.write(f"\n\n*Ghi chú: Đánh giá độc lập trên tập kiểm thử (Test Set: {len(test):,} bản ghi) từ {split_date}.*")


if __name__ == "__main__":
    main()
