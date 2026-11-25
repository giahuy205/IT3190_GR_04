import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "station_demand_processed.csv")
STATION_META_PATH = os.path.join(BASE_DIR, "data", "processed", "stations_meta.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    print("[*] Đang đọc dữ liệu để vẽ biểu đồ báo cáo...")
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["hour_timestamp"])
    meta = pd.read_csv(STATION_META_PATH)

    # 1. Biểu đồ Nhu cầu theo Giờ (Ngày thường vs Cuối tuần)
    print("    -> Vẽ: eda_hourly_demand.png")
    plt.figure(figsize=(10, 5))
    hourly = df.groupby(["hour", "is_weekend"])["demand"].mean().reset_index()
    hourly["is_weekend"] = hourly["is_weekend"].map({0: "Ngày trong tuần (Thứ 2 - 6)", 1: "Cuối tuần (Thứ 7 - CN)"})
    sns.lineplot(data=hourly, x="hour", y="demand", hue="is_weekend", marker="o", linewidth=2.5, palette=["#1f77b4", "#ff7f0e"])
    plt.title("Nhu cầu thuê xe trung bình theo 24 khung giờ tại Jersey City", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Khung giờ trong ngày (Hour)", fontsize=11)
    plt.ylabel("Số lượt thuê trung bình (xe/giờ/trạm)", fontsize=11)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_hourly_demand.png"), dpi=200)
    plt.close()

    # 2. Biểu đồ Top 10 Trạm có lưu lượng thuê cao nhất
    print("    -> Vẽ: eda_top_stations.png")
    plt.figure(figsize=(10, 6))
    top10 = meta.sort_values(by="total_trips", ascending=False).head(10)
    sns.barplot(data=top10, x="total_trips", y="station_name", palette="viridis")
    plt.title("Top 10 Trạm Citi Bike có lượt thuê cao nhất Jersey City (13 tháng)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Tổng số chuyến xe khởi hành (Trips)", fontsize=11)
    plt.ylabel("Tên trạm", fontsize=11)
    for index, value in enumerate(top10["total_trips"]):
        plt.text(value + 500, index, f"{value:,}", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_top_stations.png"), dpi=200)
    plt.close()

    # 3. Biểu đồ Xu hướng nhu cầu theo tháng (Mùa vụ)
    print("    -> Vẽ: eda_monthly_trend.png")
    plt.figure(figsize=(11, 5))
    df["month_year"] = df["hour_timestamp"].dt.to_period("M").astype(str)
    monthly = df.groupby("month_year")["demand"].sum().reset_index()
    sns.barplot(data=monthly, x="month_year", y="demand", color="#2ca02c")
    plt.title("Tổng lượt thuê xe đạp theo từng tháng (10/2025 - 10/2026)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Tháng", fontsize=11)
    plt.ylabel("Tổng lượt thuê (Trips)", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "eda_monthly_trend.png"), dpi=200)
    plt.close()

    # 4. Biểu đồ So sánh 3 Mô hình ML (MAE & RMSE)
    print("    -> Vẽ: model_comparison_bar.png")
    res_path = os.path.join(MODELS_DIR, "model_comparison_results.csv")
    if os.path.exists(res_path):
        res_df = pd.read_csv(res_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Subplot 1: MAE và RMSE
        df_melt = res_df.melt(id_vars=["Mô hình"], value_vars=["MAE (Test)", "RMSE (Test)"], var_name="Metric", value_name="Value")
        sns.barplot(data=df_melt, x="Mô hình", y="Value", hue="Metric", ax=axes[0], palette=["#3498db", "#e74c3c"])
        axes[0].set_title("So sánh sai số MAE & RMSE trên Test Set (Càng nhỏ càng tốt)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Lượt xe", fontsize=10)
        axes[0].tick_params(axis='x', rotation=15)

        # Subplot 2: R2 Score
        sns.barplot(data=res_df, x="Mô hình", y="R² Score (Test)", ax=axes[1], palette="crest")
        axes[1].set_title("So sánh Hệ số R² Score trên Test Set (Càng gần 1 càng tốt)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("R² Score", fontsize=10)
        axes[1].tick_params(axis='x', rotation=15)
        for idx, row in res_df.iterrows():
            axes[1].text(idx, row["R² Score (Test)"] + 0.01, f"{row['R² Score (Test)']:.4f}", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "model_comparison_bar.png"), dpi=200)
        plt.close()

    print(f"[+] Toàn bộ 4 biểu đồ đã được lưu tại: {FIG_DIR}")


if __name__ == "__main__":
    main()
