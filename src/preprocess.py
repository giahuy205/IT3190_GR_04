import os
import glob
import json
import numpy as np
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Danh sách ngày lễ liên bang Hoa Kỳ (2017 đến 2026 để bao phủ mọi tập dữ liệu)
US_HOLIDAYS = pd.to_datetime([
    "2017-01-01", "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04",
    "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04", "2018-09-03",
    "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-07-04",
    "2026-09-07", "2026-10-12", "2026-11-11", "2026-11-26", "2026-12-25"
])


def normalize_columns(df):
    """Chuẩn hóa tên cột hỗ trợ cả schema cũ (2017-2020) và schema mới (2021-nay)."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_dict = {
        "trip_duration": "tripduration",
        "start_time": "starttime",
        "stop_time": "stoptime",
        "bike_id": "bikeid",
        "user_type": "usertype",
        "started_at": "starttime",
        "ended_at": "stoptime",
        "start_station_latitude": "lat",
        "start_lat": "lat",
        "start_station_longitude": "lon",
        "start_lng": "lon",
        "member_casual": "usertype",
    }
    df = df.rename(columns=rename_dict)
    return df


def load_raw_trips(raw_dir=RAW_DATA_DIR):
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "JC-*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {raw_dir}")

    print(f"[*] Tìm thấy {len(csv_files)} file CSV dữ liệu thô. Bắt đầu đọc...")
    dfs = []
    for f in csv_files:
        df_month = pd.read_csv(f, low_memory=False)
        df_month = normalize_columns(df_month)
        dfs.append(df_month)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[+] Tổng số chuyến đi đọc được: {len(df_all):,} bản ghi.")
    return df_all


def clean_trips(df):
    print("[*] Tiến hành làm sạch dữ liệu chuyến đi theo chuẩn của progress_experiment...")
    initial_len = len(df)
    stats = {}
    stats["raw_trips"] = initial_len

    # Đảm bảo định dạng datetime
    df["starttime"] = pd.to_datetime(df["starttime"], format="mixed")
    if "stoptime" in df.columns:
        df["stoptime"] = pd.to_datetime(df["stoptime"], format="mixed")
        df["tripduration"] = (df["stoptime"] - df["starttime"]).dt.total_seconds()
    elif "tripduration" not in df.columns:
        raise ValueError("Không tìm thấy thông tin thời lượng chuyến đi!")

    # Loại bỏ bản ghi khuyết thiếu trạm và tọa độ
    df = df.dropna(subset=["start_station_id", "start_station_name", "lat", "lon"])
    df = df.drop_duplicates()

    # Chuyến đi < 60 giây hoặc > 24 giờ (86400s)
    df = df[(df["tripduration"] >= 60) & (df["tripduration"] <= 86400)]

    # Lọc tọa độ trạm trong khu vực Jersey City / lân cận NYC
    df = df[df["lat"].between(40.60, 40.85) & df["lon"].between(-74.15, -73.90)]
    stats["clean_trips"] = len(df)
    stats["removed_trips"] = initial_len - len(df)

    # Loại bỏ các trạm bảo trì/thử nghiệm
    invalid_keywords = ["test", "repair", "depot", "warehouse", "charging"]
    pattern = "|".join(invalid_keywords)
    df = df[~df["start_station_name"].astype(str).str.lower().str.contains(pattern, na=False)]

    # Chỉ giữ các trạm có hoạt động đáng kể (>= 100 chuyến)
    cnt = df["start_station_id"].value_counts()
    keep_stations = cnt[cnt >= 100].index
    df = df[df["start_station_id"].isin(keep_stations)].copy()

    stats["final_trips"] = len(df)
    stats["n_stations"] = int(df["start_station_id"].nunique())
    if "usertype" in df.columns:
        stats["usertype_share"] = df["usertype"].value_counts(normalize=True).round(3).to_dict()
    stats["median_duration_min"] = round(df["tripduration"].median() / 60, 1)

    print(f"[+] Hoàn tất làm sạch: {stats['final_trips']:,} chuyến hợp lệ trên {stats['n_stations']} trạm.")
    return df, stats


def aggregate_and_build_features(df, stats, processed_dir=PROCESSED_DATA_DIR):
    os.makedirs(processed_dir, exist_ok=True)
    print("[*] Đang tổng hợp nhu cầu theo [Trạm × Giờ] và tạo đặc trưng...")

    # Trích xuất metadata của các trạm
    stations = (
        df.groupby("start_station_id")
        .agg(
            name=("start_station_name", "last"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            total_trips=("start_station_id", "count"),
        )
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )

    meta_path = os.path.join(processed_dir, "stations_meta.csv")
    stations.to_csv(meta_path, index=False, encoding="utf-8-sig")

    # Gom nhóm số lượt thuê theo trạm x giờ
    df["hour_ts"] = df["starttime"].dt.floor("h")
    agg = (
        df.groupby(["start_station_id", "hour_ts"])
        .size()
        .rename("demand")
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )

    # Tạo lưới đầy đủ trạm x giờ, CHỈ TÍNH TỪ KHI TRẠM BẮT ĐẦU HOẠT ĐỘNG (kỹ thuật từ progress_experiment)
    first_seen = agg.groupby("station_id")["hour_ts"].min()
    min_time = agg["hour_ts"].min()
    max_time = agg["hour_ts"].max()
    all_hours = pd.date_range(min_time, max_time, freq="h")

    grid = pd.MultiIndex.from_product(
        [stations["station_id"], all_hours], names=["station_id", "hour_ts"]
    ).to_frame(index=False)
    grid = grid.merge(first_seen.rename("first_seen"), on="station_id")
    grid = grid[grid["hour_ts"] >= grid["first_seen"].dt.floor("D")].drop(columns="first_seen")

    data = grid.merge(agg, on=["station_id", "hour_ts"], how="left").fillna({"demand": 0})
    data["demand"] = data["demand"].astype(int)

    stats["n_samples"] = len(data)
    stats["zero_share"] = round((data["demand"] == 0).mean(), 3)
    stats["demand_mean"] = round(data["demand"].mean(), 3)
    stats["demand_max"] = int(data["demand"].max())

    # Trích xuất đặc trưng thời gian và vị trí
    t = data["hour_ts"]
    data["hour"] = t.dt.hour
    data["dayofweek"] = t.dt.dayofweek
    data["day"] = t.dt.day
    data["month"] = t.dt.month
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["is_holiday"] = t.dt.normalize().isin(US_HOLIDAYS).astype(int)
    # 0: Mùa Đông, 1: Mùa Xuân, 2: Mùa Hè, 3: Mùa Thu
    data["season"] = data["month"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    # Giờ cao điểm (Rush hours: 7-9h sáng & 17-19h chiều các ngày làm việc)
    data["is_rush"] = (data["hour"].isin([7, 8, 9, 17, 18, 19]) & (data["is_weekend"] == 0)).astype(int)

    # Mã hoá chu kỳ tuần hoàn (Cyclic encoding)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    # Ghép tọa độ trạm
    data = data.merge(stations[["station_id", "lat", "lon", "name"]], on="station_id")

    # Sắp xếp và tạo đặc trưng độ trễ (Lag features)
    data = data.sort_values(by=["station_id", "hour_ts"]).reset_index(drop=True)
    data["demand_lag_1h"] = data.groupby("station_id")["demand"].shift(1)
    data["demand_lag_2h"] = data.groupby("station_id")["demand"].shift(2)
    data["demand_lag_24h"] = data.groupby("station_id")["demand"].shift(24)
    data["rolling_mean_3h"] = (
        data.groupby("station_id")["demand"].shift(1).rolling(window=3).mean()
    )

    # Loại bỏ các dòng khuyết thiếu do lag 24h
    data = data.dropna().reset_index(drop=True)

    out_csv = os.path.join(processed_dir, "station_demand_processed.csv")
    data.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_stats = os.path.join(processed_dir, "data_stats.json")
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"[+] Đã lưu dataset: {data.shape[0]:,} dòng, {data.shape[1]} cột tại: {out_csv}")
    print(f"[+] Thống kê sơ bộ: Zero Share={stats['zero_share']}, Mean Demand={stats['demand_mean']}, Max Demand={stats['demand_max']}")
    return data, stats


def main():
    df_raw = load_raw_trips()
    df_clean, stats = clean_trips(df_raw)
    aggregate_and_build_features(df_clean, stats)


if __name__ == "__main__":
    main()
