import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Citi Bike Jersey City - Dự báo Nhu cầu Xe đạp",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(__file__)
META_PATH = os.path.join(BASE_DIR, "data", "processed", "stations_meta.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Danh sách ngày lễ liên bang Hoa Kỳ
US_HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-07-04",
    "2026-09-07", "2026-10-12", "2026-11-11", "2026-11-26", "2026-12-25"
])


@st.cache_data
def load_station_meta():
    if os.path.exists(META_PATH):
        df = pd.read_csv(META_PATH)
        # Đảm bảo có cột lat, lon, name
        if "lng" in df.columns and "lon" not in df.columns:
            df["lon"] = df["lng"]
        if "station_name" in df.columns and "name" not in df.columns:
            df["name"] = df["station_name"]
        return df
    return pd.DataFrame([
        {"station_id": "JC005", "name": "Grove St PATH", "lat": 40.7195, "lon": -74.0431, "total_trips": 50000},
        {"station_id": "JC012", "name": "Newport Pkwy", "lat": 40.7287, "lon": -74.0321, "total_trips": 42000},
        {"station_id": "JC001", "name": "Exchange Place", "lat": 40.7163, "lon": -74.0334, "total_trips": 38000},
    ])


@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def calculate_weather_factor(temp, rain_choice, wind_speed):
    """Hệ số ảnh hưởng thời tiết nhập tay (Phục vụ demo Giữa kỳ)"""
    factor = 1.0

    # Ảnh hưởng nhiệt độ (Nhiệt độ tối ưu: 20-28°C)
    if temp < 5:
        factor *= 0.65
    elif temp < 15:
        factor *= 0.85
    elif temp > 35:
        factor *= 0.80

    # Ảnh hưởng lượng mưa
    if rain_choice == "Mưa nhỏ (dưới 5mm)":
        factor *= 0.60
    elif rain_choice == "Mưa vừa / Mưa to (trên 5mm)":
        factor *= 0.25

    # Ảnh hưởng tốc độ gió
    if wind_speed > 30:
        factor *= 0.75

    return factor


def predict_demand(model_pkg, station_id, lat, lon, date_val, hour_val, temp, rain, wind):
    if model_pkg is None:
        base = 25 if 7 <= hour_val <= 9 or 17 <= hour_val <= 19 else 10
        return int(round(base * calculate_weather_factor(temp, rain, wind)))

    model = model_pkg["model"]
    st_mean_map = model_pkg.get("station_mean_map", {})
    gmean = model_pkg.get("global_mean", 1.0)
    station_mean = st_mean_map.get(station_id, gmean)

    dayofweek = date_val.weekday()
    day = date_val.day
    month = date_val.month
    is_weekend = 1 if dayofweek >= 5 else 0
    is_holiday = 1 if pd.Timestamp(date_val).normalize() in US_HOLIDAYS else 0
    season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[month]
    is_rush = 1 if (hour_val in [7, 8, 9, 17, 18, 19] and is_weekend == 0) else 0

    hour_sin = np.sin(2 * np.pi * hour_val / 24)
    hour_cos = np.cos(2 * np.pi * hour_val / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Ước lượng giá trị trễ theo quy luật của trạm
    lag_base = station_mean * (1.8 if is_rush else 0.8)
    lag_1h = max(0.0, lag_base * 0.95)
    lag_2h = max(0.0, lag_base * 0.85)
    lag_24h = max(0.0, lag_base * 1.0)
    rolling_3h = (lag_1h + lag_2h + lag_base) / 3.0

    features = pd.DataFrame([{
        "hour": hour_val,
        "dayofweek": dayofweek,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "season": season,
        "is_rush": is_rush,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "lat": lat,
        "lon": lon,
        "station_mean": station_mean,
        "demand_lag_1h": lag_1h,
        "demand_lag_2h": lag_2h,
        "demand_lag_24h": lag_24h,
        "rolling_mean_3h": rolling_3h,
    }])

    pred = model.predict(features)[0]
    pred = max(0, pred)

    # Kết hợp hệ số điều kiện thời tiết nhập tay
    w_factor = calculate_weather_factor(temp, rain, wind)
    final_pred = int(round(pred * w_factor))
    return final_pred


def main():
    st.title("🚲 Dự đoán Nhu cầu Thuê Xe đạp Công cộng (Jersey City)")
    st.markdown(
        "**Đề tài môn Khai phá Dữ liệu & Machine Learning (Báo cáo Giữa kỳ)** | Mô hình dự báo nhu cầu thuê xe theo trạm và thời gian"
    )

    stations_df = load_station_meta()
    model_pkg = load_trained_model()

    # SIDEBAR: Điều khiển & Nhập liệu
    st.sidebar.header("⚙️ Bảng Điều Khiển Dự Báo")

    # 1. Chọn Trạm
    station_options = stations_df["name"].tolist()
    selected_name = st.sidebar.selectbox("📍 Chọn Trạm Xe Đạp:", station_options, index=0)
    station_row = stations_df[stations_df["name"] == selected_name].iloc[0]
    selected_id = station_row["station_id"]
    lat = station_row["lat"]
    lon = station_row["lon"]

    # 2. Chọn Thời gian
    selected_date = st.sidebar.date_input("📅 Chọn Ngày Dự Báo:")
    selected_hour = st.sidebar.slider("⏰ Khung Giờ (Hour of Day):", 0, 23, 8)

    # 3. Form Nhập tay Thời tiết (Giữa kỳ)
    st.sidebar.subheader("🌦️ Điều Kiện Thời Tiết (Nhập tay)")
    temp = st.sidebar.slider("🌡️ Nhiệt độ (°C):", -5, 40, 24)
    rain = st.sidebar.selectbox("🌧️ Tình trạng Mưa:", ["Không mưa", "Mưa nhỏ (dưới 5mm)", "Mưa vừa / Mưa to (trên 5mm)"])
    wind = st.sidebar.slider("💨 Tốc độ gió (km/h):", 0, 50, 10)

    # Thực hiện dự đoán
    predicted_val = predict_demand(model_pkg, selected_id, lat, lon, selected_date, selected_hour, temp, rain, wind)

    # Hiển thị 4 thẻ KPI kết quả
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="🎯 Lượt xe dự kiến thuê", value=f"{predicted_val} xe/giờ")
    with col2:
        load_status = "Bình thường"
        if predicted_val >= 25:
            load_status = "🚨 Rất cao (Giờ cao điểm)"
        elif predicted_val >= 12:
            load_status = "⚠️ Trung bình cao"
        st.metric(label="📊 Mức độ tải", value=load_status)
    with col3:
        rebalance = "Đủ nguồn cung xe"
        if predicted_val >= 20:
            rebalance = "🚚 Cần tiếp thêm xe"
        st.metric(label="🔄 Khuyến nghị điều phối", value=rebalance)
    with col4:
        st.metric(label="🤖 Mô hình áp dụng", value=model_pkg["model_name"] if model_pkg else "Baseline")

    st.divider()

    # TABS CHÍNH
    tab_map, tab_chart = st.tabs(["🗺️ Bản đồ Tương tác Jersey City", "📈 Biểu đồ Chuỗi 24 Giờ của Trạm"])

    with tab_map:
        st.subheader(f"📍 Bản đồ Mạng lưới Trạm Xe đạp tại Jersey City ({len(stations_df)} trạm)")
        st.write(f"Trạm đang chọn: **{selected_name}** (`Mã trạm: {selected_id}`) | Tọa độ: `({lat:.4f}, {lon:.4f})`")

        map_data = stations_df.copy()
        map_data["is_selected"] = (map_data["station_id"] == selected_id).astype(int)

        # Trạm được chọn có màu đỏ nổi bật và bán kính lớn
        map_data["color_r"] = np.where(map_data["is_selected"] == 1, 235, 30)
        map_data["color_g"] = np.where(map_data["is_selected"] == 1, 45, 144)
        map_data["color_b"] = np.where(map_data["is_selected"] == 1, 45, 255)
        map_data["radius"] = np.where(map_data["is_selected"] == 1, 75, 35)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["lon", "lat"],
            get_color=["color_r", "color_g", "color_b", 210],
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=13.5,
            pitch=25,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{name}</b><br/>Mã trạm: {station_id}<br/>Tổng chuyến: {total_trips}", "style": {"color": "white"}},
                map_style="mapbox://styles/mapbox/light-v10",
            )
        )

    with tab_chart:
        st.subheader(f"📊 Dự báo Nhu cầu 24 Giờ trong ngày của trạm: {selected_name}")
        hours = list(range(24))
        demands_24h = [
            predict_demand(model_pkg, selected_id, lat, lon, selected_date, h, temp, rain, wind)
            for h in hours
        ]

        chart_df = pd.DataFrame({
            "Khung giờ (Hour)": [f"{h:02d}:00" for h in hours],
            "Nhu cầu dự kiến (Xe/giờ)": demands_24h,
        })

        st.line_chart(chart_df.set_index("Khung giờ (Hour)"))

        st.caption(
            "💡 Biểu đồ mô phỏng chu kỳ nhu cầu trong 24 giờ. Khi điều chỉnh thanh trượt Nhiệt độ hoặc Tình trạng mưa ở bảng điều khiển bên trái, toàn bộ đồ thị sẽ co giãn phản ánh độ nhạy của nhu cầu theo thời tiết."
        )


if __name__ == "__main__":
    main()
