# IT3190 - Dự đoán Nhu cầu Thuê Xe đạp Công cộng tại các Trạm

> **Học phần:** 20261 - Nhập môn Học máy và Khai phá dữ liệu  
> **Nhóm thực hiện:** Nhóm 4  
> **Hệ thống dữ liệu:** Citi Bike - Phân hệ Jersey City (10/2025 – 10/2026)  

---

## 📌 Giới thiệu đề tài
Hệ thống xe đạp công cộng thường xuyên gặp tình trạng mất cân bằng trạm: trạm hết xe vào giờ cao điểm đi làm hoặc trạm đầy chỗ trả xe tại khu văn phòng/trung tâm thương mại. Đề tài xây dựng mô hình Học máy (Machine Learning) giải bài toán Hồi quy có giám sát để dự đoán số lượt xe bắt đầu chuyến đi tại từng trạm trong từng khung giờ (`demand`), từ đó hỗ trợ đơn vị vận hành điều phối xe tải chuyên chở kịp thời.

---

## 📊 Dữ liệu & Kết quả thực nghiệm

- **Quy mô dữ liệu:** 13 tháng gần nhất (10/2025 - 10/2026) gồm **1,083,538 chuyến đi thô**; làm sạch còn **1,082,913 chuyến hợp lệ** trên mạng lưới **114 trạm** Jersey City.
- **Tập tổng hợp:** **1,023,828 bản ghi trạm-giờ** (chia 70% Train - 30% Test theo dòng thời gian).
- **Hệ thống đặc trưng (18 features):** Chu kỳ giờ, ngày, tháng, cờ ngày lễ liên bang Mỹ (`is_holiday`), giờ cao điểm (`is_rush`), target encoding trạm (`station_mean`), và các biến trễ chuỗi thời gian (`demand_lag_1h`, `demand_lag_24h`, `rolling_mean_3h`).

### Bảng đối sánh 5 mô hình trên Test Set (306,328 bản ghi):

| Mô hình | MAE (lượt/giờ) | RMSE (lượt/giờ) | $R^2$ Score | Thời gian huấn luyện (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: TB lịch sử (trạm, giờ, cuối tuần)** | 0.9368 | 1.8394 | 0.4256 | 0.00s |
| **Linear Regression** | 0.9439 | 1.6652 | 0.5292 | 1.21s |
| **Decision Tree (max_depth = 12)** | 0.8768 | 1.5696 | 0.5817 | 8.45s |
| **Random Forest (100 cây)** | **0.8609** | 1.5154 | 0.6101 | 95.85s |
| **XGBoost Regressor (Tối ưu)** | 0.8726 | **1.4861** | **0.6251** | **19.54s** |

> [!TIP]
> **XGBoost Regressor** được chọn làm mô hình triển khai chính thức nhờ đạt RMSE thấp nhất (1.4861), $R^2$ cao nhất (0.6251) và tốc độ huấn luyện nhanh gấp 5 lần so với Random Forest.

---

## 📁 Cấu trúc thư mục

```
IT3190_GR_04/
├── data/
│   ├── raw/                 # Lưu trữ các file zip/csv tải từ Citi Bike S3
│   └── processed/
│       ├── stations_meta.csv# Danh mục 114 trạm và tọa độ phục vụ bản đồ
│       └── data_stats.json  # Thống kê tổng quan dữ liệu
├── figures/
│   ├── fig1_eda.png         # Biểu đồ phân tích khám phá (EDA)
│   └── fig2_prediction.png  # Đồ thị thực tế vs dự báo tại trạm Grove St PATH
├── models/
│   ├── best_model.pkl       # Mô hình XGBoost tốt nhất đã đóng gói
│   └── model_comparison_results.md
├── src/
│   ├── download_data.py     # Script tự động tải dữ liệu Citi Bike từ S3
│   ├── preprocess.py        # Pipeline tiền xử lý & tạo đặc trưng trễ
│   └── train.py             # Huấn luyện 5 mô hình và đánh giá
├── app.py                   # Ứng dụng Web Demo Streamlit + Bản đồ Pydeck
├── results.json             # Bảng số liệu chi tiết thực nghiệm
├── DE_XUAT_DE_TAI.md        # Báo cáo đề xuất đề tài chi tiết
├── BAO_CAO_TIEN_DO.md       # Báo cáo tiến độ bài tập lớn giữa kỳ
└── requirements.txt
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng Web Demo (Streamlit)
```bash
python -m streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt:
- Hiển thị **bản đồ tương tác 114 trạm xe đạp** bằng Pydeck.
- Cho phép chọn trạm, ngày giờ và **nhập tay điều kiện thời tiết** (nhiệt độ, mưa, gió).
- Trả về số lượt xe dự đoán tức thì và **biểu đồ dự báo liên tục trong 24 giờ tiếp theo**.

### 3. Tái lập quy trình huấn luyện (Nếu cần)
```bash
# Bước 1: Tải dữ liệu 13 tháng từ S3
python src/download_data.py --mode all

# Bước 2: Tiền xử lý & Tạo đặc trưng
python src/preprocess.py

# Bước 3: Huấn luyện 5 mô hình & Xuất kết quả
python src/train.py
```
