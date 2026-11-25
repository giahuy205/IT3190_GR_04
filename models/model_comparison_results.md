# KẾT QUẢ THỰC NGHIỆM SO SÁNH CÁC MÔ HÌNH MACHINE LEARNING

| model | MAE | RMSE | R2 | train_time_sec |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: TB lịch sử (trạm, giờ, cuối tuần)** | 0.9368 | 1.8394 | 0.4256 | 0.0s |
| **Linear Regression** | 0.9439 | 1.6652 | 0.5292 | 1.21s |
| **Decision Tree (max_depth=12)** | 0.8768 | 1.5696 | 0.5817 | 8.45s |
| **Random Forest (100 cây)** | 0.8609 | 1.5154 | 0.6101 | 95.85s |
| **XGBoost Regressor (Tối ưu)** | 0.8726 | 1.4861 | 0.6251 | 19.54s |


*Ghi chú: Đánh giá độc lập trên tập kiểm thử (Test Set: 306,328 bản ghi) từ 2026-07-11 09:00:00.*