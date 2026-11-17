# Báo Cáo Đánh Giá Mô Hình Trên Dữ Liệu Thực Tế (Real-World Report)

> **Thời điểm xuất:** 17/11/2026 01:54:03  
> **Nguồn dữ liệu:** `data\real_orders_template.csv` (5 đơn hàng thực tế)

---

## 1. Bảng So Sánh Với Dữ Liệu Mô Phỏng

| Chỉ số | Dữ liệu Mô phỏng (Test Set) | Dữ liệu Thực tế | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Số lượng mẫu** | 1,639 đơn | 5 đơn | ⚠️ Cần thêm dữ liệu (chưa đủ 30 đơn) |
| **MAE (phút)** | 2.05 | **1.89** | Sai số thực tế |
| **RMSE (phút)** | 3.69 | **2.38** | Mức độ nhạy cảm outlier |
| **Tỷ lệ lệch < 5 phút** | 93.17% | **100.0%** | Độ tin cậy thường nhật |
| **Độ phủ khoảng (Target 80%)** | 80.23% | **80.00%** | Khả năng bao trùm thực tế |

---

## 2. Phân Tích Độ Lệch Hệ Thống (Systematic Bias)

- **Độ lệch trung bình (Thực tế - Dự đoán):** `+1.89 phút`
- **Nhận xét chuyên môn:** Mô hình đang đoán NHANH HƠN thực tế trung bình 1.89 phút (quán làm lâu hơn mô hình nghĩ).

---

## 3. Hướng Xử Lý Tiếp Theo
- Tiếp tục thu thập thêm đơn thực tế qua tính năng **"Báo thời gian thực tế"** trên website.
- Khi đạt trên 50 đơn hàng thực tế, chạy `python src/evaluate_real.py --retrain` để cập nhật trọng số cho mô hình.
