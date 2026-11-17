# Báo Cáo Đánh Giá Mô Hình Trên Dữ Liệu Thực Tế (Real-World Report)

> **Thời điểm xuất:** 17/11/2026 02:49:45  
> **Nguồn dữ liệu:** `D:\Trần Tuấn Đô\DỰ ÁN PYTHON\duandonhangn\food-eta\data\feedback.csv` (1 đơn hàng thực tế)

---

## 1. Bảng So Sánh Với Dữ Liệu Mô Phỏng

| Chỉ số | Dữ liệu Mô phỏng (Test Set) | Dữ liệu Thực tế | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Số lượng mẫu** | 1,639 đơn | 1 đơn | ⚠️ Cần thêm dữ liệu (chưa đủ 30 đơn) |
| **MAE (phút)** | 2.25 | **1.85** | Sai số thực tế |
| **RMSE (phút)** | 3.94 | **1.85** | Mức độ nhạy cảm outlier |
| **Tỷ lệ lệch < 5 phút** | 91.58% | **100.0%** | Độ tin cậy thường nhật |
| **Độ phủ khoảng (Target 80%)** | 79.32% | **100.00%** | Khả năng bao trùm thực tế |

---

## 2. Phân Tích Độ Lệch Hệ Thống (Systematic Bias)

- **Độ lệch trung bình (Thực tế - Dự đoán):** `+1.85 phút`
- **Nhận xét chuyên môn:** Mô hình đang đoán NHANH HƠN thực tế trung bình 1.85 phút (quán làm lâu hơn mô hình nghĩ).

---

## 3. Hướng Xử Lý Tiếp Theo
- Tiếp tục thu thập thêm đơn thực tế qua tính năng **"Báo thời gian thực tế"** trên website.
- Khi đạt trên 50 đơn hàng thực tế, chạy `python src/evaluate_real.py --retrain` để cập nhật trọng số cho mô hình.
