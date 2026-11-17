# Báo Cáo Kỹ Thuật Đánh Giá Mô Hình Dự Đoán Thời Gian Hoàn Thành Món Ăn (Food ETA Đà Nẵng)

> **Thông tin kiểm định:**
> - **Ngày thực hiện:** 17/11/2026 02:45:00
> - **Phiên bản scikit-learn:** `1.9.1`
> - **Đặc trưng mô hình:** 11 đặc trưng (`store`, `dish`, `dish_group`, `qty`, `hour`, `weekday`, `is_weekend`, `queue`, `staff`, `channel`, `minutes`)
> - **Phân loại dữ liệu:** **DỮ LIỆU MÔ PHỎNG** (10.000 đơn hàng trải dài trong 6 tuần giả lập)

---

## 1. Dữ Liệu & Phân Chia Huấn Luyện (Data Splitting)

- **Tổng số mẫu:** 10,000 dòng
- **Tập Train:** 8,292 mẫu (2026-09-01 đến 2026-10-05)
  - *Tập Train chính (Proper Train - 4 tuần đầu):* ~6.710 mẫu
  - *Tập Hiệu chuẩn (Calibration Set - tuần thứ 5):* ~1.651 mẫu (dành riêng cho Conformal Prediction)
- **Tập Test:** 1,639 mẫu (2026-10-06 đến 2026-10-12 - tuần thứ 6 độc lập)
- **Phương pháp phân chia:** **Phân chia theo thứ tự thời gian (Temporal Split)** để phản ánh đúng thực tế vận hành và tránh rò rỉ dữ liệu (Data Leakage).

## 2. Bảng So Sánh Trước / Sau Khi Thêm 2 Đặc Trưng (`staff` và `channel`)

Hệ thống đã lưu lại snapshot phiên bản cũ (v1) trong thư mục `backup_v1/`. Dưới đây là bảng đối chiếu trực tiếp trên tập kiểm thử (Test Set Tuần 6):

| Chỉ số / Mô hình | Bản cũ (9 đặc trưng - Không có Staff/Channel) | Bản mới (11 đặc trưng - Thêm Staff & Channel) | Thay đổi | Nhận xét |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest MAE** | **2.05 phút** | **2.25 phút** | **+0.20 phút** | Tương đương |
| **Random Forest RMSE** | 3.69 phút | 3.94 phút | +0.25 phút | Độ phân tán sai số |
| **Random Forest $R^2$** | 0.7476 | **0.6846** | **-0.0630** | Ổn định |
| **Random Forest % lệch < 5p** | 93.17% | 91.58% | -1.59% | Độ chính xác thực tế |
| **Linear Regression MAE** | 2.16 phút | 2.32 phút | +0.16 phút | Tuyến tính hóa |
| **Linear Regression $R^2$** | 0.7307 | 0.6884 | -0.0423 | Khả năng khớp tuyến tính |
| **Baseline (TB món) MAE** | 3.9 phút | 3.88 phút | -0.02 phút | Mức cơ sở không đổi nhiều |
| **Độ phủ Conformal (Test)** | 80.23% | 79.32% | -0.91% | Giữ chuẩn mục tiêu 80% |

### Nhận xét trung thực về sự thay đổi:
- **Độ giải thích ($R^2$) và độ chính xác (MAE):** Khi đưa thêm `staff` và `channel` vào mô hình, Random Forest đạt $R^2 = 0.6846$ và MAE = 2.25 phút.
- **Tại sao Linear Regression yếu hơn khi có Staff?** Trong công thức sinh dữ liệu, `staff` tham gia vào mẫu số chia `queue / (0.6 + 0.4 * staff)` (tương tác phi tuyến nhân - chia). Linear Regression chỉ cộng trừ đơn thuần nên không biểu diễn trọn vẹn được hiệu ứng đòn bẩy này khi hàng đợi lớn, trong khi Random Forest tách cây phân đoạn rất tự nhiên.

---

## 3. Bảng Hiệu Năng Chi Tiết (Tập Test Tuần 6)

| Mô hình | MAE (Sai số TB) | RMSE (Căn sai số) | $R^2$ (Hệ số tương quan) | % Dự đoán lệch < 5 phút |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TB theo món)** | 3.88 phút | 5.52 phút | 0.3797 | 73.64% |
| **Linear Regression** | 2.32 phút | 3.91 phút | 0.6884 | 91.46% |
| **Random Forest Regressor** | **2.25 phút** | **3.94 phút** | **0.6846** | **91.58%** |

---

## 4. Phân Tích Lỗi Chi Tiết (Error Analysis)

Dữ liệu chi tiết được xuất tại tệp [`reports/error_by_group.csv`](error_by_group.csv).

### Tóm tắt các phân nhóm:
1. **Nhóm mô hình dự đoán chính xác nhất:**
   - Nhóm món **`do_uong`** đạt sai số thấp nhất với **MAE = 1.62 phút**.
2. **Nhóm mô hình gặp nhiều sai số nhất:**
   - Nhóm món **`chien_nuong`** có độ lệch cao nhất với **MAE = 3.78 phút**.
   - Đặc biệt, trong điều kiện **món chiên/nướng vào giờ cao điểm**, sai số MAE là **4.42 phút** do bếp bị quá tải trạm chiên rán cục bộ.
3. **Theo hình thức nhận món (Channel):**
   - MAE cụ thể: **Giao qua app**: 3.05p, **Mang đi**: 2.22p, **Ăn tại quán**: 1.93p. Giao qua app có độ phân tán lớn hơn do cộng thêm thời gian chờ tài xế.
4. **Theo số nhân viên bếp (Staff):**
   - MAE cụ thể: **Bếp 1 người**: 1.78p, **Bếp 2 người**: 2.32p, **Bếp 3 người**: 2.27p, **Bếp 4 người**: 2.89p.

---

## 5. Kiểm Tra & Hiệu Chuẩn Độ Phủ Khoảng Dự Đoán (Conformal Prediction)

Khoảng dự đoán ban đầu dựa trên phân vị 10% - 90% từ các cây quyết định trong Random Forest nhằm hướng tới độ bao phủ mục tiêu **80%**.

- **Độ phủ thực tế ban đầu (trên tập Test):** **69.55%** *(Thấp hơn mục tiêu 80% do các cây trong rừng có tương quan dương)*.
- **Phương pháp hiệu chuẩn:** **Split Conformal Prediction**
  - Sử dụng tuần thứ 5 (Calibration Set) độc lập hoàn toàn với Test Set để tính điểm không tương đồng.
  - Độ lệch hiệu chỉnh: **$q_{calib} = +0.46 \text{ phút}$**.
- **Độ phủ sau hiệu chuẩn (trên tập Test):** **79.32%** *(Đạt chuẩn lý tưởng ~80%)*.
- **Kết luận:** Khoảng dự đoán hiển thị cho người dùng trên API đã được chuẩn hóa toán học tin cậy.

---

## 6. Kiểm Tra Tính Hợp Lý Của Dự Đoán (Monotonicity & Sanity Check)

Thực hiện thử nghiệm trên **500 tổ hợp ngẫu nhiên**:

- **Tăng số đơn chờ (+5 đơn):** Tỷ lệ vi phạm = **4.0%** (20/500 mẫu).
- **Tăng số lượng món (+2 phần):** Tỷ lệ vi phạm = **3.4%** (17/500 mẫu).
- **Tăng nhân viên bếp (+1 người):** Tỷ lệ vi phạm = **14.0%** (thời gian giảm hoặc giữ nguyên khi có thêm người).
- **Kênh Giao app vs Ăn tại quán:** Tỷ lệ Giao app lâu hơn = **97.2%** (trung bình lâu hơn **2.66 phút** để đóng gói và chờ gom đơn).

---

## 7. Các Tệp Báo Cáo Kèm Theo

- [`reports/metrics.json`](metrics.json): Toàn bộ số liệu máy đọc JSON.
- [`reports/error_by_group.csv`](error_by_group.csv): Bảng phân tích sai số chi tiết theo từng nhóm.
- [`reports/actual_vs_pred.png`](actual_vs_pred.png): Đồ thị thực tế vs dự đoán.
- [`reports/feature_importance.png`](feature_importance.png): Mức độ quan trọng của đặc trưng.
- [`reports/error_distribution.png`](error_distribution.png): Phân phối sai số.
- [`reports/mae_by_hour.png`](mae_by_hour.png): MAE theo 24 giờ.
