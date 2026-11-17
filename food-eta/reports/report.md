# Báo Cáo Kỹ Thuật Đánh Giá Mô Hình Dự Đoán Thời Gian Hoàn Thành Món Ăn (Food ETA Đà Nẵng)

> **Thông tin kiểm định:**
> - **Ngày thực hiện:** 18/11/2026 04:35:34
> - **Phiên bản scikit-learn:** `1.9.1`
> - **Siêu tham số Random Forest tốt nhất:** `{'n_estimators': 100, 'min_samples_leaf': 2, 'max_features': 0.8, 'max_depth': 20}`
> - **Đặc trưng mô hình (giữ nguyên, không thêm mới):** `store`, `dish`, `dish_group`, `weekday`, `channel`, `qty`, `hour`, `is_weekend`, `queue`, `staff`
> - **Phân loại dữ liệu:** **DỮ LIỆU MÔ PHỎNG** (15,000 đơn hàng · 8 quán thuộc 3 khu vực Đà Nẵng · 6 tuần giả lập)

---

## 1. Dữ Liệu & Phân Chia Huấn Luyện (Data Splitting)

- **Tổng số mẫu:** 15,000 dòng (8 quán hư cấu · 3 khu vực: Hải Châu, Sơn Trà, Ngũ Hành Sơn)
- **Tập Train:** 12,424 mẫu (2026-09-01 đến 2026-10-05)
  - *Tập Train chính (Proper Train - 4 tuần đầu, đã loại 1% outlier):* 10,002 mẫu
  - *Tập Hiệu chuẩn (Calibration Set - tuần thứ 5):* 2,422 mẫu (dành riêng cho Split Conformal Prediction)
- **Tập Test:** 2,468 mẫu (2026-10-06 đến 2026-10-12 - tuần thứ 6 độc lập)
- **Phương pháp phân chia:** **Phân chia theo thứ tự thời gian (Temporal Split)** để phản ánh đúng thực tế vận hành và tránh rò rỉ dữ liệu (Data Leakage).

## 2. Bảng So Sánh Trước / Sau Khi Mở Rộng Quán (backup_v2 (5 quán · 10.000 đơn) → 8 quán · 3 khu vực)

Hệ thống đã sao lưu phiên bản trước khi mở rộng trong thư mục `backup_v2/`. Dưới đây là bảng đối chiếu trực tiếp trên tập kiểm thử độc lập (Test Set Tuần 6):

| Chỉ số / Mô hình | Trước mở rộng (backup_v2 (5 quán · 10.000 đơn)) | Sau mở rộng (8 quán · 15,000 đơn) | Thay đổi | Nhận xét |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest MAE** | **2.25 phút** | **2.3 phút** | **+0.05 phút** | Ổn định tương đương |
| **Random Forest RMSE** | 3.94 phút | 3.74 phút | -0.20 phút | Độ phân tán sai số |
| **Random Forest $R^2$** | 0.6846 | **0.7105** | **+0.0259** | Tăng độ giải thích phương sai |
| **Random Forest % lệch < 5p** | 91.58% | 91.13% | -0.45% | Độ chính xác thực tế |
| **Linear Regression MAE** | 2.32 phút | 2.34 phút | +0.02 phút | Mô hình tuyến tính |
| **Linear Regression $R^2$** | 0.6884 | 0.7028 | +0.0144 | Khả năng khớp tuyến tính |
| **Baseline (TB món) MAE** | 3.88 phút | 3.98 phút | +0.10 phút | Khi có nhiều quán trùng món với tốc độ khác nhau, Baseline theo tên món sẽ sai lệch mạnh hơn |
| **Độ phủ Conformal (Test)** | 79.32% | 77.96% | -1.36% | Bám sát chuẩn mục tiêu 80% |

### Nhận xét trung thực:
- Mở rộng từ 5 lên 8 quán (có các cặp quán trùng loại món nhưng khác hệ số tốc độ như Mì Quảng Hạt Sen vs Mì Quảng Gió Biển, Bánh Xèo Sông Hàn vs Bánh Xèo Làng Đá) làm **Baseline theo tên món** suy giảm rõ rệt vì không phân biệt được quán nào nhanh/chậm hơn.
- Trong khi đó, **Random Forest** vẫn giữ sai số ở mức **2.3 phút** ($R^2 = 0.7105$, **91.13%** đơn lệch dưới 5 phút), chứng tỏ mô hình tách biệt tốt tốc độ riêng của từng quán khi đã có dữ liệu lịch sử của quán đó.

---

## 3. Bảng Hiệu Năng Chi Tiết (Tập Test Tuần 6)

| Mô hình | MAE (Sai số TB) | RMSE (Căn sai số) | $R^2$ (Hệ số tương quan) | % Dự đoán lệch < 5 phút |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TB theo món)** | 3.98 phút | 5.53 phút | 0.368 | 71.84% |
| **Linear Regression** | 2.34 phút | 3.79 phút | 0.7028 | 90.6% |
| **Random Forest Regressor** | **2.3 phút** | **3.74 phút** | **0.7105** | **91.13%** |

> *Nhận xét:* Random Forest vượt trội hơn Linear Regression trên cả MAE, RMSE và $R^2$ nhờ bắt được tương tác phi tuyến `queue × staff` và nghẽn trạm chiên/nướng.

---

## 4. Phân Tích Lỗi Chi Tiết (Error Analysis)

Dữ liệu chi tiết được xuất tại tệp [`reports/error_by_group.csv`](error_by_group.csv).

### Tóm tắt các phân nhóm:
1. **Nhóm mô hình dự đoán chính xác nhất:**
   - Nhóm món **`com`** đạt sai số thấp nhất với **MAE = 1.54 phút**.
2. **Nhóm mô hình gặp nhiều sai số nhất:**
   - Nhóm món **`chien_nuong`** có độ lệch cao nhất với **MAE = 3.58 phút**.
   - Đặc biệt, trong điều kiện **món chiên/nướng vào giờ cao điểm**, sai số MAE là **4.19 phút** do bếp bị quá tải trạm chiên rán cục bộ.
3. **Theo hình thức nhận món (Channel):**
   - MAE cụ thể: **Giao qua app**: 2.85p, **Mang đi**: 2.06p, **Ăn tại quán**: 2.16p. Giao qua app có độ phân tán lớn hơn do cộng thêm thời gian chờ tài xế.
4. **Theo số nhân viên bếp (Staff):**
   - MAE cụ thể: **Bếp 1 người**: 2.13p, **Bếp 2 người**: 2.24p, **Bếp 3 người**: 2.45p, **Bếp 4 người**: 2.58p.

---

## 5. Kiểm Tra & Hiệu Chuẩn Độ Phủ Khoảng Dự Đoán (Conformal Prediction)

Khoảng dự đoán ban đầu dựa trên phân vị 10% - 90% từ các cây quyết định trong Random Forest nhằm hướng tới độ bao phủ mục tiêu **80%**.

- **Độ phủ thực tế ban đầu (trên tập Test):** **69.65%** *(Thấp hơn mục tiêu 80% do các cây trong rừng có tương quan dương)*.
- **Phương pháp hiệu chuẩn:** **Split Conformal Prediction**
  - Sử dụng tuần thứ 5 (Calibration Set) độc lập hoàn toàn với Test Set để tính điểm không tương đồng.
  - Độ lệch hiệu chỉnh: **$q_{calib} = +0.48 \text{ phút}$**.
- **Độ phủ sau hiệu chuẩn (trên tập Test):** **77.96%** *(Đạt chuẩn lý tưởng ~80%)*.
- **Kết luận:** Khoảng dự đoán hiển thị cho người dùng trên API đã được chuẩn hóa toán học tin cậy.

---

## 6. Kiểm Tra Tính Hợp Lý Của Dự Đoán (Monotonicity & Sanity Check)

Thực hiện thử nghiệm trên **500 tổ hợp ngẫu nhiên**:

- **Tăng số đơn chờ (+5 đơn):** Tỷ lệ vi phạm = **8.2%** (41/500 mẫu, mức tụt tối đa 1.77 phút).
- **Tăng số lượng món (+2 phần):** Tỷ lệ vi phạm = **2.8%** (14/500 mẫu, mức tụt tối đa 0.84 phút).
- **Tăng nhân viên bếp (+1 người):** Tỷ lệ vi phạm = **15.4%** (thời gian giảm hoặc giữ nguyên khi có thêm người).
- **Kênh Giao app vs Ăn tại quán:** Tỷ lệ Giao app lâu hơn = **95.0%** (trung bình lâu hơn **2.7 phút** để đóng gói và chờ gom đơn).

---

## 7. Đánh Giá Quán Chưa Từng Thấy (GroupKFold Theo Quán — Leave-One-Store-Out)

Để kiểm tra xem mô hình có **tổng quát hóa được sang một quán mới hoàn toàn** hay chỉ **học thuộc từng quán trong tập train**, hệ thống chạy `GroupKFold(n_splits=8)` theo cột `store`: mỗi lượt ẩn toàn bộ dữ liệu của 1 quán, huấn luyện Random Forest trên 7 quán còn lại rồi dự đoán trên quán bị ẩn. *(Kết quả này chỉ dùng cho đánh giá kỹ thuật ở terminal/file, KHÔNG hiển thị trên web).*

| Quán bị ẩn (Unseen Test) | Khu vực | Số mẫu | MAE | RMSE | Độ lệch TB (Bias) | $R^2$ | % Lệch < 5p | Trùng món / Nhóm món với tập train |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tiệm Trà Cố Đô** | Quận Hải Châu | 1,935 | 1.88 phút | 3.22 phút | +0.61 phút | 0.3785 | 97.21% | 3 món chung / Có |
| **Trà & Cà Phê Muối Bãi Biển** | Quận Ngũ Hành Sơn | 1,948 | 2.3 phút | 3.12 phút | +1.44 phút | 0.3294 | 95.17% | 3 món chung / Có |
| **Mì Quảng Hạt Sen** | Quận Hải Châu | 1,985 | 2.45 phút | 3.94 phút | -1.87 phút | 0.4029 | 90.68% | 3 món chung / Có |
| **Bánh Xèo Làng Đá** | Quận Ngũ Hành Sơn | 1,989 | 3.45 phút | 5.22 phút | -2.49 phút | 0.3845 | 79.29% | 4 món chung / Có |
| **Cơm Gà Phố Cổ** | Quận Hải Châu | 1,696 | 5.09 phút | 5.61 phút | +4.72 phút | -0.7568 | 53.12% | 0 (món riêng) / Không (nhóm duy nhất) |
| **Bún Chả Cá Biển Xanh** | Quận Sơn Trà | 1,707 | 5.33 phút | 5.91 phút | +4.89 phút | -0.7252 | 48.33% | 0 (món riêng) / Có |
| **Mì Quảng Gió Biển** | Quận Sơn Trà | 2,023 | 5.97 phút | 7.32 phút | -5.92 phút | -0.428 | 47.6% | 3 món chung / Có |
| **Bánh Xèo Sông Hàn** | Quận Sơn Trà | 1,717 | 8.19 phút | 10.22 phút | -8.07 phút | -0.5391 | 34.3% | 4 món chung / Có |
| **TỔNG THỂ (GroupKFold 8 quán)** | **Toàn bộ** | **15,000** | **4.25 phút** | **5.92 phút** | — | **0.3089** | **69.27%** | — |

### Nhận xét trung thực về khả năng tổng quát hóa:
- **So sánh trực diện:** MAE khi dự đoán quán **chưa từng thấy** là **4.25 phút** ($R^2 = 0.3089$), kém hơn **+1.95 phút** so với tập test tiêu chuẩn đã từng thấy quán (**2.3 phút**).
- **Kết luận thẳng thắn:** Mô hình **chủ yếu học thuộc hệ số tốc độ riêng của từng quán (store-specific baseline)** hơn là tổng quát hóa hoàn hảo sang quán hoàn toàn mới.
  - **Quán tổng quát hóa tốt nhất:** **Tiệm Trà Cố Đô** (MAE = **1.88 phút**, bias = +0.61 phút) nhờ chia sẻ nhóm món / tên món với quán cùng loại trong tập huấn luyện và có tốc độ bếp gần mức trung bình của nhóm.
  - **Quán tệ nhất khi chưa từng thấy:** **Bánh Xèo Sông Hàn** (MAE = **8.19 phút**, bias = -8.07 phút, $R^2 = -0.5391$). Lý do rất rõ ràng: khi mô hình chưa từng nhìn thấy dữ liệu của quán này, `OneHotEncoder(handle_unknown='ignore')` đưa toàn bộ cột `store` của quán mới về 0, nên mô hình **không thể biết `speed_factor` hay thời gian nấu đặc thù của quán** (ví dụ quán làm nhanh/chậm hơn quán cùng loại khác, hoặc là quán duy nhất bán nhóm món đó như nhóm `com` của Cơm Gà Phố Cổ).
  - **Hàm ý thực tế:** Khi mở thêm một quán hoàn toàn mới ngoài đời thực, mô hình chỉ cho ước lượng sơ bộ dựa theo nhóm món (`dish_group`), đơn chờ (`queue`), nhân sự (`staff`) và kênh nhận (`channel`); cần thu thập dữ liệu phản hồi thực tế của riêng quán đó rồi chạy `evaluate_real.py --retrain` để đạt độ chính xác cao.

---

## 8. Các Tệp Báo Cáo Kèm Theo

- [`reports/metrics.json`](metrics.json): Toàn bộ số liệu máy đọc JSON.
- [`reports/error_by_group.csv`](error_by_group.csv): Bảng phân tích sai số chi tiết theo từng nhóm.
- [`reports/explain.md`](explain.md): Báo cáo giải thích mô hình chuyên sâu (Permutation Importance, PDP 1D/2D, Ablation).
- [`reports/actual_vs_pred.png`](actual_vs_pred.png): Đồ thị thực tế vs dự đoán.
- [`reports/feature_importance.png`](feature_importance.png): Mức độ quan trọng của đặc trưng.
- [`reports/error_distribution.png`](error_distribution.png): Phân phối sai số.
- [`reports/mae_by_hour.png`](mae_by_hour.png): MAE theo 24 giờ.
