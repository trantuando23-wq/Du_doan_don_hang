# Báo Cáo Kỹ Thuật Đánh Giá Mô Hình Dự Đoán Thời Gian Hoàn Thành Món Ăn (Food ETA Đà Nẵng)

> **Thông tin kiểm định:**
> - **Ngày thực hiện:** 17/11/2026 01:52:55
> - **Phiên bản scikit-learn:** `1.9.1`
> - **Phân loại dữ liệu:** **DỮ LIỆU MÔ PHỎNG** (10.000 đơn hàng trải dài trong 6 tuần giả lập)

---

## 1. Dữ Liệu & Phân Chia Huấn Luyện (Data Splitting)

- **Tổng số mẫu:** 10,000 dòng
- **Tập Train:** 8,287 mẫu (2026-09-01 đến 2026-10-05)
  - *Tập Train chính (Proper Train - 4 tuần đầu):* ~6.710 mẫu
  - *Tập Hiệu chuẩn (Calibration Set - tuần thứ 5):* ~1.651 mẫu (dành riêng cho Conformal Prediction)
- **Tập Test:** 1,639 mẫu (2026-10-06 đến 2026-10-12 - tuần thứ 6 độc lập)
- **Phương pháp phân chia:** **Phân chia theo thứ tự thời gian (Temporal Split)** để phản ánh đúng thực tế vận hành và tránh rò rỉ dữ liệu (Data Leakage).

---

## 2. Bảng So Sánh Hiệu Năng Các Mô Hình

| Mô hình | MAE (Sai số TB) | RMSE (Căn sai số) | $R^2$ (Hệ số tương quan) | % Dự đoán lệch < 5 phút |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TB theo món)** | 3.9 phút | 5.46 phút | 0.4476 | 72.97% |
| **Linear Regression** | 2.16 phút | 3.82 phút | 0.7307 | 92.13% |
| **Random Forest Regressor** | **2.05 phút** | **3.69 phút** | **0.7476** | **93.17%** |

### Nhận xét trung thực về mô hình:
- **Biến đổi Log-Transform:** Mô hình áp dụng hàm $\ln(\text{minutes})$ khi huấn luyện và $\exp$ khi dự đoán. Điều này giúp giảm MAE từ 2.18p xuống 2.15p, đồng thời loại bỏ hoàn toàn khả năng dự đoán giá trị âm hoặc lệch phi vật lý do phân phối thời gian làm bếp có tính chất lệch phải (lognormal).
- **So sánh Linear vs Random Forest:** 
  - Mô hình Linear Regression đạt kết quả rất tốt (2.16 phút MAE, $R^2 = 0.7307$).
  - Tuy nhiên, **Random Forest Regressor vượt trội hơn** (2.05 phút MAE, $R^2 = 0.7476$) bởi vì thuật toán cây quyết định học được các **tương tác phi tuyến tính** như hiện tượng nghẽn chảo chiên / vỉ nướng khi lượng khách tăng đột biến vào giờ cao điểm cuối tuần.

---

## 3. Phân Tích Lỗi Chi Tiết (Error Analysis)

Dữ liệu chi tiết được xuất tại tệp [`reports/error_by_group.csv`](error_by_group.csv).

### Tóm tắt các phân nhóm:
1. **Nhóm mô hình dự đoán chính xác nhất:**
   - Nhóm món **`do_uong`** đạt sai số thấp nhất với **MAE = 1.41 phút** (thời gian chuẩn bị đồ uống ổn định, ít phụ thuộc nhiệt độ chảo hay đơn hàng phức tạp).
2. **Nhóm mô hình gặp nhiều sai số nhất:**
   - Nhóm món **`chien_nuong`** có độ lệch cao nhất với **MAE = 3.8 phút**.
   - Đặc biệt, trong điều kiện **món chiên/nướng vào giờ cao điểm**, sai số MAE tăng lên đến **4.14 phút** do bếp bị quá tải chảo rán cục bộ, tạo ra dao động ngẫu nhiên lớn.
3. **Theo khung giờ:**
   - Khung giờ **`Cao điểm trưa (11-13h)`** có sai số MAE cao nhất (**2.31 phút**), trong khi khung giờ thấp điểm dự đoán sát hơn đáng kể.

---

## 4. Kiểm Tra & Hiệu Chuẩn Độ Phủ Khoảng Dự Đoán (Conformal Prediction)

Khoảng dự đoán ban đầu dựa trên phân vị 10% - 90% từ các cây quyết định trong Random Forest nhằm hướng tới độ bao phủ mục tiêu **80%**.

- **Độ phủ thực tế ban đầu (trên tập Test):** **66.02%** *(Thấp hơn mục tiêu 80% do các cây trong rừng có tương quan dương, làm khoảng phân vị bị thu hẹp)*.
- **Phương pháp hiệu chuẩn:** **Split Conformal Prediction**
  - Sử dụng tuần thứ 5 (Calibration Set) độc lập hoàn toàn với Test Set để tính điểm không tương đồng (non-conformity score).
  - Độ lệch hiệu chỉnh tính được: **$q_{calib} = +0.64 \text{ phút}$**.
- **Độ phủ sau hiệu chuẩn (trên tập Test):** **80.23%** *(Đạt chuẩn lý tưởng ~80%, sai lệch dưới 1%)*.
- **Kết luận:** Khoảng dự đoán hiển thị cho người dùng trên API đã được hiệu chuẩn đảm bảo 80% đơn hàng thực tế sẽ nằm trọn trong khoảng này.

---

## 5. Kiểm Tra Tính Hợp Lý Của Dự Đoán (Monotonicity Sanity Check)

Thực hiện thử nghiệm trên **500 tổ hợp ngẫu nhiên** để kiểm tra các quy luật vật lý của nhà hàng:

1. **Quy luật Hàng đợi (Queue Monotonicity):**
   - Khi số đơn chờ tăng thêm 5 đơn, thời gian dự đoán phải tăng hoặc giữ nguyên.
   - **Tỷ lệ vi phạm:** **3.6%** (18/500 mẫu).
   - Sai số giảm tối đa: 2.06 phút.
2. **Quy luật Số lượng (Quantity Monotonicity):**
   - Khi số lượng phần ăn tăng thêm 2 phần, thời gian dự đoán phải tăng.
   - **Tỷ lệ vi phạm:** **0.6%** (3/500 mẫu).

---

## 6. Biểu Đồ Minh Họa Kèm Theo

Các biểu đồ sau đã được tạo tự động tại thư mục `reports/`:
- [`actual_vs_pred.png`](actual_vs_pred.png): Biểu đồ phân tán Thực tế vs Dự đoán.
- [`feature_importance.png`](feature_importance.png): Mức độ quan trọng của các đặc trưng.
- [`error_distribution.png`](error_distribution.png): Phân phối mật độ sai số của mô hình.
- [`mae_by_hour.png`](mae_by_hour.png): Biến thiên sai số theo từng giờ trong ngày.
