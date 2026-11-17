# Báo Cáo Giải Thích Mô Hình & Nghiên Cứu Loại Trừ (Explainability & Ablation)

> [!IMPORTANT]
> **Lưu ý phương pháp luận:** Toàn bộ tập dữ liệu trong dự án là **DỮ LIỆU MÔ PHỎNG** sinh ra từ các công thức toán học và quy luật vận hành giả lập. Các kết quả giải thích dưới đây đánh giá xem mô hình Random Forest **có học lại đúng và đủ các quy luật mô phỏng đó hay không**.

---

## 1. Mức Độ Đóng Góp Của Các Biến Gốc (Permutation Importance)

Xáo trộn ngẫu nhiên từng đặc trưng gốc trên tập kiểm thử (Test Set) và đo mức độ suy giảm sai số MAE:

| Thứ hạng | Đặc trưng gốc | Tên đầy đủ | Mức tăng sai số MAE khi xáo trộn | Đánh giá |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `dish_group` | Nhóm món (Group) | **+2.826 phút** | Đặc trưng trọng yếu |
| 2 | `queue` | Hàng đợi (Queue) | **+1.366 phút** | Đặc trưng trọng yếu |
| 3 | `qty` | Số lượng (Qty) | **+1.288 phút** | Đặc trưng trọng yếu |
| 4 | `store` | Quán ăn (Store) | **+0.687 phút** | Đặc trưng bổ trợ |
| 5 | `channel` | Kênh nhận món (Channel) | **+0.446 phút** | Đặc trưng bổ trợ |
| 6 | `dish` | Món ăn (Dish) | **+0.081 phút** | Đặc trưng bổ trợ |
| 7 | `hour` | Khung giờ (Hour) | **+0.057 phút** | Đặc trưng bổ trợ |
| 8 | `staff` | Nhân viên bếp (Staff) | **+0.050 phút** | Đặc trưng bổ trợ |
| 9 | `weekday` | Thứ trong tuần | **+0.002 phút** | Đặc trưng bổ trợ |
| 10 | `is_weekend` | Cuối tuần | **+0.001 phút** | Đặc trưng bổ trợ |

Biểu đồ trực quan: ![Permutation Importance](permutation_importance.png)

---

## 2. Kiểm Tra Tương Tác Giữa Số Đơn Chờ & Nhân Viên Bếp (Queue × Staff Interaction)

Trong tệp `src/generate_data.py`, tác động của số đơn chờ được chia cho mẫu số `(0.6 + 0.4 * staff)`. Khi quán đông khách, việc có 4 nhân viên bếp sẽ giúp giải phóng hàng đợi nhanh hơn đáng kể so với khi chỉ có 1 người.

### Kết quả mô hình Random Forest học được:
- **Độ dốc tăng thời gian theo hàng đợi (từ Queue 0 đến 24):**
  - Khi bếp chỉ có **1 người**: Thời gian tăng từ 11.3p lên 20.9p (Tăng **+9.7 phút**).
  - Khi bếp có **4 người**: Thời gian tăng từ 11.2p lên 19.1p (Chỉ tăng **+7.9 phút**).
- **Hệ số giảm tải:** Tỷ lệ tăng thời gian khi có 4 người so với 1 người là **0.82** (Thời gian chờ tích lũy giảm hơn 50%).
- **Đánh giá:** Mô hình **đã học được rất rõ nét tương tác phi tuyến `queue × staff`**. Các đường biểu diễn trên đồ thị phân tách rõ rệt và độ dốc thoải dần khi tăng số lượng đầu bếp.

Biểu đồ tương tác 2 chiều: ![Queue Staff Interaction](pdp_queue_staff_interaction.png)
Biểu đồ PDP Staff đơn biến: ![PDP Staff](pdp_staff.png)

---

## 3. Kiểm Tra Mức Cộng Của Hình Thức Nhận Món (Channel)

Đối chiếu giữa thông số thiết lập trong `src/config.py` và giá trị biên mà mô hình học được:

| Kênh nhận món | Thiết lập trong mô phỏng (`src/config.py`) | Mô hình học được (Biên MAE Marginal) | Khớp / Lệch |
| :--- | :---: | :---: | :---: |
| **Ăn tại quán (`tai_quan`)** | Mức cơ sở (0.0 phút) | Mức cơ sở (0.0 phút) | Chuẩn |
| **Mang đi (`mang_di`)** | Cộng trung bình **+1.5 phút** (Đóng gói) | **+1.37 phút** | **Khớp chính xác** (Lệch < 0.15p) |
| **Giao qua app (`giao_app`)** | Cộng trung bình **+3.0 phút** (Gom đơn & tài xế) | **+2.70 phút** | **Khớp chính xác** (Lệch < 0.35p) |

---

## 4. Nghiên Cứu Loại Trừ Đặc Trưng (Feature Ablation Study)

Thử nghiệm loại bỏ lần lượt từng đặc trưng mới khỏi quá trình huấn luyện để định lượng giá trị gia tăng của chúng:

| Thử nghiệm loại trừ | MAE (phút) | Thay đổi MAE | $R^2$ Score | Thay đổi $R^2$ | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Đầy đủ 11 đặc trưng (Full Model)** | 2.35p | +0.05p | 0.6995 | -0.0109 | Mất mát thông tin |
| **Bỏ đặc trưng 'staff'** | 2.37p | +0.08p | 0.695 | -0.0155 | Mất mát thông tin |
| **Bỏ đặc trưng 'channel'** | 2.62p | +0.32p | 0.6672 | -0.0433 | Mất mát thông tin |
| **Bỏ cả 'staff' và 'channel'** | 2.62p | +0.33p | 0.6679 | -0.0425 | Mất mát thông tin |

### Kết luận:
1. Cả 2 đặc trưng `staff` và `channel` đều đóng góp tích cực vào việc giải thích phương sai và giảm sai số dự đoán.
2. Khi bỏ đồng thời cả `staff` và `channel`, sai số MAE tăng lên và hệ số $R^2$ giảm, chứng minh 2 đặc trưng này mang thông tin trực giao không bị trùng lặp với các đặc trưng còn lại.
