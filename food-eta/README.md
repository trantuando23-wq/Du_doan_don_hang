# Bếp Đà Nẵng · Food ETA (Hệ Thống Dự Đoán Thời Gian Hoàn Thành Món Ăn)

Hệ thống web Machine Learning dự đoán thời gian hoàn thành món ăn (**Food ETA**) cho **8 quán ăn hư cấu tại 3 khu vực Đà Nẵng** (*Hải Châu, Sơn Trà, Ngũ Hành Sơn*). Ứng dụng áp dụng Random Forest Regressor, hiệu chuẩn khoảng tin cậy **Split Conformal Prediction (80%)**, giải thích từng dự đoán bằng tiếng Việt thân thiện, gợi ý giờ đặt nhanh nhất, và hỗ trợ so sánh trực tiếp cùng một món ăn giữa các quán ở khu vực khác nhau.

> ⚠️ **LƯU Ý:** Dữ liệu khởi tạo là **DỮ LIỆU MÔ PHỎNG** (15.000 đơn hàng trong 6 tuần giả lập). Toàn bộ tên quán đều là quán hư cấu, không có địa chỉ thật. Chỉ số đánh giá mô hình chỉ lưu ở terminal và thư mục `reports/`, **không hiển thị trên giao diện web**.

---

## 🗺️ Cấu Trúc 3 Khu Vực & 8 Quán Ăn Hư Cấu (`src/config.py`)

Hệ thống tổ chức phân cấp **Khu vực → Quán → Món** (cố ý có các cặp quán cùng dòng món ở khu vực khác nhau với tốc độ bếp khác nhau để so sánh):

| Khu vực | Tên quán hư cấu | Giờ mở cửa | Số món | Đặc điểm nổi bật |
| :--- | :--- | :---: | :---: | :--- |
| **🏙️ Quận Hải Châu** | **Mì Quảng Hạt Sen** | `06:30 – 21:30` | 5 | Mì Quảng làm nhanh hơn (1–3 bếp) |
| | **Cơm Gà Phố Cổ** | `09:30 – 21:30` | 4 | Cơm gà xối mỡ, bếp lớn 2–4 người |
| | **Tiệm Trà Cố Đô** | `07:00 – 22:30` | 5 | Trà hoa & cà phê phin trung tâm |
| **🌊 Quận Sơn Trà** | **Bún Chả Cá Biển Xanh** | `06:00 – 21:00` | 4 | Bún chả cá biển tươi, thời gian ổn định |
| | **Bánh Xèo Sông Hàn** | `10:00 – 22:00` | 4 | Chảo gang truyền thống, lâu hơn khi cao điểm |
| | **Mì Quảng Gió Biển** | `06:30 – 21:00` | 5 | Nấu mẻ nhỏ (1–2 bếp), trùng 3 món với Hạt Sen |
| **⛰️ Quận Ngũ Hành Sơn** | **Trà & Cà Phê Muối Bãi Biển** | `07:00 – 22:30` | 5 | Pha chế nhanh, trùng 3 món với Tiệm Trà Cố Đô |
| | **Bánh Xèo Làng Đá** | `10:30 – 22:00` | 5 | Vỏ mỏng đổ nhanh hơn Sông Hàn, trùng 4 món |

---

## ✨ Tính Năng Nổi Bật Trên Giao Diện Web (`app/static/index.html`)

- **Luồng 4 bước trực quan, trạng thái ban đầu để trống:** `1. Khu vực` → `2. Quán` → `3. Món` → `4. Tùy chọn (Hình thức nhận món, Số lượng, Thời điểm)`. Không chọn sẵn quán/món/kênh, không lưu lựa chọn cũ bằng `localStorage`.
- **Nút Dự đoán thông minh:** Chỉ sáng lên khi đã chọn đủ quán, món và hình thức nhận món; hiển thị rõ thông báo còn thiếu bước nào. Đổi lựa chọn tự động xóa kết quả cũ.
- **Trạng thái bếp chỉ đọc:** Chỉ hiển thị sau khi chọn quán (*"Quán hiện đang: ... · Bếp X người (ước lượng mô phỏng)"*).
- **Ảnh đồng bộ chuẩn tỉ lệ & tương phản WCAG AA:** Ảnh bìa quán **16:9 (`800×450`)**, ảnh món **4:3 (`800×600`)**, có nền placeholder tránh nhảy bố cục, hiển thị sắc nét cả chế độ Sáng và Tối.
- **Chế độ Demo kín đáo ở chân trang:** Mặc định tắt; cho phép ghi đè số đơn chờ (0–35) và số người làm bếp (1–4).

---

## 🖼️ Chuẩn Bị & Đồng Bộ Ảnh (`scripts/prepare_images.py`)

Toàn bộ 35 ảnh (8 ảnh bìa quán + 27 ảnh món, các món trùng giữa các quán dùng chung ảnh) đều được tải cục bộ vào `app/static/images/` (không hotlink, ghi đầy đủ nguồn miễn phí bản quyền trong [CREDITS.md](CREDITS.md)).

Để tự động cắt giữa (center crop), chuyển định dạng WebP và ép đúng kích thước cố định cho ảnh mới:
```powershell
# Chuẩn hóa mọi ảnh đang có trong app/static/images/ và báo ảnh còn thiếu (nếu có)
python scripts/prepare_images.py

# Hoặc nhập trực tiếp ảnh thô (.jpg/.png) từ một thư mục nguồn
python scripts/prepare_images.py --input-dir path/to/raw_photos
```
- Ảnh có tên bắt đầu bằng `store-` sẽ tự cắt tỉ lệ **16:9 (`800×450`)**.
- Ảnh món ăn sẽ tự cắt tỉ lệ **4:3 (`800×600`)**.
- Có thể chỉnh vùng cắt (`CROP_FOCUS`) cho từng file ngay đầu script nếu phần chính của món bị lệch tâm.

---

## ➕ Cách Thêm Quán Mới Vào Cấu Hình

1. Mở [src/config.py](src/config.py), thêm một mục quán vào danh sách `"stores"` của khu vực tương ứng trong `REGIONS`:
   - Điền đầy đủ: `store`, `district`, `emoji`, `description` (ngắn gọn dưới ~75 ký tự để không bị cắt dòng), `open_hours`, `cover_image`, `speed_factor`, `variance`, `staff_range`, `dishes`.
2. Đặt ảnh bìa quán (`store-<slug>.webp`/`.jpg`) và ảnh các món mới (`<slug>.webp`/`.jpg`) vào `app/static/images/` rồi chạy `python scripts/prepare_images.py` (ghi nguồn vào `CREDITS.md`).
3. Chạy lại pipeline:
   ```powershell
   python src/generate_data.py
   python src/train.py
   python src/explain.py
   ```
4. Khởi động lại server `uvicorn app.main:app --port 8000` — giao diện web và API sẽ tự động nhận khu vực/quán/món mới.

---

## 📊 Tóm Tắt Đánh Giá Mô Hình (Terminal & Thư Mục `reports/`)

Chi tiết đầy đủ xem tại [reports/report.md](reports/report.md) và [reports/explain.md](reports/explain.md):

### 1. So sánh Trước / Sau khi mở rộng từ 5 quán (`backup_v2/`) lên 8 quán (15.000 đơn):
| Mô hình / Chỉ số | Trước mở rộng (5 quán · 10.000 đơn) | Sau mở rộng (8 quán · 15.000 đơn) |
| :--- | :---: | :---: |
| **Baseline (TB món) MAE / $R^2$** | 4.08 phút / 0.369 | 3.98 phút / 0.3680 |
| **Linear Regression MAE / $R^2$** | 2.35 phút / 0.656 | 2.34 phút / 0.7028 |
| **Random Forest MAE / RMSE / $R^2$** | 2.25p / 3.94p / 0.6846 | **2.30p / 3.74p / 0.7105** |
| **Random Forest % lệch < 5 phút** | 91.58% | **91.13%** |
| **Độ phủ Conformal (Test, mục tiêu 80%)** | 79.32% | **77.96%** ($q_{calib} = +0.48$p) |

### 2. Đánh giá quán chưa từng thấy (`GroupKFold` theo quán):
- **MAE tổng thể khi ẩn toàn bộ quán khỏi tập train:** **4.25 phút** (so với **2.30 phút** trên tập test thông thường).
- Quán chia sẻ nhóm món và có tốc độ gần mức trung bình (như *Tiệm Trà Cố Đô* 1.88p, *Mì Quảng Hạt Sen* 2.45p) dự đoán khá tốt; ngược lại các quán có tốc độ riêng chậm hơn hẳn (*Bánh Xèo Sông Hàn* 8.19p, *Mì Quảng Gió Biển* 5.97p) hoặc nhóm món duy nhất (*Cơm Gà Phố Cổ* 5.09p) bị lệch hệ thống vì mô hình chưa từng biết hệ số tốc độ của quán mới.

---

## 🛠️ Hướng Dẫn Chạy Toàn Bộ Pipeline

```powershell
cd food-eta
pip install -r requirements.txt

# 1. Chuẩn hóa ảnh (16:9 cho quán, 4:3 cho món)
python scripts/prepare_images.py

# 2. Sinh 15.000 đơn hàng mô phỏng (3 khu vực · 8 quán)
python src/generate_data.py

# 3. Huấn luyện, hiệu chuẩn Conformal & đánh giá quán chưa từng thấy (GroupKFold)
python src/train.py

# 4. Chạy báo cáo giải thích mô hình (PDP, Permutation Importance, Ablation)
python src/explain.py

# 5. Chạy Web Server
uvicorn app.main:app --port 8000
```
Truy cập: 👉 **http://localhost:8000**

---

## 📋 Danh Sách Việc Bạn Cần Tự Làm

1. **Ảnh quán và ảnh món:** Hiện đã đầy đủ **35/35 ảnh chuẩn kích thước** (không còn ảnh thiếu). Nếu sau này bạn muốn thay ảnh chụp thực tế của riêng bạn, chỉ cần đặt file đúng tên slug vào `app/static/images/` rồi chạy lại `python scripts/prepare_images.py`.
2. **Đẩy bản nâng cấp lên GitHub:** File mô hình `models/rf.joblib` đã được nén (`~12.1 MB`, dưới giới hạn 25MB của GitHub Web). Bạn có thể chạy `git add .`, `git commit`, `git push` để cập nhật kho mã nguồn.
3. **Thu thập dữ liệu thật cho quán mới:** Khi muốn tinh chỉnh mô hình sát thực tế cho quán mới mở, điền đơn thật vào `data/real_orders_template.csv` rồi chạy `python src/evaluate_real.py --retrain`.
