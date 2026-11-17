# Hướng Dẫn Triển Khai Dự Án Food ETA Đà Nẵng Lên Mạng Internet (Production Deployment)

Tài liệu này hướng dẫn chi tiết từng bước (dành cho người mới) để đưa trang web **Food ETA Đà Nẵng** lên mạng Internet có địa chỉ công khai để bạn bè hoặc khách hàng có thể truy cập trên điện thoại và máy tính.

---

## 1. Chuẩn Bị & Đẩy Code Lên GitHub

Mọi nền tảng đám mây hiện đại (Render, Hugging Face, Vercel, Railway) đều kết nối trực tiếp với kho lưu trữ GitHub của bạn để tự động triển khai.

### Các bước thực hiện:
1. Đăng nhập vào [GitHub.com](https://github.com) và tạo một repository mới (ví dụ đặt tên: `food-eta-danang`), chọn chế độ **Public** (hoặc **Private**).
2. Mở Terminal (PowerShell) tại thư mục dự án `food-eta` trên máy bạn:
   ```powershell
   cd "d:\Trần Tuấn Đô\DỰ ÁN PYTHON\duandonhangn\food-eta"
   git init
   git add .
   git commit -m "feat: complete Food ETA Da Nang web app"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/food-eta-danang.git
   git push -u origin main
   ```

---

## 2. Triển Khai Miễn Phí Trên Render (Khuyên dùng)

Render cung cấp gói miễn phí (Free Tier) hỗ trợ Python/FastAPI rất tốt. Dự án đã được tạo sẵn file cấu hình `render.yaml`.

### Cách 1: Triển khai tự động bằng Blueprint
1. Truy cập [dashboard.render.com](https://dashboard.render.com) (đăng nhập bằng tài khoản GitHub).
2. Bấm nút **New +** ở góc trên bên phải → chọn **Blueprint**.
3. Chọn repository `food-eta-danang` bạn vừa đẩy lên.
4. Render sẽ tự đọc file `render.yaml` có sẵn và hiển thị tên dịch vụ. Bấm **Apply**.
5. Đợi Render cài đặt thư viện và huấn luyện mô hình (khoảng 2-3 phút).

### Cách 2: Triển khai thủ công (Manual Web Service)
1. Tại Render Dashboard, bấm **New +** → chọn **Web Service**.
2. Kết nối với repo `food-eta-danang`.
3. Điền các thông số:
   - **Name:** Nhập tên dịch vụ bạn muốn, ví dụ: `food-eta-danang` (địa chỉ web của bạn sẽ là: `https://food-eta-danang.onrender.com`).
   - **Region:** Singapore (để tốc độ tải từ Việt Nam nhanh nhất).
   - **Environment:** `Python 3`.
   - **Build Command:** 
     ```bash
     pip install -r requirements.txt && python src/generate_data.py && python src/train.py
     ```
   - **Start Command:** 
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan:** Chọn **Free**.
4. Bấm **Create Web Service**. Khi màn hình hiển thị `Application startup complete`, trang web của bạn đã chính thức online!

---

## 3. Triển Khai Trên Hugging Face Spaces (Dành cho ML & AI)

Hugging Face Spaces là nền tảng máy chủ AI miễn phí tuyệt vời, không giới hạn RAM và hỗ trợ Docker.

1. Đăng ký/Đăng nhập tại [huggingface.co](https://huggingface.co).
2. Bấm vào ảnh đại diện → chọn **New Space**.
3. Điền thông tin:
   - **Space name:** `food-eta-danang`.
   - **License:** `mit` hoặc `apache-2.0`.
   - **Select the Space SDK:** Chọn **Docker** (chọn Blank).
   - **Space hardware:** `CPU basic · 2 vCPU · 16GB RAM · Free`.
4. Bấm **Create Space**.
5. Sao chép toàn bộ code trong thư mục `food-eta/` đẩy lên Space bằng Git:
   ```powershell
   git remote add space https://huggingface.co/spaces/<username>/food-eta-danang
   git push space main
   ```
6. Hugging Face sẽ tự động đọc file `Dockerfile` đã có sẵn trong dự án (chạy cổng 7860) và khởi động ứng dụng của bạn.

---

## 4. Gắn Tên Miền Riêng (Custom Domain)

Nếu bạn có tên miền riêng (ví dụ: `foodeta.vn` hoặc `danangfood.com`):

1. Vào Dashboard dịch vụ trên Render → chọn Web Service của bạn.
2. Vào mục **Settings** ở thanh bên trái → cuộn xuống phần **Custom Domains**.
3. Bấm **Add Custom Domain** và nhập tên miền của bạn (ví dụ: `app.danangfood.com`).
4. Truy cập trang quản trị tên miền của bạn (Cloudflare, Namecheap, MatBao, PA Vietnam...), tạo một bản ghi **CNAME**:
   - **Type:** `CNAME`
   - **Name:** `app` (hoặc `@` nếu là tên miền gốc)
   - **Target / Content:** Địa chỉ onrender được cấp (ví dụ: `food-eta-danang.onrender.com`).
5. Render sẽ tự động cấp chứng chỉ bảo mật SSL (HTTPS) miễn phí trong vòng vài phút.

---

## 5. Demo Tạm Thời Từ Máy Cá Nhân (Không Cần Đẩy Code)

Nếu bạn muốn gửi link cho bạn bè xem ngay mà chưa muốn đưa lên Render, bạn có thể tạo đường hầm (Tunnel) trực tiếp từ máy tính của bạn:

### Cách A: Dùng Cloudflare Tunnel (Miễn phí, không cần đăng nhập)
1. Tải công cụ `cloudflared` từ [Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
2. Đảm bảo app đang chạy trên máy (`uvicorn app.main:app --port 8000`).
3. Mở cửa sổ Terminal mới và chạy:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```
4. Cloudflare sẽ tạo ra một link tạm thời dạng: `https://random-words.trycloudflare.com` có thể truy cập được từ bất kỳ đâu trên thế giới!

### Cách B: Dùng Ngrok
1. Tải [ngrok](https://ngrok.com/) và đăng ký tài khoản miễn phí.
2. Chạy lệnh:
   ```powershell
   ngrok http 8000
   ```
3. Sao chép link `https://...ngrok-free.app` gửi cho bạn bè để trải nghiệm ngay.

---

## 6. Lưu Ý Quan Trọng Về Gói Miễn Phí (Free Tier Spin-down)

- **Cơ chế "ngủ đông" (Spin-down):** Trên Render Free Tier, nếu không có ai truy cập trong vòng 15 phút, máy chủ sẽ tự động tạm dừng để tiết kiệm tài nguyên.
- **Thời gian đánh thức (Cold Start):** Khi có một người mới mở lại trang web sau khi máy chủ đã "ngủ", Render sẽ cần khoảng **30 đến 50 giây** để khởi động lại máy chủ. Các lượt truy cập tiếp theo sẽ phản hồi ngay lập tức (<100ms).
- **Mẹo giữ máy chủ luôn thức:** Bạn có thể dùng các dịch vụ ping miễn phí như [UptimeRobot](https://uptimerobot.com) hoặc [Cron-job.org](https://cron-job.org) để gửi request tự động đến đường dẫn `https://ten-dich-vu.onrender.com/health` mỗi 10 phút một lần.

