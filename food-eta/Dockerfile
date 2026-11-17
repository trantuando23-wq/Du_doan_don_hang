# Dockerfile - Hỗ trợ triển khai Docker độc lập & Hugging Face Spaces (Port 7860)
FROM python:3.11-slim

# Thiết lập biến môi trường hệ thống
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PORT=7860

WORKDIR /app

# Cài đặt curl phục vụ healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào image
COPY . .

# Tự động sinh dữ liệu và huấn luyện mô hình khi build image (đảm bảo đồng nhất môi trường)
RUN python src/generate_data.py && python src/train.py

# Cấp quyền đọc ghi cho user mặc định của Hugging Face Spaces (user ID 1000)
RUN useradd -m -u 1000 user || true
RUN chown -R 1000:1000 /app
USER 1000

# Mở cổng mặc định (7860 cho Hugging Face Spaces, hoặc ghi đè bằng biến $PORT trên Render/Cloud Run)
EXPOSE 7860

# Khởi động máy chủ FastAPI qua Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

