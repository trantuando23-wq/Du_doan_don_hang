"""
main.py - FastAPI Backend cho dự án Bếp Đà Nẵng (Food ETA).
Cung cấp các API:
  - GET /health: Health check cho Render, Hugging Face Spaces & load balancers
  - GET /api/catalog: Danh mục quán hư cấu, món ăn, ảnh, quận huyện
  - POST /api/predict: Dự đoán thời gian (RF với khoảng tin cậy Split Conformal, giải thích lý do)
  - GET /api/hourly-eta: Đồ thị thời gian theo từng giờ và gợi ý giờ đặt nhanh nhất
  - POST /api/feedback: Ghi nhận thời gian thực tế của người dùng (Thread-safe, rate-limited)
  - GET /api/feedback/export: Xuất dữ liệu phản hồi (yêu cầu ADMIN_TOKEN)
  - GET /api/metrics: API nội bộ chỉ mở khi có ADMIN_TOKEN
  - GET /: Giao diện web chính
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json
import threading
import logging
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Header, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

# Cấu hình logging server-side
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("food-eta")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    REGIONS, STORES, DISH_GROUP_NAMES, CHANNELS, CHANNEL_LABELS, CHANNEL_EFFECTS,
    MODELS_DIR, DATA_DIR, STATIC_DIR, FEEDBACK_FILE
)

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
WEEKDAY_VIETNAMESE = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

app = FastAPI(
    title="Bếp Đà Nẵng · Food ETA API",
    description="Hệ thống AI dự đoán thời gian hoàn thành món ăn tại Đà Nẵng",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp models & caches
RF_BUNDLE = None
CALIBRATION_CACHE = None
QUEUE_PROFILE_CACHE = None
feedback_lock = threading.Lock()

# Rate limiter trong bộ nhớ (In-memory IP rate limiter)
ip_rate_limits = defaultdict(list)
rate_limit_lock = threading.Lock()

def check_rate_limit(ip: str, max_requests: int = 60, window_seconds: int = 60):
    now = datetime.now()
    cutoff = now - timedelta(seconds=window_seconds)
    with rate_limit_lock:
        timestamps = ip_rate_limits[ip]
        # Xóa các mốc thời gian quá hạn
        ip_rate_limits[ip] = [t for t in timestamps if t > cutoff]
        if len(ip_rate_limits[ip]) >= max_requests:
            return False
        ip_rate_limits[ip].append(now)
        return True

def load_resources():
    global RF_BUNDLE, CALIBRATION_CACHE, QUEUE_PROFILE_CACHE
    rf_path = MODELS_DIR / "rf.joblib"
    calib_path = MODELS_DIR / "calibration.json"
    qp_path = MODELS_DIR / "queue_profile.json"
    
    if rf_path.exists():
        RF_BUNDLE = joblib.load(rf_path)
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            CALIBRATION_CACHE = json.load(f)
    if qp_path.exists():
        with open(qp_path, "r", encoding="utf-8") as f:
            QUEUE_PROFILE_CACHE = json.load(f)

load_resources()

# Tự động lắng nghe thêm cổng 80 cho máy local nếu khả dụng
def start_optional_port_80_forwarder(target_port: int = 8000):
    import socket
    def handle_client(client_socket):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.connect(('127.0.0.1', target_port))
            def forward(src, dst):
                try:
                    while True:
                        data = src.recv(4096)
                        if not data: break
                        dst.sendall(data)
                except Exception: pass
                finally:
                    src.close(); dst.close()
            threading.Thread(target=forward, args=(client_socket, server_socket), daemon=True).start()
            threading.Thread(target=forward, args=(server_socket, client_socket), daemon=True).start()
        except Exception:
            client_socket.close()

    try:
        proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        proxy.bind(('0.0.0.0', 80))
        proxy.listen(50)
        while True:
            client, _ = proxy.accept()
            threading.Thread(target=handle_client, args=(client,), daemon=True).start()
    except Exception:
        pass

threading.Thread(target=start_optional_port_80_forwarder, args=(8000,), daemon=True).start()

# Custom handler để lỗi Pydantic 422 trả thông điệp tiếng Việt thân thiện, gọn gàng
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        first_msg = str(errors[0].get("msg", "Dữ liệu không hợp lệ"))
        # Loại bỏ tiền tố "Value error, " do Pydantic v2 thêm vào
        if first_msg.startswith("Value error, "):
            first_msg = first_msg[len("Value error, "):]
        return JSONResponse(status_code=422, content={"detail": first_msg})
    return JSONResponse(status_code=422, content={"detail": "Dữ liệu gửi lên không hợp lệ."})

# Schema Pydantic
class CartItem(BaseModel):
    dish: str = Field(..., description="Tên món ăn")
    qty: int = Field(1, ge=1, le=20, description="Số lượng phần")

class PredictRequest(BaseModel):
    store: str = Field(..., description="Tên quán ăn ở Đà Nẵng")
    dish: str = Field(..., description="Tên món ăn trong thực đơn của quán")
    qty: int = Field(1, ge=1, le=20, description="Số lượng phần ăn")
    channel: str | None = Field(None, description="Hình thức nhận món bắt buộc: tai_quan, mang_di, giao_app")
    mode: str = Field("now", description="Chế độ đặt: 'now' (Bây giờ) hoặc 'scheduled' (Hẹn giờ)")
    hour: int | None = Field(None, ge=0, le=23, description="Giờ hẹn (0-23)")
    minute: int | None = Field(None, ge=0, le=59, description="Phút hẹn (0-59)")
    weekday: str | None = Field(None, description="Thứ hẹn (Thứ Hai -> Chủ Nhật)")
    queue_override: int | None = Field(None, ge=0, le=50, description="Chế độ Demo ghi đè số đơn")
    staff_override: int | None = Field(None, ge=1, le=4, description="Chế độ Demo ghi đè số nhân viên bếp (1-4)")
    cart: list[CartItem] | None = Field(None, description="Danh sách các món trong giỏ hàng")

    @model_validator(mode="after")
    def validate_catalog_and_channel(self):
        if not self.store or self.store not in STORES:
            raise ValueError(f"Quán '{self.store}' không tồn tại trong hệ thống.")
        store_dishes = {d["dish"] for d in STORES[self.store]["dishes"]}
        if not self.dish or self.dish not in store_dishes:
            raise ValueError(f"Món '{self.dish}' không có trong thực đơn của quán '{self.store}'.")
        if self.channel is None or str(self.channel).strip() == "":
            raise ValueError("Vui lòng chọn hình thức nhận món")
        if self.channel not in CHANNELS:
            raise ValueError("Hình thức nhận món không hợp lệ (chọn: Ăn tại quán, Mang đi hoặc Giao qua app).")
        if self.cart:
            for item in self.cart:
                if item.dish not in store_dishes:
                    raise ValueError(f"Món '{item.dish}' trong giỏ hàng không thuộc thực đơn của quán '{self.store}'.")
        return self

class FeedbackRequest(BaseModel):
    store: str = Field(...)
    dish: str = Field(...)
    qty: int = Field(1, ge=1, le=20)
    channel: str = Field("tai_quan", description="Hình thức nhận món: tai_quan, mang_di, giao_app")
    staff: int = Field(2, ge=1, le=4, description="Số nhân viên bếp (1-4)")
    hour: int = Field(..., ge=0, le=23)
    weekday: str = Field(...)
    queue_estimated: int = Field(..., ge=0)
    predicted_minutes: float = Field(..., ge=0.5)
    actual_minutes: float = Field(..., ge=1.0, le=180.0, description="Số phút đợi thực tế (1 - 180)")
    is_demo: int = Field(0, ge=0, le=1)

def get_vietnam_now():
    return datetime.now(VN_TZ)

def classify_queue(queue_count: int) -> dict:
    if queue_count <= 4:
        return {"label": "Vắng khách", "key": "vang", "color": "emerald"}
    elif queue_count <= 11:
        return {"label": "Vừa phải", "key": "vua", "color": "blue"}
    elif queue_count <= 18:
        return {"label": "Khá đông", "key": "kha_dong", "color": "amber"}
    else:
        return {"label": "Rất đông", "key": "rat_dong", "color": "rose"}

def get_profile_queue(store: str, is_weekend: int, hour: int) -> float:
    global QUEUE_PROFILE_CACHE
    if QUEUE_PROFILE_CACHE is None:
        load_resources()
    day_type = "weekend" if is_weekend == 1 else "weekday"
    hour_str = str(hour)
    base_queue = 6.0
    if QUEUE_PROFILE_CACHE and "profile" in QUEUE_PROFILE_CACHE:
        store_prof = QUEUE_PROFILE_CACHE["profile"].get(store, {})
        day_prof = store_prof.get(day_type, {})
        base_queue = day_prof.get(hour_str, 6.0)
    return base_queue

def get_profile_staff(store: str, is_weekend: int, hour: int) -> int:
    global QUEUE_PROFILE_CACHE
    if QUEUE_PROFILE_CACHE is None:
        load_resources()
    day_type = "weekend" if is_weekend == 1 else "weekday"
    hour_str = str(hour)
    default_staff = 2
    if QUEUE_PROFILE_CACHE and "staff_profile" in QUEUE_PROFILE_CACHE:
        store_staff = QUEUE_PROFILE_CACHE["staff_profile"].get(store, {})
        day_staff = store_staff.get(day_type, {})
        return int(day_staff.get(hour_str, default_staff))
    return default_staff

def estimate_queue(store: str, is_weekend: int, hour: int, mode: str, now_vn: datetime) -> tuple[int, str]:
    base_queue = get_profile_queue(store, is_weekend, hour)
    if mode == "now":
        time_bucket = f"{now_vn.year}_{now_vn.month}_{now_vn.day}_{now_vn.hour}_{now_vn.minute // 5}"
        seed_str = f"{store}_{time_bucket}"
        hash_val = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:6], 16)
        jitter = (hash_val % 5) - 2
        queue_val = max(1, int(round(base_queue + jitter)))
        return queue_val, "ước lượng thời gian thực (ổn định 5 phút)"
    else:
        queue_val = max(1, int(round(base_queue)))
        return queue_val, "ước lượng theo hồ sơ lịch sử giờ hẹn"

def _predict_raw(pipeline, store: str, dish: str, dish_group: str, qty: int, hour: int, weekday: str, is_weekend: int, queue: int, staff: int, channel: str, use_log: bool):
    """Hàm chạy pipeline nội bộ cho 1 món."""
    row = pd.DataFrame([{
        "store": store,
        "dish": dish,
        "dish_group": dish_group,
        "qty": qty,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "queue": queue,
        "staff": staff,
        "channel": channel
    }])
    pred_raw = pipeline.predict(row)[0]
    pred = np.exp(pred_raw) if use_log else pred_raw
    return max(2.0, round(float(pred), 1))

@app.get("/health")
def health_check():
    now_vn = get_vietnam_now()
    return {
        "status": "healthy",
        "service": "food-eta",
        "models_loaded": RF_BUNDLE is not None,
        "vietnam_time": now_vn.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Ho_Chi_Minh"
    }

@app.get("/api/catalog")
def get_catalog():
    """
    Trả về cấu trúc phân cấp Khu vực -> Quán -> Món ăn.
    KHÔNG trả thời gian gốc (base_time) hay nhóm món (group) ra cho giao diện người dùng.
    """
    regions_payload = []
    for reg in REGIONS:
        stores_payload = []
        for s_info in reg["stores"]:
            dishes_list = [
                {
                    "name": d["dish"],
                    "slug": d["slug"],
                    "image": f"/static/images/{d['image']}"
                }
                for d in s_info["dishes"]
            ]
            stores_payload.append({
                "store": s_info["store"],
                "district": s_info["district"],
                "region_id": reg["id"],
                "region_name": reg["name"],
                "emoji": s_info["emoji"],
                "description": s_info["description"],
                "open_hours": s_info["open_hours"],
                "cover_image": f"/static/images/{s_info['cover_image']}",
                "dishes": dishes_list
            })
        regions_payload.append({
            "id": reg["id"],
            "name": reg["name"],
            "emoji": reg["emoji"],
            "tagline": reg["tagline"],
            "store_count": len(stores_payload),
            "stores": stores_payload
        })

    now_vn = get_vietnam_now()
    return {
        "regions": regions_payload,
        "current_time": {
            "hour": now_vn.hour,
            "minute": now_vn.minute,
            "weekday": WEEKDAY_VIETNAMESE[now_vn.weekday()],
            "formatted": now_vn.strftime("%H:%M, %d/%m/%Y"),
            "timezone": "Asia/Ho_Chi_Minh"
        }
    }

@app.get("/api/store-status")
def get_store_status(
    store: str,
    mode: str = "now",
    hour: int | None = None,
    weekday: str | None = None
):
    """Trả về trạng thái ước lượng (đông/vắng, đơn chờ, nhân sự bếp) cho 1 quán đã chọn."""
    if store not in STORES:
        raise HTTPException(status_code=400, detail=f"Quán '{store}' không tồn tại trong hệ thống.")
    now_vn = get_vietnam_now()
    m = mode.lower().strip()
    if m == "now":
        h = now_vn.hour
        wd = WEEKDAY_VIETNAMESE[now_vn.weekday()]
    else:
        h = hour if (hour is not None and 0 <= hour <= 23) else 12
        wd = weekday if weekday in WEEKDAY_VIETNAMESE else WEEKDAY_VIETNAMESE[now_vn.weekday()]
    is_weekend = 1 if wd in ["Thứ Bảy", "Chủ Nhật"] else 0

    queue, queue_note = estimate_queue(store, is_weekend, h, m, now_vn)
    staff = get_profile_staff(store, is_weekend, h)
    queue_meta = classify_queue(queue)
    status_text = (
        f"Quán hiện đang: {queue_meta['label']} (~{queue} đơn chờ) · Bếp {staff} người"
        if m == "now" else
        f"Thời điểm này quán dự kiến: {queue_meta['label']} (~{queue} đơn chờ) · Bếp {staff} người"
    )
    return {
        "store": store,
        "queue_estimated": queue,
        "queue_label": queue_meta["label"],
        "queue_badge_color": queue_meta["color"],
        "staff_estimated": staff,
        "staff_label": f"Bếp {staff} người",
        "queue_status_text": status_text,
        "queue_note": queue_note
    }

@app.post("/api/predict")
def predict_eta(req: PredictRequest, request: Request):
    """
    Dự đoán thời gian hoàn thành (Random Forest đã hiệu chỉnh Split Conformal).
    Tự động sinh mảng lý do giải thích (reasons) và ước lượng đơn hàng chờ.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, max_requests=60, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Bạn thao tác quá nhanh. Vui lòng thử lại sau giây lát."
        )

    try:
        if req.store not in STORES:
            raise HTTPException(status_code=400, detail=f"Quán '{req.store}' không tồn tại.")
        store_info = STORES[req.store]
        dishes_dict = {d["dish"]: d for d in store_info["dishes"]}
        if req.dish not in dishes_dict:
            raise HTTPException(status_code=400, detail=f"Món '{req.dish}' không có trong thực đơn quán.")
        if not req.channel or req.channel not in CHANNELS:
            raise HTTPException(status_code=400, detail="Vui lòng chọn hình thức nhận món")

        dish_info = dishes_dict[req.dish]
        dish_group = dish_info["group"]
        now_vn = get_vietnam_now()

        mode = req.mode.lower().strip()
        if mode == "now":
            hour = now_vn.hour
            minute = now_vn.minute
            weekday = WEEKDAY_VIETNAMESE[now_vn.weekday()]
            target_dt = now_vn
            time_display = f"{hour:02d}:{minute:02d} ({weekday})"
        else:
            hour = req.hour if req.hour is not None else 12
            minute = req.minute if req.minute is not None else 0
            weekday = req.weekday if req.weekday in WEEKDAY_VIETNAMESE else WEEKDAY_VIETNAMESE[now_vn.weekday()]
            target_dt = now_vn.replace(hour=hour, minute=minute, second=0, microsecond=0)
            time_display = f"{hour:02d}:{minute:02d} ({weekday})"

        is_weekend = 1 if weekday in ["Thứ Bảy", "Chủ Nhật"] else 0

        # Channel đã được validate bởi Pydantic & kiểm tra trên
        channel = req.channel

        # Ước lượng Queue & Staff
        is_demo_used = 0
        if req.queue_override is not None:
            queue = max(0, min(50, int(req.queue_override)))
            queue_note = "Chế độ Demo (Ghi đè đơn)"
            is_demo_used = 1
        else:
            queue, queue_note = estimate_queue(req.store, is_weekend, hour, mode, now_vn)

        if req.staff_override is not None:
            staff = max(1, min(4, int(req.staff_override)))
            staff_note = "Chế độ Demo (Ghi đè nhân sự)"
            is_demo_used = 1
        else:
            staff = get_profile_staff(req.store, is_weekend, hour)
            staff_note = "Ước lượng theo ca làm"

        queue_meta = classify_queue(queue)
        queue_status_text = (
            f"Quán hiện đang: {queue_meta['label']} (~{queue} đơn chờ) · Bếp {staff} người (ước lượng mô phỏng)"
            if mode == "now" else
            f"Thời điểm này quán dự kiến: {queue_meta['label']} (~{queue} đơn chờ) · Bếp {staff} người (ước lượng mô phỏng)"
        )

        if RF_BUNDLE is None:
            load_resources()
            if RF_BUNDLE is None:
                raise HTTPException(status_code=500, detail="Mô hình chưa sẵn sàng. Vui lòng liên hệ quản trị.")

        rf_pipeline = RF_BUNDLE["pipeline"]
        rf_log = RF_BUNDLE.get("use_log_transform", True)

        # 1. Dự đoán món chính
        input_row = pd.DataFrame([{
            "store": req.store,
            "dish": req.dish,
            "dish_group": dish_group,
            "qty": req.qty,
            "hour": hour,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "queue": queue,
            "staff": staff,
            "channel": channel
        }])

        pred_raw = rf_pipeline.predict(input_row)[0]
        pred_rf = np.exp(pred_raw) if rf_log else pred_raw
        pred_rf = max(2.0, round(float(pred_rf), 1))

        # 2. Tính khoảng dự đoán ĐÃ HIỆU CHUẨN (Split Conformal Calibration)
        rf_prep = rf_pipeline.named_steps["preprocessor"]
        rf_reg = rf_pipeline.named_steps["regressor"]
        X_trans = rf_prep.transform(input_row)
        tree_preds = np.array([tree.predict(X_trans)[0] for tree in rf_reg.estimators_])
        if rf_log:
            tree_preds = np.exp(tree_preds)

        L_raw = float(np.percentile(tree_preds, 10))
        U_raw = float(np.percentile(tree_preds, 90))

        q_calib = 0.46
        if CALIBRATION_CACHE and "q_calib" in CALIBRATION_CACHE:
            q_calib = float(CALIBRATION_CACHE["q_calib"])

        p10 = max(2.0, round(L_raw - q_calib, 1))
        p90 = round(U_raw + q_calib, 1)
        p10 = min(p10, pred_rf)
        p90 = max(p90, pred_rf)

        # 3. GIẢI THÍCH TỪNG DỰ ĐOÁN (COUNTERFACTUAL EXPLANATION)
        reasons = []
        is_peak = (11 <= hour <= 13) or (18 <= hour <= 20)
        
        # Yếu tố 1: Giờ cao điểm
        if is_peak:
            offpeak_hour = 9
            offpeak_queue = int(round(get_profile_queue(req.store, is_weekend, offpeak_hour)))
            offpeak_staff = get_profile_staff(req.store, is_weekend, offpeak_hour)
            pred_offpeak = _predict_raw(rf_pipeline, req.store, req.dish, dish_group, req.qty, offpeak_hour, weekday, is_weekend, offpeak_queue, offpeak_staff, channel, rf_log)
            delta_peak = round(pred_rf - pred_offpeak, 1)
            if delta_peak >= 1.0:
                reasons.append(f"Giờ cao điểm đông khách (+{int(round(delta_peak))} phút)")

        # Yếu tố 2: Hàng đợi (Queue)
        if queue > 4:
            pred_empty_queue = _predict_raw(rf_pipeline, req.store, req.dish, dish_group, req.qty, hour, weekday, is_weekend, 2, staff, channel, rf_log)
            delta_queue = round(pred_rf - pred_empty_queue, 1)
            if delta_queue >= 1.0:
                reasons.append(f"Quán đang có nhiều đơn chờ (+{int(round(delta_queue))} phút)")

        # Yếu tố 3: Số lượng đặt
        if req.qty > 1:
            pred_single_qty = _predict_raw(rf_pipeline, req.store, req.dish, dish_group, 1, hour, weekday, is_weekend, queue, staff, channel, rf_log)
            delta_qty = round(pred_rf - pred_single_qty, 1)
            if delta_qty >= 1.0:
                reasons.append(f"Đặt số lượng {req.qty} phần ăn (+{int(round(delta_qty))} phút)")

        # Yếu tố 4: Nhân sự bếp (Staff) - So sánh với mức đủ người của quán
        max_staff = store_info.get("staff_range", (1, 3))[1]
        if staff < max_staff:
            pred_full_staff = _predict_raw(rf_pipeline, req.store, req.dish, dish_group, req.qty, hour, weekday, is_weekend, queue, max_staff, channel, rf_log)
            delta_staff = round(pred_rf - pred_full_staff, 1)
            if delta_staff >= 1.0:
                reasons.append(f"Bếp đang ít người phục vụ (+{int(round(delta_staff))} phút)")

        # Yếu tố 5: Hình thức nhận món (Channel) - So sánh với Ăn tại quán
        if channel != "tai_quan":
            pred_tai_quan = _predict_raw(rf_pipeline, req.store, req.dish, dish_group, req.qty, hour, weekday, is_weekend, queue, staff, "tai_quan", rf_log)
            delta_channel = round(pred_rf - pred_tai_quan, 1)
            if delta_channel >= 1.0:
                if channel == "mang_di":
                    reasons.append(f"Đóng gói mang đi (+{int(round(delta_channel))} phút)")
                elif channel == "giao_app":
                    reasons.append(f"Đóng gói & chờ tài xế gom đơn (+{int(round(delta_channel))} phút)")

        if not reasons:
            reasons_display = ["Thời gian chuẩn bị trong điều kiện bình thường."]
            reason_summary = "Thời gian chuẩn bị bình thường"
        else:
            reasons_display = reasons
            reason_summary = "Lâu hơn thường do: " + " và ".join(reasons)

        # 4. Giỏ hàng nhiều món
        total_eta = pred_rf
        total_p10 = p10
        total_p90 = p90
        is_multi_item = False
        cart_predictions = []

        if req.cart and len(req.cart) > 0:
            is_multi_item = True
            rf_times = []
            for c_item in req.cart:
                if c_item.dish in dishes_dict:
                    c_group = dishes_dict[c_item.dish]["group"]
                    c_eta = _predict_raw(rf_pipeline, req.store, c_item.dish, c_group, c_item.qty, hour, weekday, is_weekend, queue, staff, channel, rf_log)
                    rf_times.append(c_eta)
                    cart_predictions.append({"dish": c_item.dish, "qty": c_item.qty, "pred_rf": c_eta})
            if rf_times:
                total_eta = round(max(rf_times) + 2.0, 1)
                total_p10 = round(p10 + 1.5, 1)
                total_p90 = round(p90 + 2.5, 1)

        ready_dt = target_dt + timedelta(minutes=total_eta)
        ready_time_str = ready_dt.strftime("%H:%M")

        return {
            "store": req.store,
            "district": store_info["district"],
            "dish": req.dish,
            "qty": req.qty,
            "channel": channel,
            "channel_label": CHANNEL_LABELS.get(channel, "Ăn tại quán"),
            "mode": mode,
            "time_display": time_display,
            "hour": hour,
            "minute": minute,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "queue_estimated": queue,
            "queue_label": queue_meta["label"],
            "queue_badge_color": queue_meta["color"],
            "queue_status_text": queue_status_text,
            "queue_note": queue_note,
            "staff_estimated": staff,
            "staff_label": f"Bếp {staff} người",
            "staff_note": staff_note,
            "is_multi_item": is_multi_item,
            "is_demo": is_demo_used,
            "cart_details": cart_predictions if is_multi_item else None,
            "reasons": reasons_display,
            "reason_summary": reason_summary,
            "prediction": {
                "random_forest": total_eta,
                "interval_10_90": [total_p10, total_p90],
                "calibrated_coverage": "80%",
                "estimated_ready_time": ready_time_str
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi dự đoán: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Đã xảy ra sự cố trong quá trình tính toán. Vui lòng thử lại sau ít phút."
        )

@app.get("/api/hourly-eta")
def get_hourly_eta(store: str, dish: str, qty: int = 1, channel: str = "tai_quan", weekday: str | None = None, current_hour: int | None = None):
    """
    Trả về đồ thị thời gian hoàn thành theo từng giờ trong ngày (7h - 22h)
    và gợi ý giờ đặt nhanh nhất cho người dùng.
    """
    if store not in STORES:
        raise HTTPException(status_code=400, detail="Quán không tồn tại.")
    store_info = STORES[store]
    dishes_dict = {d["dish"]: d for d in store_info["dishes"]}
    if dish not in dishes_dict:
        raise HTTPException(status_code=400, detail="Món không tồn tại.")

    dish_group = dishes_dict[dish]["group"]
    now_vn = get_vietnam_now()
    if not weekday or weekday not in WEEKDAY_VIETNAMESE:
        weekday = WEEKDAY_VIETNAMESE[now_vn.weekday()]
    is_weekend = 1 if weekday in ["Thứ Bảy", "Chủ Nhật"] else 0

    if current_hour is None:
        current_hour = now_vn.hour

    if channel not in CHANNELS:
        channel = "tai_quan"

    if RF_BUNDLE is None:
        load_resources()
    rf_pipeline = RF_BUNDLE["pipeline"]
    rf_log = RF_BUNDLE.get("use_log_transform", True)

    hours = list(range(7, 23))
    hourly_results = []
    min_hour = 7
    min_eta = 999.0

    curr_eta = 15.0

    for h in hours:
        q = int(round(get_profile_queue(store, is_weekend, h)))
        s = get_profile_staff(store, is_weekend, h)
        eta = _predict_raw(rf_pipeline, store, dish, dish_group, qty, h, weekday, is_weekend, q, s, channel, rf_log)
        hourly_results.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "eta": eta,
            "queue": q,
            "staff": s,
            "channel": channel,
            "is_peak": (11 <= h <= 13) or (18 <= h <= 20)
        })
        if eta < min_eta:
            min_eta = eta
            min_hour = h
        if h == current_hour:
            curr_eta = eta

    # So sánh thời gian
    diff = round(curr_eta - min_eta, 1)
    if diff >= 2.0:
        recommendation = f"Đặt lúc {min_hour:02d}:00 sẽ nhanh hơn khoảng {int(round(diff))} phút so với thời điểm bạn chọn."
        has_rec = True
    else:
        recommendation = "Khung giờ bạn chọn đã khá thuận lợi và nhanh chóng."
        has_rec = False

    return {
        "store": store,
        "dish": dish,
        "channel": channel,
        "channel_label": CHANNEL_LABELS.get(channel, "Ăn tại quán"),
        "current_hour": current_hour,
        "current_eta": curr_eta,
        "best_hour": min_hour,
        "best_eta": min_eta,
        "diff_minutes": diff,
        "recommendation": recommendation,
        "has_recommendation": has_rec,
        "hourly_data": hourly_results
    }

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, request: Request):
    """
    Ghi nhận thời gian thực tế người dùng đã đợi vào data/feedback.csv.
    Bảo vệ bằng threading lock và giới hạn tần suất gửi theo IP.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(f"feedback_{client_ip}", max_requests=10, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Bạn gửi phản hồi quá nhanh. Vui lòng thử lại sau ít phút."
        )

    # Đảm bảo thư mục data tồn tại
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    channel = req.channel if req.channel in CHANNELS else "tai_quan"
    staff = max(1, min(4, int(req.staff)))

    row_dict = {
        "timestamp": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "store": req.store,
        "dish": req.dish,
        "qty": req.qty,
        "channel": channel,
        "staff": staff,
        "hour": req.hour,
        "weekday": req.weekday,
        "queue_estimated": req.queue_estimated,
        "predicted_minutes": round(req.predicted_minutes, 1),
        "actual_minutes": round(req.actual_minutes, 1),
        "is_demo": req.is_demo
    }

    with feedback_lock:
        file_exists = FEEDBACK_FILE.exists() and FEEDBACK_FILE.stat().st_size > 0
        df_row = pd.DataFrame([row_dict])
        df_row.to_csv(FEEDBACK_FILE, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")

    logger.info(f"Feedback recorded: {req.store} - {req.dish}: Pred={req.predicted_minutes}m, Actual={req.actual_minutes}m, Channel={channel}, Staff={staff}")
    return {
        "status": "success",
        "message": "Cảm ơn bạn! Thông tin thực tế đã được ghi nhận để cải thiện độ chính xác mô hình."
    }

@app.get("/api/feedback/export")
def export_feedback(x_admin_token: str | None = Header(None, alias="X-Admin-Token")):
    """Xuất file feedback.csv - Yêu cầu ADMIN_TOKEN bảo mật."""
    admin_token_env = os.environ.get("ADMIN_TOKEN")
    if not admin_token_env:
        raise HTTPException(status_code=404, detail="Endpoint export không khả dụng. Chưa thiết lập ADMIN_TOKEN.")
    if x_admin_token != admin_token_env:
        raise HTTPException(status_code=401, detail="Xác thực không thành công. Sai ADMIN_TOKEN.")
    if not FEEDBACK_FILE.exists():
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu phản hồi nào.")
    return FileResponse(FEEDBACK_FILE, media_type="text/csv", filename="feedback.csv")

@app.get("/api/metrics")
def get_metrics_protected(x_admin_token: str | None = Header(None, alias="X-Admin-Token")):
    """Endpoint nội bộ trả về metrics - Chỉ truy cập được khi có ADMIN_TOKEN."""
    admin_token_env = os.environ.get("ADMIN_TOKEN")
    if not admin_token_env or x_admin_token != admin_token_env:
        raise HTTPException(status_code=404, detail="Trang không tồn tại.")
    met_path = MODELS_DIR / "metrics.json"
    if not met_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có metrics.json.")
    with open(met_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Phục vụ Static Files
@app.get("/")
def serve_home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "Chưa tìm thấy index.html"}, status_code=404)
    return FileResponse(index_file)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Khởi động Bếp Đà Nẵng tại http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
