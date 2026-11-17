"""
generate_data.py - Sinh 15.000 dòng dữ liệu mô phỏng thời gian hoàn thành món ăn tại Đà Nẵng
trải dài trên 3 khu vực (8 quán hư cấu, 6 tuần giả lập, ~1.500–2.100 đơn/quán).
Giữ nguyên toàn bộ quy luật vận hành: giờ cao điểm, hàng đợi (queue), nhân sự bếp (staff),
hình thức nhận món (channel: tai_quan, mang_di, giao_app), tương tác phi tuyến,
nhiễu lognormal lệch phải và 1.5% outlier.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Thiết lập đường dẫn import config
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    STORES, STORE_TO_REGION, DATA_DIR, CHANNELS, CHANNEL_EFFECTS,
    STAFF_INTERACTION_BASE, STAFF_INTERACTION_SLOPE
)


def generate_orders(n_samples: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Sinh n_samples đơn hàng trải dài trên 6 tuần giả lập cho 8 quán thuộc 3 khu vực."""
    np.random.seed(seed)

    # Mốc thời gian 6 tuần (42 ngày): từ 2026-09-01 07:00 đến 2026-10-12 22:00
    start_time = datetime(2026, 9, 1, 7, 0, 0)

    # 1. Trọng số giờ hoạt động (7h - 22h), cao điểm trưa 11-13h & tối 18-20h
    hour_weights = {
        7: 0.04, 8: 0.05, 9: 0.04, 10: 0.05,
        11: 0.12, 12: 0.14, 13: 0.08,  # Trưa cao điểm
        14: 0.03, 15: 0.04, 16: 0.05, 17: 0.07,
        18: 0.13, 19: 0.11, 20: 0.05,  # Tối cao điểm
        21: 0.03, 22: 0.02
    }
    hours_pool = list(hour_weights.keys())
    probs_pool = np.array(list(hour_weights.values()))
    probs_pool = probs_pool / probs_pool.sum()

    days_offset = np.random.randint(0, 42, size=n_samples)
    chosen_hours = np.random.choice(hours_pool, size=n_samples, p=probs_pool)
    chosen_minutes = np.random.randint(0, 60, size=n_samples)
    chosen_seconds = np.random.randint(0, 60, size=n_samples)

    timestamps = [
        start_time + timedelta(days=int(d), hours=int(h) - 7, minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days_offset, chosen_hours, chosen_minutes, chosen_seconds)
    ]

    store_names = list(STORES.keys())
    # Phân bổ tương đối đồng đều giữa 8 quán (~1.600 - 2.100 dòng/quán)
    base_weights = np.array([1.10 if len(STORES[s]["dishes"]) >= 5 else 0.95 for s in store_names])
    store_probs = base_weights / base_weights.sum()

    records = []
    weekday_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

    for i in range(n_samples):
        ts = timestamps[i]
        hour = ts.hour
        weekday_idx = ts.weekday()
        weekday = weekday_names[weekday_idx]
        is_weekend = 1 if weekday_idx >= 5 else 0

        # Chọn quán
        store_name = str(np.random.choice(store_names, p=store_probs))
        store_info = STORES[store_name]

        # Chọn món ăn từ thực đơn của quán
        dish_item = np.random.choice(store_info["dishes"])
        dish = dish_item["dish"]
        dish_group = dish_item["group"]
        base_time = dish_item["base_time"]

        # Số lượng phần ăn: đa số đặt 1-2 phần, ít hơn đặt 3-5 phần
        qty = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.26, 0.12, 0.05, 0.02]))

        # Khung giờ cao điểm: Trưa 11-13h, Tối 18-20h
        is_lunch_peak = 1 if (11 <= hour <= 13) else 0
        is_dinner_peak = 1 if (18 <= hour <= 20) else 0
        is_peak = is_lunch_peak or is_dinner_peak

        # Hàng đợi (queue): phụ thuộc vào khung giờ, cuối tuần và đặc thù khu vực/quán
        region_busy_mult = 1.08 if store_info["region_id"] == "hai_chau" else (0.95 if store_info["region_id"] == "ngu_hanh_son" else 1.0)
        if is_peak:
            lam = (14.0 if is_weekend else 11.0) * region_busy_mult
        elif 9 <= hour <= 21:
            lam = (6.5 if is_weekend else 5.0) * region_busy_mult
        else:
            lam = 2.5 * region_busy_mult
        queue = int(np.clip(np.random.poisson(lam=lam), 0, 35))

        # --- STAFF (Số nhân viên bếp làm ca: 1-4 người) ---
        min_s, max_s = store_info.get("staff_range", (1, 3))
        if is_peak:
            if max_s - min_s == 1:
                staff = int(np.random.choice([min_s, max_s], p=[0.25, 0.75]))
            elif max_s - min_s >= 2:
                mid_s = min_s + 1
                staff = int(np.random.choice([min_s, mid_s, max_s], p=[0.20, 0.35, 0.45]))
            else:
                staff = min_s
        else:
            if max_s - min_s == 1:
                staff = int(np.random.choice([min_s, max_s], p=[0.75, 0.25]))
            elif max_s - min_s >= 2:
                mid_s = min_s + 1
                staff = int(np.random.choice([min_s, mid_s, max_s], p=[0.60, 0.30, 0.10]))
            else:
                staff = min_s
        staff = int(np.clip(staff, 1, 4))

        # --- CHANNEL (Hình thức nhận món) ---
        if dish_group == "do_uong":
            c_probs = [0.35, 0.45, 0.20]  # quán trà/cà phê: mang đi cao
        elif is_dinner_peak:
            c_probs = [0.35, 0.25, 0.40]  # tối giao app tăng mạnh
        elif is_lunch_peak:
            c_probs = [0.50, 0.30, 0.20]  # trưa ăn tại quán đông
        else:
            c_probs = [0.55, 0.30, 0.15]
        channel = str(np.random.choice(CHANNELS, p=c_probs))

        # --- TÍNH TOÁN THỜI GIAN HOÀN THÀNH (MINUTES) ---
        # 1. Thời gian chuẩn bị cơ sở kèm hệ số làm gộp món (batch prep)
        batch_factor = 1.0 + 0.35 * (qty - 1)
        dish_prep = base_time * batch_factor * store_info["speed_factor"]

        # 2. Tương tác Queue × Staff
        staff_divisor = STAFF_INTERACTION_BASE + STAFF_INTERACTION_SLOPE * staff
        queue_effect = (queue * 0.42) / staff_divisor

        # 3. Phụ trội giờ cao điểm
        peak_delay = 2.0 if is_peak else 0.0

        # 4. Tương tác phi tuyến trạm chiên/nướng và cơm khi đông
        non_linear_delay = 0.0
        if dish_group == "chien_nuong" and is_peak and queue > 8:
            non_linear_delay = (((queue - 8) ** 1.25) * 0.45) / (0.7 + 0.3 * staff)
        elif dish_group == "com" and queue > 15:
            non_linear_delay = ((queue - 15) * 0.3) / (0.7 + 0.3 * staff)

        # 5. Phụ phí đóng gói / giao app theo Channel
        if channel == "mang_di":
            channel_addition = np.random.uniform(1.0, 2.0)
        elif channel == "giao_app":
            channel_addition = np.random.uniform(2.0, 4.0)
        else:
            channel_addition = 0.0

        # 6. Nhiễu lognormal lệch phải
        channel_noise_scale = CHANNEL_EFFECTS[channel]["noise_scale"]
        sigma = store_info["variance"] * channel_noise_scale
        log_noise = np.random.lognormal(mean=0.0, sigma=sigma)

        minutes = (dish_prep + queue_effect + peak_delay + non_linear_delay + channel_addition) * log_noise

        # 7. Outlier ngẫu nhiên (1.5% trường hợp)
        if np.random.rand() < 0.015:
            minutes += np.random.uniform(14.0, 26.0)

        minutes = max(2.0, round(minutes, 1))

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "store": store_name,
            "dish": dish,
            "dish_group": dish_group,
            "qty": qty,
            "hour": hour,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "queue": queue,
            "staff": staff,
            "channel": channel,
            "minutes": minutes
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    print("=" * 72)
    print("🚀 BƯỚC 1: SINH 15.000 ĐƠN HÀNG MÔ PHỎNG (3 KHU VỰC · 8 QUÁN)")
    print("=" * 72)

    df = generate_orders(n_samples=15000, seed=42)
    output_path = DATA_DIR / "orders.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ Đã lưu tập dữ liệu tại: {output_path}")
    print(f"📊 Kích thước: {df.shape[0]:,} dòng x {df.shape[1]} cột")
    print(f"📅 Khoảng thời gian: {df['timestamp'].min()} đến {df['timestamp'].max()}")

    df_tmp = df.copy()
    df_tmp["region"] = df_tmp["store"].map(STORE_TO_REGION)

    print("\n--- THỐNG KÊ SỐ DÒNG THEO KHU VỰC ---")
    print(df_tmp.groupby("region")["minutes"].agg(so_don="count", tg_tb="mean", tg_std="std").round(2))

    print("\n--- THỐNG KÊ SỐ DÒNG THEO TỪNG QUÁN ---")
    print(df_tmp.groupby(["region", "store"])["minutes"].agg(so_don="count", tg_tb="mean", tg_std="std").round(2))

    print("\n--- THỐNG KÊ BIẾN MỤC TIÊU (MINUTES) ---")
    print(df["minutes"].describe().round(2))

    print("\n--- THỐNG KÊ STAFF & CHANNEL ---")
    print("Staff:", df["staff"].value_counts().sort_index().to_dict())
    print(df.groupby("channel")["minutes"].agg(["count", "mean", "std"]).round(2))
    print("=" * 72)


if __name__ == "__main__":
    main()
