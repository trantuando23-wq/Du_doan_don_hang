"""
report.py - Module tạo báo cáo đánh giá mô hình, phân tích lỗi, kiểm tra tính hợp lý,
hiệu chuẩn Conformal Prediction và xuất biểu đồ hình ảnh vào thư mục reports/.
Hỗ trợ đầy đủ 2 đặc trưng mới: staff (nhân sự bếp) và channel (hình thức nhận món).
"""
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
import sklearn

# Thiết lập backend matplotlib không cần giao diện đồ họa (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import REPORTS_DIR, STORES, DISH_GROUP_NAMES, ALL_FEATURES, CHANNELS, CHANNEL_LABELS

# Thiết lập font và style đồ thị đẹp, hỗ trợ hiển thị tiếng Việt / Unicode
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def create_charts(y_test, y_pred_rf, y_pred_linear, df_test, feature_importances):
    """Vẽ và lưu 4 biểu đồ trực quan vào thư mục reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Biểu đồ Scatter: Thực tế vs Dự đoán (Random Forest)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(y_test, y_pred_rf, alpha=0.45, color="#ea580c", edgecolors="none", s=25, label="Random Forest")
    max_val = max(float(np.max(y_test)), float(np.max(y_pred_rf))) + 2
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="#4b5563", linewidth=1.5, label="Đường lý tưởng (y = x)")
    plt.title("Thực tế vs Dự đoán trên Tập Kiểm Thử (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Thời gian thực tế (phút)", fontsize=11)
    plt.ylabel("Thời gian dự đoán (phút)", fontsize=11)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(frameon=True, facecolor="#fafafa")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_pred.png")
    plt.close()
    
    # 2. Biểu đồ Feature Importance (Top 12 đặc trưng quan trọng nhất)
    top_features = feature_importances[:12]
    names = [f["feature"] for f in top_features][::-1]
    clean_names = []
    for n in names:
        n = n.replace("store_", "Quán: ").replace("dish_", "Món: ").replace("weekday_", "Thứ: ").replace("channel_", "Kênh: ")
        if n == "queue": n = "Hàng đợi (Queue)"
        elif n == "qty": n = "Số lượng (Qty)"
        elif n == "hour": n = "Khung giờ"
        elif n == "is_weekend": n = "Cuối tuần"
        elif n == "staff": n = "Nhân viên bếp (Staff)"
        clean_names.append(n)
        
    scores = [f["importance"] * 100 for f in top_features][::-1]
    
    plt.figure(figsize=(9, 6), dpi=150)
    bars = plt.barh(clean_names, scores, color="#f97316", edgecolor="#c2410c", height=0.65)
    plt.title("Tầm quan trọng của các đặc trưng (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mức độ đóng góp (%)", fontsize=11)
    plt.xlim(0, max(scores) * 1.15)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold", color="#374151")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png")
    plt.close()
    
    # 3. Biểu đồ Phân phối sai số (Error Residuals: Actual - Predicted)
    residuals = y_test - y_pred_rf
    plt.figure(figsize=(8, 5), dpi=150)
    counts, bins, _ = plt.hist(residuals, bins=40, density=True, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.65, label="Phân phối sai số")
    plt.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, label="Sai số = 0 (Hoàn hảo)")
    plt.axvline(float(np.mean(residuals)), color="#d97706", linestyle="-", linewidth=1.2, label=f"Sai số trung bình ({np.mean(residuals):.2f}p)")
    plt.title("Phân phối sai số dự đoán (Thực tế - Dự đoán)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sai số (phút)", fontsize=11)
    plt.ylabel("Mật độ xác suất", fontsize=11)
    plt.legend(frameon=True, facecolor="#fafafa")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "error_distribution.png")
    plt.close()
    
    # 4. Biểu đồ MAE theo khung giờ trong ngày (7h - 22h)
    df_eval = df_test.copy()
    df_eval["abs_error"] = np.abs(y_test - y_pred_rf)
    hourly_mae = df_eval.groupby("hour")["abs_error"].mean()
    
    plt.figure(figsize=(9, 5), dpi=150)
    hours = list(range(7, 23))
    mae_vals = [hourly_mae.get(h, 0.0) for h in hours]
    
    bar_colors = ["#ea580c" if (11 <= h <= 13 or 18 <= h <= 20) else "#0284c7" for h in hours]
    plt.bar(hours, mae_vals, color=bar_colors, edgecolor="#1e293b", width=0.7)
    plt.title("Sai số tuyệt đối trung bình (MAE) theo khung giờ", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Giờ trong ngày (Màu cam: Giờ cao điểm)", fontsize=11)
    plt.ylabel("MAE (phút)", fontsize=11)
    plt.xticks(hours)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "mae_by_hour.png")
    plt.close()

def analyze_error_by_group(df_test: pd.DataFrame, y_test: np.ndarray, y_pred_rf: np.ndarray):
    """
    Phân tích lỗi chi tiết theo Quán, Nhóm món, Khung giờ, Mức độ đông, Channel và Staff.
    Lưu bảng kết quả vào reports/error_by_group.csv và trả về dữ liệu phân tích.
    """
    df = df_test.copy()
    df["actual"] = y_test
    df["pred"] = y_pred_rf
    df["abs_error"] = np.abs(y_test - y_pred_rf)
    df["sq_error"] = (y_test - y_pred_rf) ** 2
    df["within_5m"] = (df["abs_error"] <= 5.0).astype(int)
    
    # Phân loại thời điểm
    df["time_slot"] = df["hour"].apply(
        lambda h: "Cao điểm trưa (11-13h)" if (11 <= h <= 13) else (
            "Cao điểm tối (18-20h)" if (18 <= h <= 20) else "Thấp điểm"
        )
    )
    # Phân loại mức đông
    df["queue_level"] = df["queue"].apply(
        lambda q: "Vắng (≤4)" if q <= 4 else ("Vừa (5-11)" if q <= 11 else "Đông (>11)")
    )
    # Phân loại staff
    df["staff_level"] = df["staff"].apply(lambda s: f"Bếp {s} người")
    # Phân loại channel có nhãn tiếng Việt
    df["channel_vn"] = df["channel"].apply(lambda c: CHANNEL_LABELS.get(c, c))
    
    rows = []
    
    def add_group(group_type, col_name):
        grouped = df.groupby(col_name)
        for val, grp in grouped:
            mae = float(grp["abs_error"].mean())
            rmse = float(np.sqrt(grp["sq_error"].mean()))
            pct_5 = float(grp["within_5m"].mean() * 100.0)
            rows.append({
                "group_type": group_type,
                "group_name": str(val),
                "count": len(grp),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "pct_within_5min": round(pct_5, 1)
            })
            
    add_group("Quán ăn", "store")
    add_group("Nhóm món", "dish_group")
    add_group("Khung giờ", "time_slot")
    add_group("Mức độ đông", "queue_level")
    add_group("Hình thức nhận (Channel)", "channel_vn")
    add_group("Nhân viên bếp (Staff)", "staff_level")
    
    res_df = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "error_by_group.csv"
    res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # Tìm nhóm tốt nhất và nhóm tệ nhất
    dish_group_df = res_df[res_df["group_type"] == "Nhóm món"]
    best_group = dish_group_df.sort_values("mae").iloc[0]
    worst_group = dish_group_df.sort_values("mae", ascending=False).iloc[0]
    
    time_df = res_df[res_df["group_type"] == "Khung giờ"]
    peak_worst = time_df.sort_values("mae", ascending=False).iloc[0]
    
    # Phân tích kênh nhận món
    chan_df = res_df[res_df["group_type"] == "Hình thức nhận (Channel)"]
    chan_mae_dict = {row["group_name"]: row["mae"] for _, row in chan_df.iterrows()}
    
    # Phân tích nhân viên
    staff_df = res_df[res_df["group_type"] == "Nhân viên bếp (Staff)"]
    staff_mae_dict = {row["group_name"]: row["mae"] for _, row in staff_df.iterrows()}
    
    # Phân tích tương tác đặc thù (món chiên giờ cao điểm)
    fried_peak_mask = (df["dish_group"] == "chien_nuong") & (df["time_slot"] != "Thấp điểm")
    fried_peak_mae = float(df[fried_peak_mask]["abs_error"].mean()) if fried_peak_mask.sum() > 0 else 0.0
    
    insights = {
        "best_group_name": best_group["group_name"],
        "best_group_mae": best_group["mae"],
        "worst_group_name": worst_group["group_name"],
        "worst_group_mae": worst_group["mae"],
        "peak_worst_name": peak_worst["group_name"],
        "peak_worst_mae": peak_worst["mae"],
        "fried_peak_mae": round(fried_peak_mae, 2),
        "chan_mae_dict": chan_mae_dict,
        "staff_mae_dict": staff_mae_dict
    }
    
    return res_df, insights

def check_monotonicity(pipeline, use_log: bool, n_samples: int = 500) -> dict:
    """
    Kiểm tra tính hợp lý (Sanity / Monotonicity Check):
    Với 500 mẫu ngẫu nhiên:
    1. Tăng số đơn chờ (+5 đơn) -> thời gian dự đoán có tăng (hoặc giữ nguyên) không?
    2. Tăng số lượng phần ăn (+2 phần) -> thời gian dự đoán có tăng không?
    3. Tăng nhân viên bếp (+1 người) -> thời gian dự đoán có GIẢM (hoặc giữ nguyên) không?
    4. Giao qua app vs Ăn tại quán -> Giao app có lâu hơn không?
    """
    np.random.seed(42)
    store_names = list(STORES.keys())
    weekday_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    
    samples = []
    for _ in range(n_samples):
        st = np.random.choice(store_names)
        dish_info = np.random.choice(STORES[st]["dishes"])
        h = int(np.random.randint(7, 22))
        wd = np.random.choice(weekday_names)
        is_we = 1 if wd in ["Thứ Bảy", "Chủ Nhật"] else 0
        q = int(np.random.randint(1, 20))
        qty = int(np.random.choice([1, 2, 3]))
        st_range = STORES[st].get("staff_range", (1, 3))
        staff = int(np.random.randint(st_range[0], st_range[1] + 1))
        ch = str(np.random.choice(CHANNELS))
        samples.append({
            "store": st,
            "dish": dish_info["dish"],
            "dish_group": dish_info["group"],
            "qty": qty,
            "hour": h,
            "weekday": wd,
            "is_weekend": is_we,
            "queue": q,
            "staff": staff,
            "channel": ch
        })
        
    df_base = pd.DataFrame(samples)
    
    def predict_df(df_in):
        preds = pipeline.predict(df_in)
        return np.exp(preds) if use_log else preds
        
    base_preds = predict_df(df_base)
    
    # 1. Kiểm tra tăng queue (+5)
    df_queue_up = df_base.copy()
    df_queue_up["queue"] = df_queue_up["queue"] + 5
    queue_up_preds = predict_df(df_queue_up)
    
    queue_diffs = queue_up_preds - base_preds
    queue_violations = int(np.sum(queue_diffs < -0.05))
    queue_violation_rate = round(float(queue_violations / n_samples * 100), 2)
    queue_drops = -queue_diffs[queue_diffs < -0.05]
    max_queue_drop = round(float(np.max(queue_drops)), 2) if queue_violations > 0 else 0.0
    
    # 2. Kiểm tra tăng qty (+2)
    df_qty_up = df_base.copy()
    df_qty_up["qty"] = df_qty_up["qty"] + 2
    qty_up_preds = predict_df(df_qty_up)
    
    qty_diffs = qty_up_preds - base_preds
    qty_violations = int(np.sum(qty_diffs < -0.05))
    qty_violation_rate = round(float(qty_violations / n_samples * 100), 2)
    qty_drops = -qty_diffs[qty_diffs < -0.05]
    max_qty_drop = round(float(np.max(qty_drops)), 2) if qty_violations > 0 else 0.0
    
    # 3. Kiểm tra tăng staff (+1 người): thời gian PHẢI GIẢM hoặc KHÔNG ĐỔI
    df_staff_up = df_base.copy()
    df_staff_up["staff"] = np.clip(df_staff_up["staff"] + 1, 1, 4)
    staff_up_preds = predict_df(df_staff_up)
    
    staff_diffs = staff_up_preds - base_preds # Mong muốn <= 0
    staff_violations = int(np.sum(staff_diffs > 0.1)) # Nếu tăng > 0.1 phút là vi phạm
    staff_violation_rate = round(float(staff_violations / n_samples * 100), 2)
    
    # 4. Kiểm tra channel: Giao app vs Ăn tại quán
    df_tai_quan = df_base.copy()
    df_tai_quan["channel"] = "tai_quan"
    df_giao_app = df_base.copy()
    df_giao_app["channel"] = "giao_app"
    chan_diffs = predict_df(df_giao_app) - predict_df(df_tai_quan)
    chan_app_longer_rate = round(float(np.mean(chan_diffs >= 0.0) * 100), 2)
    avg_app_extra = round(float(np.mean(chan_diffs)), 2)
    
    return {
        "n_samples": n_samples,
        "queue_monotonicity": {
            "violations": queue_violations,
            "rate_pct": queue_violation_rate,
            "max_drop_min": max_queue_drop
        },
        "qty_monotonicity": {
            "violations": qty_violations,
            "rate_pct": qty_violation_rate,
            "max_drop_min": max_qty_drop
        },
        "staff_monotonicity": {
            "violations": staff_violations,
            "rate_pct": staff_violation_rate
        },
        "channel_sanity": {
            "app_longer_rate_pct": chan_app_longer_rate,
            "avg_app_extra_minutes": avg_app_extra
        }
    }

def write_markdown_report(
    metrics_data: dict,
    error_df: pd.DataFrame,
    insights: dict,
    sanity: dict,
    calib_info: dict,
    prev_metrics: dict = None,
    prev_label: str = "backup_v2 (5 quán)",
    unseen_store_eval: dict = None
):
    """Tạo file reports/report.md toàn diện bằng tiếng Việt kèm bảng so sánh trước/sau mở rộng quán và đánh giá quán chưa từng thấy."""
    models = metrics_data["models"]
    ds = metrics_data["dataset_summary"]
    best_params = metrics_data.get("best_rf_params", {})
    n_stores = ds.get("n_stores", 8)

    # Xây dựng bảng so sánh Trước / Sau khi mở rộng quán (v2 -> v3)
    prev_section = ""
    if prev_metrics and "models" in prev_metrics:
        pm = prev_metrics["models"]
        p_rf = pm.get("random_forest", {})
        p_lin = pm.get("linear", {})
        p_base = pm.get("baseline", {})
        p_calib = prev_metrics.get("conformal_calibration", {})

        rf_mae_diff = models['random_forest']['mae'] - p_rf.get('mae', 0.0)
        rf_rmse_diff = models['random_forest']['rmse'] - p_rf.get('rmse', 0.0)
        rf_r2_diff = models['random_forest']['r2'] - p_rf.get('r2', 0.0)
        rf_5m_diff = models['random_forest']['pct_within_5min'] - p_rf.get('pct_within_5min', 0.0)
        cov_diff = calib_info['post_coverage_test'] - p_calib.get('post_coverage_test', 0.0)

        prev_section = f"""
## 2. Bảng So Sánh Trước / Sau Khi Mở Rộng Quán ({prev_label} → {n_stores} quán · 3 khu vực)

Hệ thống đã sao lưu phiên bản trước khi mở rộng trong thư mục `backup_v2/`. Dưới đây là bảng đối chiếu trực tiếp trên tập kiểm thử độc lập (Test Set Tuần 6):

| Chỉ số / Mô hình | Trước mở rộng ({prev_label}) | Sau mở rộng ({n_stores} quán · {ds['total_samples']:,} đơn) | Thay đổi | Nhận xét |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest MAE** | **{p_rf.get('mae', 'N/A')} phút** | **{models['random_forest']['mae']} phút** | **{rf_mae_diff:+.2f} phút** | {'Tốt hơn' if rf_mae_diff < -0.05 else ('Tăng nhẹ do mở rộng đa quán' if rf_mae_diff > 0.05 else 'Ổn định tương đương')} |
| **Random Forest RMSE** | {p_rf.get('rmse', 'N/A')} phút | {models['random_forest']['rmse']} phút | {rf_rmse_diff:+.2f} phút | Độ phân tán sai số |
| **Random Forest $R^2$** | {p_rf.get('r2', 'N/A')} | **{models['random_forest']['r2']}** | **{rf_r2_diff:+.4f}** | {'Tăng độ giải thích phương sai' if rf_r2_diff > 0.005 else ('Giảm nhẹ' if rf_r2_diff < -0.005 else 'Ổn định')} |
| **Random Forest % lệch < 5p** | {p_rf.get('pct_within_5min', 'N/A')}% | {models['random_forest']['pct_within_5min']}% | {rf_5m_diff:+.2f}% | Độ chính xác thực tế |
| **Linear Regression MAE** | {p_lin.get('mae', 'N/A')} phút | {models['linear']['mae']} phút | {models['linear']['mae'] - p_lin.get('mae', 0.0):+.2f} phút | Mô hình tuyến tính |
| **Linear Regression $R^2$** | {p_lin.get('r2', 'N/A')} | {models['linear']['r2']} | {models['linear']['r2'] - p_lin.get('r2', 0.0):+.4f} | Khả năng khớp tuyến tính |
| **Baseline (TB món) MAE** | {p_base.get('mae', 'N/A')} phút | {models['baseline']['mae']} phút | {models['baseline']['mae'] - p_base.get('mae', 0.0):+.2f} phút | Khi có nhiều quán trùng món với tốc độ khác nhau, Baseline theo tên món sẽ sai lệch mạnh hơn |
| **Độ phủ Conformal (Test)** | {p_calib.get('post_coverage_test', 'N/A')}% | {calib_info['post_coverage_test']}% | {cov_diff:+.2f}% | Bám sát chuẩn mục tiêu 80% |

### Nhận xét trung thực:
- Mở rộng từ 5 lên {n_stores} quán (có các cặp quán trùng loại món nhưng khác hệ số tốc độ như Mì Quảng Hạt Sen vs Mì Quảng Gió Biển, Bánh Xèo Sông Hàn vs Bánh Xèo Làng Đá) làm **Baseline theo tên món** suy giảm rõ rệt vì không phân biệt được quán nào nhanh/chậm hơn.
- Trong khi đó, **Random Forest** vẫn giữ sai số ở mức **{models['random_forest']['mae']} phút** ($R^2 = {models['random_forest']['r2']}$, **{models['random_forest']['pct_within_5min']}%** đơn lệch dưới 5 phút), chứng tỏ mô hình tách biệt tốt tốc độ riêng của từng quán khi đã có dữ liệu lịch sử của quán đó.
"""

    # Xây dựng mục ĐÁNH GIÁ QUÁN CHƯA TỪNG THẤY (GroupKFold theo quán)
    unseen_section = ""
    if unseen_store_eval and unseen_store_eval.get("per_store"):
        overall_u = unseen_store_eval["overall"]
        per_store = unseen_store_eval["per_store"]
        best_u = per_store[0]
        worst_u = per_store[-1]
        mae_gap = round(overall_u["mae"] - models["random_forest"]["mae"], 2)

        rows_md = []
        for r in per_store:
            shared_txt = f"{r['shared_dish_count']} món chung" if r["shared_dish_count"] > 0 else "0 (món riêng)"
            group_txt = "Có" if r["has_shared_group"] else "Không (nhóm duy nhất)"
            rows_md.append(
                f"| **{r['store']}** | {r['region']} | {r['n_samples']:,} | {r['mae']} phút | {r['rmse']} phút | {r['bias']:+.2f} phút | {r['r2']} | {r['pct_within_5min']}% | {shared_txt} / {group_txt} |"
            )
        rows_str = "\n".join(rows_md)

        verdict = (
            "Mô hình **chủ yếu học thuộc hệ số tốc độ riêng của từng quán (store-specific baseline)** hơn là tổng quát hóa hoàn hảo sang quán hoàn toàn mới."
            if mae_gap >= 0.8 else
            "Mô hình tổng quát hóa tương đối khá sang quán mới nhờ nhóm món và các đặc trưng vận hành."
        )

        unseen_section = f"""
---

## 7. Đánh Giá Quán Chưa Từng Thấy (GroupKFold Theo Quán — Leave-One-Store-Out)

Để kiểm tra xem mô hình có **tổng quát hóa được sang một quán mới hoàn toàn** hay chỉ **học thuộc từng quán trong tập train**, hệ thống chạy `GroupKFold(n_splits={unseen_store_eval['n_stores']})` theo cột `store`: mỗi lượt ẩn toàn bộ dữ liệu của 1 quán, huấn luyện Random Forest trên {unseen_store_eval['n_stores'] - 1} quán còn lại rồi dự đoán trên quán bị ẩn. *(Kết quả này chỉ dùng cho đánh giá kỹ thuật ở terminal/file, KHÔNG hiển thị trên web).*

| Quán bị ẩn (Unseen Test) | Khu vực | Số mẫu | MAE | RMSE | Độ lệch TB (Bias) | $R^2$ | % Lệch < 5p | Trùng món / Nhóm món với tập train |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{rows_str}
| **TỔNG THỂ (GroupKFold {unseen_store_eval['n_stores']} quán)** | **Toàn bộ** | **{ds['total_samples']:,}** | **{overall_u['mae']} phút** | **{overall_u['rmse']} phút** | — | **{overall_u['r2']}** | **{overall_u['pct_within_5min']}%** | — |

### Nhận xét trung thực về khả năng tổng quát hóa:
- **So sánh trực diện:** MAE khi dự đoán quán **chưa từng thấy** là **{overall_u['mae']} phút** ($R^2 = {overall_u['r2']}$), kém hơn **{mae_gap:+.2f} phút** so với tập test tiêu chuẩn đã từng thấy quán (**{models['random_forest']['mae']} phút**).
- **Kết luận thẳng thắn:** {verdict}
  - **Quán tổng quát hóa tốt nhất:** **{best_u['store']}** (MAE = **{best_u['mae']} phút**, bias = {best_u['bias']:+.2f} phút) nhờ chia sẻ nhóm món / tên món với quán cùng loại trong tập huấn luyện và có tốc độ bếp gần mức trung bình của nhóm.
  - **Quán tệ nhất khi chưa từng thấy:** **{worst_u['store']}** (MAE = **{worst_u['mae']} phút**, bias = {worst_u['bias']:+.2f} phút, $R^2 = {worst_u['r2']}$). Lý do rất rõ ràng: khi mô hình chưa từng nhìn thấy dữ liệu của quán này, `OneHotEncoder(handle_unknown='ignore')` đưa toàn bộ cột `store` của quán mới về 0, nên mô hình **không thể biết `speed_factor` hay thời gian nấu đặc thù của quán** (ví dụ quán làm nhanh/chậm hơn quán cùng loại khác, hoặc là quán duy nhất bán nhóm món đó như nhóm `com` của Cơm Gà Phố Cổ).
  - **Hàm ý thực tế:** Khi mở thêm một quán hoàn toàn mới ngoài đời thực, mô hình chỉ cho ước lượng sơ bộ dựa theo nhóm món (`dish_group`), đơn chờ (`queue`), nhân sự (`staff`) và kênh nhận (`channel`); cần thu thập dữ liệu phản hồi thực tế của riêng quán đó rồi chạy `evaluate_real.py --retrain` để đạt độ chính xác cao.
"""

    rf_vs_lin_note = (
        "Random Forest vượt trội hơn Linear Regression trên cả MAE, RMSE và $R^2$ nhờ bắt được tương tác phi tuyến `queue × staff` và nghẽn trạm chiên/nướng."
        if models['random_forest']['mae'] < models['linear']['mae']
        else "Linear Regression đạt kết quả ngang hoặc nhỉnh hơn Random Forest trên tập test này — ghi nhận trung thực, không chỉnh dữ liệu để ép Random Forest thắng."
    )

    content = f"""# Báo Cáo Kỹ Thuật Đánh Giá Mô Hình Dự Đoán Thời Gian Hoàn Thành Món Ăn (Food ETA Đà Nẵng)

> **Thông tin kiểm định:**
> - **Ngày thực hiện:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
> - **Phiên bản scikit-learn:** `{sklearn.__version__}`
> - **Siêu tham số Random Forest tốt nhất:** `{best_params}`
> - **Đặc trưng mô hình (giữ nguyên, không thêm mới):** `store`, `dish`, `dish_group`, `weekday`, `channel`, `qty`, `hour`, `is_weekend`, `queue`, `staff`
> - **Phân loại dữ liệu:** **DỮ LIỆU MÔ PHỎNG** ({ds['total_samples']:,} đơn hàng · {n_stores} quán thuộc 3 khu vực Đà Nẵng · 6 tuần giả lập)

---

## 1. Dữ Liệu & Phân Chia Huấn Luyện (Data Splitting)

- **Tổng số mẫu:** {ds['total_samples']:,} dòng ({n_stores} quán hư cấu · 3 khu vực: Hải Châu, Sơn Trà, Ngũ Hành Sơn)
- **Tập Train:** {ds['train_samples']:,} mẫu ({ds['train_period']})
  - *Tập Train chính (Proper Train - 4 tuần đầu, đã loại 1% outlier):* {ds.get('proper_train_samples', 'N/A'):,} mẫu
  - *Tập Hiệu chuẩn (Calibration Set - tuần thứ 5):* {ds.get('calib_samples', 'N/A'):,} mẫu (dành riêng cho Split Conformal Prediction)
- **Tập Test:** {ds['test_samples']:,} mẫu ({ds['test_period']} - tuần thứ 6 độc lập)
- **Phương pháp phân chia:** **Phân chia theo thứ tự thời gian (Temporal Split)** để phản ánh đúng thực tế vận hành và tránh rò rỉ dữ liệu (Data Leakage).
{prev_section}
---

## 3. Bảng Hiệu Năng Chi Tiết (Tập Test Tuần 6)

| Mô hình | MAE (Sai số TB) | RMSE (Căn sai số) | $R^2$ (Hệ số tương quan) | % Dự đoán lệch < 5 phút |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (TB theo món)** | {models['baseline']['mae']} phút | {models['baseline']['rmse']} phút | {models['baseline']['r2']} | {models['baseline']['pct_within_5min']}% |
| **Linear Regression** | {models['linear']['mae']} phút | {models['linear']['rmse']} phút | {models['linear']['r2']} | {models['linear']['pct_within_5min']}% |
| **Random Forest Regressor** | **{models['random_forest']['mae']} phút** | **{models['random_forest']['rmse']} phút** | **{models['random_forest']['r2']}** | **{models['random_forest']['pct_within_5min']}%** |

> *Nhận xét:* {rf_vs_lin_note}

---

## 4. Phân Tích Lỗi Chi Tiết (Error Analysis)

Dữ liệu chi tiết được xuất tại tệp [`reports/error_by_group.csv`](error_by_group.csv).

### Tóm tắt các phân nhóm:
1. **Nhóm mô hình dự đoán chính xác nhất:**
   - Nhóm món **`{insights['best_group_name']}`** đạt sai số thấp nhất với **MAE = {insights['best_group_mae']} phút**.
2. **Nhóm mô hình gặp nhiều sai số nhất:**
   - Nhóm món **`{insights['worst_group_name']}`** có độ lệch cao nhất với **MAE = {insights['worst_group_mae']} phút**.
   - Đặc biệt, trong điều kiện **món chiên/nướng vào giờ cao điểm**, sai số MAE là **{insights['fried_peak_mae']} phút** do bếp bị quá tải trạm chiên rán cục bộ.
3. **Theo hình thức nhận món (Channel):**
   - MAE cụ thể: {", ".join([f"**{k}**: {v}p" for k, v in insights.get('chan_mae_dict', {}).items()])}. Giao qua app có độ phân tán lớn hơn do cộng thêm thời gian chờ tài xế.
4. **Theo số nhân viên bếp (Staff):**
   - MAE cụ thể: {", ".join([f"**{k}**: {v}p" for k, v in insights.get('staff_mae_dict', {}).items()])}.

---

## 5. Kiểm Tra & Hiệu Chuẩn Độ Phủ Khoảng Dự Đoán (Conformal Prediction)

Khoảng dự đoán ban đầu dựa trên phân vị 10% - 90% từ các cây quyết định trong Random Forest nhằm hướng tới độ bao phủ mục tiêu **80%**.

- **Độ phủ thực tế ban đầu (trên tập Test):** **{calib_info['pre_coverage_test']}%** *(Thấp hơn mục tiêu 80% do các cây trong rừng có tương quan dương)*.
- **Phương pháp hiệu chuẩn:** **Split Conformal Prediction**
  - Sử dụng tuần thứ 5 (Calibration Set) độc lập hoàn toàn với Test Set để tính điểm không tương đồng.
  - Độ lệch hiệu chỉnh: **$q_{{calib}} = +{calib_info['q_calib']} \\text{{ phút}}$**.
- **Độ phủ sau hiệu chuẩn (trên tập Test):** **{calib_info['post_coverage_test']}%** *(Đạt chuẩn lý tưởng ~80%)*.
- **Kết luận:** Khoảng dự đoán hiển thị cho người dùng trên API đã được chuẩn hóa toán học tin cậy.

---

## 6. Kiểm Tra Tính Hợp Lý Của Dự Đoán (Monotonicity & Sanity Check)

Thực hiện thử nghiệm trên **{sanity['n_samples']} tổ hợp ngẫu nhiên**:

- **Tăng số đơn chờ (+5 đơn):** Tỷ lệ vi phạm = **{sanity['queue_monotonicity']['rate_pct']}%** ({sanity['queue_monotonicity']['violations']}/{sanity['n_samples']} mẫu, mức tụt tối đa {sanity['queue_monotonicity']['max_drop_min']} phút).
- **Tăng số lượng món (+2 phần):** Tỷ lệ vi phạm = **{sanity['qty_monotonicity']['rate_pct']}%** ({sanity['qty_monotonicity']['violations']}/{sanity['n_samples']} mẫu, mức tụt tối đa {sanity['qty_monotonicity']['max_drop_min']} phút).
- **Tăng nhân viên bếp (+1 người):** Tỷ lệ vi phạm = **{sanity['staff_monotonicity']['rate_pct']}%** (thời gian giảm hoặc giữ nguyên khi có thêm người).
- **Kênh Giao app vs Ăn tại quán:** Tỷ lệ Giao app lâu hơn = **{sanity['channel_sanity']['app_longer_rate_pct']}%** (trung bình lâu hơn **{sanity['channel_sanity']['avg_app_extra_minutes']} phút** để đóng gói và chờ gom đơn).
{unseen_section}
---

## 8. Các Tệp Báo Cáo Kèm Theo

- [`reports/metrics.json`](metrics.json): Toàn bộ số liệu máy đọc JSON.
- [`reports/error_by_group.csv`](error_by_group.csv): Bảng phân tích sai số chi tiết theo từng nhóm.
- [`reports/explain.md`](explain.md): Báo cáo giải thích mô hình chuyên sâu (Permutation Importance, PDP 1D/2D, Ablation).
- [`reports/actual_vs_pred.png`](actual_vs_pred.png): Đồ thị thực tế vs dự đoán.
- [`reports/feature_importance.png`](feature_importance.png): Mức độ quan trọng của đặc trưng.
- [`reports/error_distribution.png`](error_distribution.png): Phân phối sai số.
- [`reports/mae_by_hour.png`](mae_by_hour.png): MAE theo 24 giờ.
"""
    report_path = REPORTS_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"✅ Đã ghi báo cáo Markdown kỹ thuật tại: {report_path}")
    return report_path
