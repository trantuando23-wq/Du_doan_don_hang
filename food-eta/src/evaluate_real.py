"""
evaluate_real.py - Script đánh giá mô hình trên dữ liệu thực tế (từ data/feedback.csv hoặc real_orders.csv).
Hỗ trợ cờ --retrain để huấn luyện lại kết hợp dữ liệu mô phỏng và dữ liệu thật (trọng số 5x),
tự động sao lưu và bảo vệ mô hình cũ nếu mô hình mới không tốt hơn.
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import shutil
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_DIR, MODELS_DIR, REPORTS_DIR, BACKUP_MODELS_DIR,
    STORES, ALL_FEATURES, TARGET_COL
)

def evaluate_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0
    pct_5min = float(np.mean(np.abs(y_true - y_pred) <= 5.0) * 100.0)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "pct_within_5min": round(pct_5min, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên dữ liệu thực tế và huấn luyện lại.")
    parser.add_argument("--file", "-f", type=str, default=None, help="Đường dẫn tới file CSV dữ liệu thật (mặc định data/feedback.csv)")
    parser.add_argument("--retrain", action="store_true", help="Huấn luyện lại mô hình kết hợp dữ liệu thật (trọng số 5x)")
    args = parser.parse_args()

    print("=" * 70)
    print("🔍 ĐÁNH GIÁ MÔ HÌNH TRÊN DỮ LIỆU THỰC TẾ (REAL-WORLD EVALUATION)")
    print("=" * 70)

    # 1. Xác định file dữ liệu
    target_file = None
    if args.file:
        target_file = Path(args.file)
    else:
        # Thử tìm theo thứ tự: data/feedback.csv -> data/real_orders.csv -> data/real_orders_template.csv
        candidates = [
            DATA_DIR / "feedback.csv",
            DATA_DIR / "real_orders.csv",
            DATA_DIR / "real_orders_template.csv"
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 50:
                target_file = c
                break

    if target_file is None or not target_file.exists():
        print("❌ Chưa tìm thấy dữ liệu thực tế!")
        print(f"👉 Vui lòng ghi nhận phản hồi trên web hoặc điền vào file mẫu: {DATA_DIR / 'real_orders_template.csv'}")
        return

    print(f"📂 Đang đọc dữ liệu từ: {target_file}")
    df_real = pd.read_csv(target_file)

    # Chuẩn hóa tên cột
    # feedback.csv: timestamp, store, dish, qty, hour, weekday, queue_estimated, predicted_minutes, actual_minutes, is_demo
    if "actual_minutes" in df_real.columns:
        actual_col = "actual_minutes"
    elif "minutes" in df_real.columns:
        actual_col = "minutes"
    else:
        print("❌ File không có cột 'actual_minutes' hoặc 'minutes' làm nhãn thực tế!")
        return

    # Lọc bỏ các dòng demo nếu có cột is_demo
    if "is_demo" in df_real.columns:
        df_real = df_real[df_real["is_demo"] == 0].copy()

    n_samples = len(df_real)
    print(f"📊 Số lượng đơn hàng thực tế: {n_samples} đơn")

    if n_samples < 30:
        print(f"\n⚠️  CẢNH BÁO: Tập dữ liệu chỉ có {n_samples} dòng (< 30 dòng).")
        print("   Chưa đủ dữ liệu để đưa ra kết luận thống kê tin cậy, nhưng vẫn tiến hành đánh giá sơ bộ.\n")

    # Bổ sung các cột phụ nếu thiếu (dish_group, is_weekend, queue, staff, channel)
    if "queue" not in df_real.columns and "queue_estimated" in df_real.columns:
        df_real["queue"] = df_real["queue_estimated"]
    elif "queue" not in df_real.columns:
        df_real["queue"] = 5

    if "staff" not in df_real.columns:
        df_real["staff"] = 2

    if "channel" not in df_real.columns:
        df_real["channel"] = "tai_quan"

    if "is_weekend" not in df_real.columns:
        df_real["is_weekend"] = df_real["weekday"].apply(lambda w: 1 if w in ["Thứ Bảy", "Chủ Nhật"] else 0)

    # Map dish_group từ config
    dish_to_group = {}
    for s_info in STORES.values():
        for d in s_info["dishes"]:
            dish_to_group[d["dish"]] = d["group"]
    if "dish_group" not in df_real.columns:
        df_real["dish_group"] = df_real["dish"].map(dish_to_group).fillna("nuoc_dung")

    # 2. Nạp mô hình hiện tại
    rf_path = MODELS_DIR / "rf.joblib"
    calib_path = MODELS_DIR / "calibration.json"
    if not rf_path.exists():
        print(f"❌ Chưa có file mô hình {rf_path}. Vui lòng chạy python src/train.py trước!")
        return

    rf_bundle = joblib.load(rf_path)
    pipeline = rf_bundle["pipeline"]
    use_log = rf_bundle.get("use_log_transform", True)

    q_calib = 0.64
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
            q_calib = calib_data.get("q_calib", 0.64)

    # 3. Dự đoán với mô hình hiện tại
    X_real = df_real[ALL_FEATURES]
    y_true = df_real[actual_col].values

    rf_preds_raw = pipeline.predict(X_real)
    y_pred = np.maximum(2.0, np.exp(rf_preds_raw) if use_log else rf_preds_raw)

    # Tính khoảng dự đoán hiệu chuẩn
    rf_reg = pipeline.named_steps["regressor"]
    rf_prep = pipeline.named_steps["preprocessor"]
    X_trans = rf_prep.transform(X_real)
    tree_preds = np.array([tree.predict(X_trans) for tree in rf_reg.estimators_])
    if use_log:
        tree_preds = np.exp(tree_preds)
    L_raw = np.percentile(tree_preds, 10, axis=0)
    U_raw = np.percentile(tree_preds, 90, axis=0)
    L_calib = np.maximum(2.0, L_raw - q_calib)
    U_calib = U_raw + q_calib

    # Độ phủ
    coverage = float(np.mean((y_true >= L_calib) & (y_true <= U_calib)) * 100.0)
    real_metrics = evaluate_metrics(y_true, y_pred)

    # Đọc kết quả mô phỏng cũ để so sánh
    metrics_path = MODELS_DIR / "metrics.json"
    sim_metrics = {"mae": "N/A", "pct_within_5min": "N/A"}
    sim_cov = "N/A"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            sim_metrics = m["models"]["random_forest"]
            sim_cov = m["conformal_calibration"]["post_coverage_test"]

    # Độ lệch hệ thống (Bias)
    mean_diff = float(np.mean(y_true - y_pred))
    bias_desc = (
        f"Mô hình đang đoán NHANH HƠN thực tế trung bình {abs(mean_diff):.2f} phút (quán làm lâu hơn mô hình nghĩ)."
        if mean_diff > 0.5 else (
            f"Mô hình đang đoán CHẬM HƠN thực tế trung bình {abs(mean_diff):.2f} phút (quán làm nhanh hơn mô hình nghĩ)."
            if mean_diff < -0.5 else
            f"Mô hình không có độ lệch hệ thống đáng kể (độ lệch trung bình chỉ {abs(mean_diff):.2f} phút)."
        )
    )

    # IN KẾT QUẢ RA TERMINAL
    print("📋 BẢNG SO SÁNH HIỆU NĂNG: DỮ LIỆU THẬT VS DỮ LIỆU MÔ PHỎNG")
    print("-" * 70)
    comp_df = pd.DataFrame([
        {"Chỉ số": "Số lượng mẫu kiểm tra", "Dữ liệu mô phỏng (Test)": "1,639 đơn", "Dữ liệu THỰC TẾ": f"{n_samples} đơn"},
        {"Chỉ số": "MAE (Sai số tuyệt đối TB)", "Dữ liệu mô phỏng (Test)": f"{sim_metrics.get('mae')} phút", "Dữ liệu THỰC TẾ": f"{real_metrics['mae']} phút"},
        {"Chỉ số": "RMSE (Căn sai số bậc hai)", "Dữ liệu mô phỏng (Test)": f"{sim_metrics.get('rmse')} phút", "Dữ liệu THỰC TẾ": f"{real_metrics['rmse']} phút"},
        {"Chỉ số": "Tỷ lệ dự đoán lệch < 5 phút", "Dữ liệu mô phỏng (Test)": f"{sim_metrics.get('pct_within_5min')}%", "Dữ liệu THỰC TẾ": f"{real_metrics['pct_within_5min']}%"},
        {"Chỉ số": "Độ phủ khoảng dự đoán (80%)", "Dữ liệu mô phỏng (Test)": f"{sim_cov}%", "Dữ liệu THỰC TẾ": f"{coverage:.2f}%"}
    ])
    print(comp_df.to_string(index=False))
    print("\n🔍 NHẬN XÉT TRUNG THỰC:")
    print(f"   👉 {bias_desc}")

    # Ghi báo cáo ra reports/real_report.md
    real_report_md = f"""# Báo Cáo Đánh Giá Mô Hình Trên Dữ Liệu Thực Tế (Real-World Report)

> **Thời điểm xuất:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  
> **Nguồn dữ liệu:** `{target_file}` ({n_samples} đơn hàng thực tế)

---

## 1. Bảng So Sánh Với Dữ Liệu Mô Phỏng

| Chỉ số | Dữ liệu Mô phỏng (Test Set) | Dữ liệu Thực tế | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Số lượng mẫu** | 1,639 đơn | {n_samples} đơn | {'⚠️ Cần thêm dữ liệu (chưa đủ 30 đơn)' if n_samples < 30 else '✅ Đủ độ tin cậy'} |
| **MAE (phút)** | {sim_metrics.get('mae')} | **{real_metrics['mae']}** | Sai số thực tế |
| **RMSE (phút)** | {sim_metrics.get('rmse')} | **{real_metrics['rmse']}** | Mức độ nhạy cảm outlier |
| **Tỷ lệ lệch < 5 phút** | {sim_metrics.get('pct_within_5min')}% | **{real_metrics['pct_within_5min']}%** | Độ tin cậy thường nhật |
| **Độ phủ khoảng (Target 80%)** | {sim_cov}% | **{coverage:.2f}%** | Khả năng bao trùm thực tế |

---

## 2. Phân Tích Độ Lệch Hệ Thống (Systematic Bias)

- **Độ lệch trung bình (Thực tế - Dự đoán):** `{mean_diff:+.2f} phút`
- **Nhận xét chuyên môn:** {bias_desc}

---

## 3. Hướng Xử Lý Tiếp Theo
- Tiếp tục thu thập thêm đơn thực tế qua tính năng **"Báo thời gian thực tế"** trên website.
- Khi đạt trên 50 đơn hàng thực tế, chạy `python src/evaluate_real.py --retrain` để cập nhật trọng số cho mô hình.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "real_report.md", "w", encoding="utf-8") as f:
        f.write(real_report_md.strip() + "\n")
    print(f"\n📄 Đã lưu báo cáo chi tiết tại: {REPORTS_DIR / 'real_report.md'}")

    # 4. TÙY CHỌN HUẤN LUYỆN LẠI (--retrain)
    if args.retrain:
        print("\n" + "=" * 70)
        print("🔄 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN LẠI (RETRAIN VỚI DỮ LIỆU THỰC TẾ)")
        print("=" * 70)

        if n_samples < 10:
            print(f"❌ Dữ liệu thật quá ít ({n_samples} mẫu < 10 mẫu). Không thể chia tập train/test thực tế để retrain an toàn!")
            return

        # Chia dữ liệu thật: 70% train, 30% test độc lập để kiểm chứng
        real_train, real_test = train_test_split(df_real, test_size=0.30, random_state=42)
        print(f"📊 Chia dữ liệu thật: {len(real_train)} mẫu train / {len(real_test)} mẫu test kiểm chứng")

        # Đọc dữ liệu mô phỏng
        sim_csv = DATA_DIR / "orders.csv"
        if not sim_csv.exists():
            print(f"❌ Không tìm thấy dữ liệu mô phỏng tại {sim_csv}!")
            return
        df_sim = pd.read_csv(sim_csv)

        # Gộp dữ liệu: Sim (weight=1.0) + Real (weight=5.0)
        df_sim_train = df_sim[df_sim["timestamp"] < "2026-10-06"].copy()
        
        # Tạo sample weights
        sim_weights = np.ones(len(df_sim_train)) * 1.0
        real_weights = np.ones(len(real_train)) * 5.0

        X_combined = pd.concat([df_sim_train[ALL_FEATURES], real_train[ALL_FEATURES]], ignore_index=True)
        y_combined = np.concatenate([df_sim_train[TARGET_COL].values, real_train[actual_col].values])
        weights_combined = np.concatenate([sim_weights, real_weights])

        # Đánh giá mô hình HIỆN TẠI trên tập real_test
        X_real_test = real_test[ALL_FEATURES]
        y_real_test = real_test[actual_col].values
        old_pred = np.maximum(2.0, np.exp(pipeline.predict(X_real_test)) if use_log else pipeline.predict(X_real_test))
        old_mae = mean_absolute_error(y_real_test, old_pred)
        print(f"📌 MAE của mô hình HIỆN TẠI trên tập real_test (30%): {old_mae:.2f} phút")

        # Huấn luyện mô hình MỚI (clone cấu trúc)
        print("⏳ Đang huấn luyện mô hình ứng viên mới với trọng số 5x cho dữ liệu thật...")
        from sklearn.base import clone
        new_pipeline = clone(pipeline)
        
        target_combined = np.log(y_combined) if use_log else y_combined
        
        # Fit với sample_weight truyền vào regressor
        new_pipeline.fit(X_combined, target_combined, regressor__sample_weight=weights_combined)

        new_pred = np.maximum(2.0, np.exp(new_pipeline.predict(X_real_test)) if use_log else new_pipeline.predict(X_real_test))
        new_mae = mean_absolute_error(y_real_test, new_pred)
        print(f"🏆 MAE của mô hình MỚI trên tập real_test (30%): {new_mae:.2f} phút")

        # SO SÁNH & QUYẾT ĐỊNH
        if new_mae <= old_mae:
            print("\n✅ MÔ HÌNH MỚI TỐT HƠN HOẶC TƯƠNG ĐƯƠNG TRÊN DỮ LIỆU THẬT!")
            # Sao lưu mô hình cũ
            BACKUP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_rf = BACKUP_MODELS_DIR / f"rf_backup_{ts_str}.joblib"
            backup_cal = BACKUP_MODELS_DIR / f"calibration_backup_{ts_str}.json"
            shutil.copy(rf_path, backup_rf)
            if calib_path.exists():
                shutil.copy(calib_path, backup_cal)
            print(f"💾 Đã sao lưu mô hình cũ vào: {backup_rf.name}")

            # Lưu mô hình mới (compress=3)
            joblib.dump({"pipeline": new_pipeline, "use_log_transform": use_log}, rf_path, compress=3)
            print(f"🎉 ĐÃ CẬP NHẬT MÔ HÌNH MỚI VÀO: {rf_path} (MAE cải thiện: {old_mae:.2f}p -> {new_mae:.2f}p)")
        else:
            print(f"\n⚠️  MÔ HÌNH MỚI BỊ THOÁI HÓA TRÊN DỮ LIỆU THẬT (MAE tăng từ {old_mae:.2f}p -> {new_mae:.2f}p)!")
            print("🛡️  CƠ CHẾ BẢO VỆ: GIỮ NGUYÊN MÔ HÌNH CŨ, KHÔNG GHI ĐÈ ĐỂ TRÁNH SUY GIẢM ĐỘ CHÍNH XÁC.")

    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

