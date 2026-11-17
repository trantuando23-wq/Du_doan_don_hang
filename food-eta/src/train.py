"""
train.py - Huấn luyện mô hình Linear Regression và Random Forest Regressor.
Chia dữ liệu theo thời gian (Proper Train -> Calibration Set -> Test Set).
Hiệu chuẩn khoảng tin cậy bằng Split Conformal Prediction.
Tạo Queue & Staff Profile mở rộng, kiểm tra tính hợp lý, xuất báo cáo và biểu đồ vào reports/.
Hỗ trợ đầy đủ 2 đặc trưng mới: staff (nhân sự bếp) và channel (hình thức nhận món).
"""
import sys
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GroupKFold
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    STORES, STORE_TO_REGION, DATA_DIR, MODELS_DIR, REPORTS_DIR,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES, TARGET_COL
)
from src.report import (
    create_charts, analyze_error_by_group, check_monotonicity, write_markdown_report
)

def evaluate_predictions(y_true, y_pred):
    """Tính các chỉ số MAE, RMSE, R2, và % sai số dưới 5 phút."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pct_5min = float(np.mean(np.abs(y_true - y_pred) <= 5.0) * 100.0)
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "pct_within_5min": round(pct_5min, 2)
    }

def build_queue_profile(df: pd.DataFrame) -> dict:
    """
    Tính toán số đơn chờ (queue) và số nhân viên bếp (staff) trung bình theo (quán, loại ngày, giờ).
    Lưu cả dạng truyền thống và mở rộng để tương thích tuyệt đối với API.
    """
    profile = {}
    staff_profile = {}
    extended = {}
    
    # Mặc định theo từng quán
    for store_name, s_info in STORES.items():
        profile[store_name] = {"weekday": {}, "weekend": {}}
        staff_profile[store_name] = {"weekday": {}, "weekend": {}}
        extended[store_name] = {"weekday": {}, "weekend": {}}
        min_s, max_s = s_info.get("staff_range", (1, 3))
        
        for h in range(24):
            is_peak = (11 <= h <= 13) or (18 <= h <= 20)
            if is_peak:
                def_wd_q, def_we_q = 11.0, 14.0
                def_staff = max_s
            elif 7 <= h <= 21:
                def_wd_q, def_we_q = 5.0, 6.5
                def_staff = max(min_s, max_s - 1)
            else:
                def_wd_q, def_we_q = 1.0, 1.5
                def_staff = min_s
                
            h_str = str(h)
            profile[store_name]["weekday"][h_str] = round(def_wd_q, 1)
            profile[store_name]["weekend"][h_str] = round(def_we_q, 1)
            staff_profile[store_name]["weekday"][h_str] = int(def_staff)
            staff_profile[store_name]["weekend"][h_str] = int(def_staff)
            extended[store_name]["weekday"][h_str] = {"queue": round(def_wd_q, 1), "staff": int(def_staff)}
            extended[store_name]["weekend"][h_str] = {"queue": round(def_we_q, 1), "staff": int(def_staff)}
            
    # Tính toán thực tế từ dữ liệu
    grouped = df.groupby(["store", "is_weekend", "hour"])[["queue", "staff"]].agg("mean").reset_index()
    for _, row in grouped.iterrows():
        s = str(row["store"])
        if s not in profile:
            continue
        is_we = int(row["is_weekend"])
        day_type = "weekend" if is_we == 1 else "weekday"
        h_str = str(int(row["hour"]))
        q_val = round(float(row["queue"]), 1)
        st_val = int(round(float(row["staff"])))
        
        profile[s][day_type][h_str] = q_val
        staff_profile[s][day_type][h_str] = st_val
        extended[s][day_type][h_str] = {"queue": q_val, "staff": st_val}
        
    return {
        "profile": profile,
        "staff_profile": staff_profile,
        "extended_profile": extended,
        "labels": {
            "thresholds": [
                {"max": 4, "key": "vang", "label": "Vắng khách", "badge_color": "emerald"},
                {"max": 11, "key": "vua", "label": "Vừa phải", "badge_color": "blue"},
                {"max": 18, "key": "kha_dong", "label": "Khá đông", "badge_color": "amber"},
                {"max": 999, "key": "rat_dong", "label": "Rất đông", "badge_color": "rose"}
            ]
        }
    }

def evaluate_unseen_stores(df: pd.DataFrame, rf_pipeline_template) -> dict:
    """
    ĐÁNH GIÁ QUÁN CHƯA TỪNG THẤY (Leave-One-Store-Out bằng GroupKFold theo cột store).
    Huấn luyện Random Forest trên các quán còn lại (đã loại 1% outlier trên tập train),
    rồi kiểm thử trực tiếp trên quán bị loại để đo khả năng tổng quát hóa sang quán mới.
    """
    unique_stores = df["store"].unique()
    n_splits = len(unique_stores)
    gkf = GroupKFold(n_splits=n_splits)

    X_all = df[ALL_FEATURES]
    y_all = df[TARGET_COL].values
    groups = df["store"].values

    per_store_results = []
    all_y_true = []
    all_y_pred = []

    print(f"\n🔬 Đang chạy đánh giá quán chưa từng thấy (GroupKFold {n_splits} quán)...")
    for train_idx, test_idx in gkf.split(X_all, y_all, groups=groups):
        held_out_store = str(groups[test_idx[0]])
        X_tr = X_all.iloc[train_idx]
        y_tr = y_all[train_idx]

        # Loại 1% outlier cực đoan trên tập train giống quy trình huấn luyện chính
        q99_tr = np.quantile(y_tr, 0.99)
        clean_mask = y_tr <= q99_tr
        X_tr_clean = X_tr.iloc[clean_mask]
        y_tr_clean = y_tr[clean_mask]

        X_te = X_all.iloc[test_idx]
        y_te = y_all[test_idx]

        model = clone(rf_pipeline_template)
        model.fit(X_tr_clean, np.log(y_tr_clean))
        y_pred = np.maximum(2.0, np.exp(model.predict(X_te)))

        m = evaluate_predictions(y_te, y_pred)
        bias = round(float(np.mean(y_pred - y_te)), 2)

        # Kiểm tra xem quán bị loại có dùng chung nhóm món với ít nhất 1 quán khác trong tập train không
        held_groups = set(X_te["dish_group"].unique())
        train_groups = set(X_tr_clean["dish_group"].unique())
        has_shared_group = len(held_groups & train_groups) > 0

        held_dishes = set(X_te["dish"].unique())
        train_dishes = set(X_tr_clean["dish"].unique())
        shared_dish_count = len(held_dishes & train_dishes)

        per_store_results.append({
            "store": held_out_store,
            "region": STORE_TO_REGION.get(held_out_store, ""),
            "n_samples": int(len(y_te)),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "pct_within_5min": m["pct_within_5min"],
            "mean_actual": round(float(np.mean(y_te)), 2),
            "mean_pred": round(float(np.mean(y_pred)), 2),
            "bias": bias,
            "has_shared_group": has_shared_group,
            "shared_dish_count": shared_dish_count
        })
        all_y_true.append(y_te)
        all_y_pred.append(y_pred)

    all_y_true_arr = np.concatenate(all_y_true)
    all_y_pred_arr = np.concatenate(all_y_pred)
    overall = evaluate_predictions(all_y_true_arr, all_y_pred_arr)
    per_store_results.sort(key=lambda r: r["mae"])

    return {
        "n_stores": n_splits,
        "overall": overall,
        "per_store": per_store_results
    }


def train_and_evaluate():
    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN, HIỆU CHUẨN & ĐÁNH GIÁ MÔ HÌNH HỌC MÁY")
    print("   ĐẶC TRƯNG MỚI: staff (nhân viên bếp) và channel (hình thức nhận món)")
    print("=" * 70)
    
    # 1. Đọc dữ liệu
    csv_path = DATA_DIR / "orders.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy {csv_path}. Hãy chạy generate_data.py trước!")
        
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Phân chia theo thời gian (Temporal Split)
    split_calib_date = pd.to_datetime("2026-09-29 00:00:00")
    split_test_date = pd.to_datetime("2026-10-06 00:00:00")
    
    proper_mask = df["timestamp"] < split_calib_date
    calib_mask = (df["timestamp"] >= split_calib_date) & (df["timestamp"] < split_test_date)
    test_mask = df["timestamp"] >= split_test_date
    
    df_proper = df[proper_mask].copy()
    df_calib = df[calib_mask].copy()
    df_test = df[test_mask].copy()
    
    df_train_full = df[df["timestamp"] < split_test_date].copy()
    q99 = df_train_full[TARGET_COL].quantile(0.99)
    df_proper_cleaned = df_proper[df_proper[TARGET_COL] <= q99].copy()
    
    print(f"📊 Tập Proper Train (4 tuần đầu): {len(df_proper_cleaned):,} mẫu")
    print(f"📊 Tập Calibration   (tuần thứ 5): {len(df_calib):,} mẫu (Dành cho Split Conformal)")
    print(f"📊 Tập Test         (tuần thứ 6): {len(df_test):,} mẫu")
    print(f"🔍 Danh sách Categorical: {CATEGORICAL_FEATURES}")
    print(f"🔍 Danh sách Numeric:     {NUMERIC_FEATURES}")
    
    # Lưu Queue Profile mở rộng
    queue_profile_data = build_queue_profile(df_train_full)
    with open(MODELS_DIR / "queue_profile.json", "w", encoding="utf-8") as f:
        json.dump(queue_profile_data, f, ensure_ascii=False, indent=2)
        
    X_proper = df_proper_cleaned[ALL_FEATURES]
    y_proper = df_proper_cleaned[TARGET_COL].values
    
    X_calib = df_calib[ALL_FEATURES]
    y_calib = df_calib[TARGET_COL].values
    
    X_test = df_test[ALL_FEATURES]
    y_test = df_test[TARGET_COL].values
    
    # 3. Baseline Model (Trung bình theo món)
    dish_means = df_proper_cleaned.groupby("dish")[TARGET_COL].mean().to_dict()
    global_mean = float(df_proper_cleaned[TARGET_COL].mean())
    y_pred_baseline = X_test["dish"].map(dish_means).fillna(global_mean).values
    baseline_metrics = evaluate_predictions(y_test, y_pred_baseline)
    
    # 4. Kiểm tra Log Transform
    use_log_transform = True
    target_proper = np.log(y_proper)
    
    # 5. Huấn luyện Linear Regression
    preprocessor_linear = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES)
        ]
    )
    linear_pipeline = Pipeline([
        ("preprocessor", preprocessor_linear),
        ("regressor", LinearRegression())
    ])
    linear_pipeline.fit(X_proper, target_proper)
    y_pred_linear = np.maximum(2.0, np.exp(linear_pipeline.predict(X_test)))
    linear_metrics = evaluate_predictions(y_test, y_pred_linear)
    
    # 6. Huấn luyện Random Forest Regressor & Tuning
    preprocessor_rf = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES)
        ]
    )
    rf_base = RandomForestRegressor(random_state=42, n_jobs=2)
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor_rf),
        ("regressor", rf_base)
    ])
    
    param_dist = {
        "regressor__n_estimators": [80, 100, 120],
        "regressor__max_depth": [10, 15, 20, None],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", 0.8, 1.0]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    rf_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=1,
        verbose=0
    )
    rf_search.fit(X_proper, target_proper)
    best_rf_pipeline = rf_search.best_estimator_
    
    # Dự đoán Random Forest trên test
    y_pred_rf = np.maximum(2.0, np.exp(best_rf_pipeline.predict(X_test)))
    rf_metrics = evaluate_predictions(y_test, y_pred_rf)
    
    # 7. HIỆU CHUẨN KHOẢNG DỰ ĐOÁN BẰNG SPLIT CONFORMAL PREDICTION
    rf_reg = best_rf_pipeline.named_steps["regressor"]
    rf_prep = best_rf_pipeline.named_steps["preprocessor"]
    
    X_calib_trans = rf_prep.transform(X_calib)
    calib_tree_preds = np.exp([tree.predict(X_calib_trans) for tree in rf_reg.estimators_])
    L_calib = np.percentile(calib_tree_preds, 10, axis=0)
    U_calib = np.percentile(calib_tree_preds, 90, axis=0)
    
    scores = np.maximum(L_calib - y_calib, y_calib - U_calib)
    n_calib = len(scores)
    q_level = min(1.0, np.ceil((n_calib + 1) * 0.80) / n_calib)
    q_calib = float(np.quantile(scores, q_level))
    q_calib = round(max(0.0, q_calib), 2)
    
    X_test_trans = rf_prep.transform(X_test)
    test_tree_preds = np.exp([tree.predict(X_test_trans) for tree in rf_reg.estimators_])
    L_test_raw = np.percentile(test_tree_preds, 10, axis=0)
    U_test_raw = np.percentile(test_tree_preds, 90, axis=0)
    
    pre_cov = float(np.mean((y_test >= L_test_raw) & (y_test <= U_test_raw)) * 100.0)
    
    L_test_calib = np.maximum(2.0, L_test_raw - q_calib)
    U_test_calib = U_test_raw + q_calib
    post_cov = float(np.mean((y_test >= L_test_calib) & (y_test <= U_test_calib)) * 100.0)
    
    calib_info = {
        "target_coverage": 80.0,
        "q_calib": q_calib,
        "pre_coverage_test": round(pre_cov, 2),
        "post_coverage_test": round(post_cov, 2),
        "calibration_samples": n_calib
    }
    
    with open(MODELS_DIR / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_info, f, ensure_ascii=False, indent=2)
        
    # 8. Feature Importances
    cat_encoder = best_rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_trans_feature_names = cat_feature_names + NUMERIC_FEATURES
    rf_importances = rf_reg.feature_importances_
    sorted_rf_idx = np.argsort(rf_importances)[::-1]
    feature_importance_list = [
        {"feature": all_trans_feature_names[i], "importance": round(float(rf_importances[i]), 4)}
        for i in sorted_rf_idx[:15]
    ]
    
    # 9. Lưu Models Joblib (Nén compress=3 để kích thước file < 10MB, dễ tải lên GitHub)
    joblib.dump({"pipeline": linear_pipeline, "use_log_transform": True}, MODELS_DIR / "linear.joblib")
    joblib.dump({
        "pipeline": best_rf_pipeline,
        "use_log_transform": True,
        "calibration": calib_info
    }, MODELS_DIR / "rf.joblib", compress=3)
    
    # 10. Tạo các biểu đồ reports/*.png
    create_charts(y_test, y_pred_rf, y_pred_linear, df_test, feature_importance_list)
    
    # 11. Phân tích lỗi chi tiết & xuất reports/error_by_group.csv
    error_df, insights = analyze_error_by_group(df_test, y_test, y_pred_rf)
    
    # 12. Kiểm tra tính hợp lý (Monotonicity Check 500 mẫu)
    sanity = check_monotonicity(best_rf_pipeline, use_log=True, n_samples=500)

    # 12b. Đánh giá quán chưa từng thấy (GroupKFold theo quán - Leave-One-Store-Out)
    unseen_store_eval = evaluate_unseen_stores(df, best_rf_pipeline)

    # 13. Xuất metrics.json và reports/report.md
    metrics_data = {
        "models": {
            "baseline": baseline_metrics,
            "linear": linear_metrics,
            "random_forest": rf_metrics
        },
        "conformal_calibration": calib_info,
        "sanity_check": sanity,
        "unseen_store_evaluation": unseen_store_eval,
        "best_rf_params": {k.replace("regressor__", ""): v for k, v in rf_search.best_params_.items()},
        "dataset_summary": {
            "total_samples": len(df),
            "n_stores": len(STORES),
            "proper_train_samples": len(df_proper_cleaned),
            "calib_samples": len(df_calib),
            "train_samples": len(df_proper_cleaned) + len(df_calib),
            "test_samples": len(df_test),
            "train_period": f"{df_train_full['timestamp'].min().strftime('%Y-%m-%d')} đến {df_train_full['timestamp'].max().strftime('%Y-%m-%d')}",
            "test_period": f"{df_test['timestamp'].min().strftime('%Y-%m-%d')} đến {df_test['timestamp'].max().strftime('%Y-%m-%d')}"
        }
    }

    # Đọc bản cũ trước khi mở rộng quán (backup_v2 -> fallback backup_v1) để so sánh
    prev_metrics = None
    prev_label = "backup_v2 (5 quán · 10.000 đơn)"
    prev_metrics_path = BASE_DIR / "backup_v2" / "models" / "metrics.json"
    if not prev_metrics_path.exists():
        prev_metrics_path = BASE_DIR / "backup_v1" / "models" / "metrics.json"
        prev_label = "backup_v1 (5 quán)"
    if prev_metrics_path.exists():
        try:
            with open(prev_metrics_path, "r", encoding="utf-8") as f:
                prev_metrics = json.load(f)
        except Exception:
            pass

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    report_path = write_markdown_report(
        metrics_data, error_df, insights, sanity, calib_info,
        prev_metrics=prev_metrics, prev_label=prev_label,
        unseen_store_eval=unseen_store_eval
    )

    # 14. IN TÓM TẮT GỌN GÀNG RA TERMINAL
    print("\n" + "=" * 74)
    print("📈 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (TUẦN 6 · 8 QUÁN · 3 KHU VỰC)")
    print("=" * 74)
    summary_table = pd.DataFrame([
        {
            "Mô hình": "Baseline (TB món)",
            "MAE (phút)": baseline_metrics["mae"],
            "RMSE (phút)": baseline_metrics["rmse"],
            "R² Score": baseline_metrics["r2"],
            "% Lệch < 5p": f"{baseline_metrics['pct_within_5min']}%"
        },
        {
            "Mô hình": "Linear Regression",
            "MAE (phút)": linear_metrics["mae"],
            "RMSE (phút)": linear_metrics["rmse"],
            "R² Score": linear_metrics["r2"],
            "% Lệch < 5p": f"{linear_metrics['pct_within_5min']}%"
        },
        {
            "Mô hình": "Random Forest Regressor",
            "MAE (phút)": rf_metrics["mae"],
            "RMSE (phút)": rf_metrics["rmse"],
            "R² Score": rf_metrics["r2"],
            "% Lệch < 5p": f"{rf_metrics['pct_within_5min']}%"
        }
    ])
    print(summary_table.to_string(index=False))

    if prev_metrics and "models" in prev_metrics:
        p_rf = prev_metrics["models"]["random_forest"]
        print(f"\n--- SO SÁNH TRƯỚC / SAU MỞ RỘNG QUÁN ({prev_label} -> 8 quán · 15.000 đơn) ---")
        print(f"Random Forest MAE:  {p_rf['mae']}p -> {rf_metrics['mae']}p ({rf_metrics['mae'] - p_rf['mae']:+.2f}p)")
        print(f"Random Forest RMSE: {p_rf['rmse']}p -> {rf_metrics['rmse']}p ({rf_metrics['rmse'] - p_rf['rmse']:+.2f}p)")
        print(f"Random Forest R²:   {p_rf['r2']} -> {rf_metrics['r2']} ({rf_metrics['r2'] - p_rf['r2']:+.4f})")
        print(f"Độ phủ Conformal:   {prev_metrics.get('conformal_calibration', {}).get('post_coverage_test', 'N/A')}% -> {calib_info['post_coverage_test']}%")

    print("\n--- HIỆU CHUẨN SPLIT CONFORMAL PREDICTION (TARGET: 80%) ---")
    print(f"Độ phủ trước hiệu chuẩn: {calib_info['pre_coverage_test']}%")
    print(f"Độ lệch hiệu chỉnh (q_calib): +{calib_info['q_calib']} phút")
    print(f"Độ phủ sau hiệu chuẩn:  {calib_info['post_coverage_test']}%")

    print("\n--- ĐÁNH GIÁ QUÁN CHƯA TỪNG THẤY (GroupKFold THEO QUÁN) ---")
    unseen_df = pd.DataFrame([
        {
            "Quán bị ẩn (Test)": r["store"],
            "MAE (p)": r["mae"],
            "Bias (p)": f"{r['bias']:+.2f}",
            "R²": r["r2"],
            "% <5p": f"{r['pct_within_5min']}%",
            "Món chung": r["shared_dish_count"]
        }
        for r in unseen_store_eval["per_store"]
    ])
    print(unseen_df.to_string(index=False))
    print(f"👉 MAE tổng thể trên quán chưa từng thấy: {unseen_store_eval['overall']['mae']} phút (so với Test chuẩn: {rf_metrics['mae']} phút)")

    print("\n--- KIỂM TRA TÍNH ĐƠN ĐIỆU (500 MẪU) ---")
    print(f"Queue tăng (+5): Vi phạm {sanity['queue_monotonicity']['rate_pct']}% ({sanity['queue_monotonicity']['violations']}/500)")
    print(f"Qty tăng (+2):   Vi phạm {sanity['qty_monotonicity']['rate_pct']}% ({sanity['qty_monotonicity']['violations']}/500)")
    print(f"Staff tăng (+1): Vi phạm {sanity['staff_monotonicity']['rate_pct']}% (thời gian giảm/giữ nguyên)")
    print(f"Kênh Giao app vs Ăn tại quán: {sanity['channel_sanity']['app_longer_rate_pct']}% lâu hơn, TB +{sanity['channel_sanity']['avg_app_extra_minutes']}p")
    print(f"\n📄 Báo cáo đầy đủ: {report_path}")
    print("=" * 74)

if __name__ == "__main__":
    train_and_evaluate()
