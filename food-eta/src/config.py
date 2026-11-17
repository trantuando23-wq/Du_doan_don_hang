"""
config.py - Cấu hình dùng chung cho toàn bộ dự án Food ETA Đà Nẵng.
Cấu trúc phân cấp: Khu vực (REGIONS) -> Danh sách quán hư cấu -> Thực đơn món ăn.
Hoàn toàn không sử dụng tên quán thật hay địa chỉ thật.
"""
from pathlib import Path

# Thư mục gốc dự án food-eta
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "app" / "static"
IMAGES_DIR = STATIC_DIR / "images"
BACKUP_MODELS_DIR = MODELS_DIR / "backup"

# Đảm bảo các thư mục tồn tại
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = DATA_DIR / "feedback.csv"
REAL_ORDERS_TEMPLATE = DATA_DIR / "real_orders_template.csv"

# Cấu trúc phân cấp 3 khu vực -> 8 quán ăn hư cấu tại Đà Nẵng
# Cố ý có các quán cùng dòng món ở khu vực khác nhau (Mì Quảng, Bánh Xèo, Trà & Cà Phê)
# với tốc độ bếp, nhân sự và thời gian gốc khác nhau để người dùng so sánh.
REGIONS = [
    {
        "id": "hai_chau",
        "name": "Quận Hải Châu",
        "emoji": "🏙️",
        "tagline": "Trung tâm nhộn nhịp · Mì Quảng, cơm gà & trà phố",
        "stores": [
            {
                "store": "Mì Quảng Hạt Sen",
                "district": "Quận Hải Châu, Đà Nẵng",
                "emoji": "🍜",
                "description": "Mì Quảng nước nhưn đậm đà, bánh tráng mè giòn rụm",
                "open_hours": "06:30 – 21:30",
                "cover_image": "store-mi-quang.webp",
                "speed_factor": 0.92,
                "variance": 0.12,
                "staff_range": (1, 3),
                "dishes": [
                    {"dish": "Mì Quảng gà ta", "slug": "mi-quang-ga-ta", "group": "nuoc_dung", "base_time": 7.0, "image": "mi-quang-ga-ta.webp"},
                    {"dish": "Mì Quảng tôm thịt", "slug": "mi-quang-tom-thit", "group": "nuoc_dung", "base_time": 7.5, "image": "mi-quang-tom-thit.webp"},
                    {"dish": "Mì Quảng ếch đồng", "slug": "mi-quang-ech-dong", "group": "nuoc_dung", "base_time": 8.5, "image": "mi-quang-ech-dong.webp"},
                    {"dish": "Mì Quảng cá lóc", "slug": "mi-quang-ca-loc", "group": "nuoc_dung", "base_time": 9.0, "image": "mi-quang-ca-loc.webp"},
                    {"dish": "Mì Quảng bò tái", "slug": "mi-quang-bo-tai", "group": "nuoc_dung", "base_time": 8.0, "image": "mi-quang-bo-tai.webp"},
                ]
            },
            {
                "store": "Cơm Gà Phố Cổ",
                "district": "Quận Hải Châu, Đà Nẵng",
                "emoji": "🍗",
                "description": "Cơm gà xối mỡ vàng ươm, nấu bằng nước luộc gà thả vườn",
                "open_hours": "09:30 – 21:30",
                "cover_image": "store-com-ga.webp",
                "speed_factor": 0.88,
                "variance": 0.11,
                "staff_range": (2, 4),
                "dishes": [
                    {"dish": "Cơm gà xé Tam Kỳ", "slug": "com-ga-xe-tam-ky", "group": "com", "base_time": 6.0, "image": "com-ga-xe-tam-ky.webp"},
                    {"dish": "Cơm gà quay da giòn", "slug": "com-ga-quay-da-gion", "group": "com", "base_time": 7.5, "image": "com-ga-quay-da-gion.webp"},
                    {"dish": "Cơm gà luộc chặt", "slug": "com-ga-luoc-chat", "group": "com", "base_time": 6.5, "image": "com-ga-luoc-chat.webp"},
                    {"dish": "Cơm lòng mề gà xào nghệ", "slug": "com-long-me-ga-xao-nghe", "group": "com", "base_time": 7.0, "image": "com-long-me-ga-xao-nghe.webp"},
                ]
            },
            {
                "store": "Tiệm Trà Cố Đô",
                "district": "Quận Hải Châu, Đà Nẵng",
                "emoji": "🧋",
                "description": "Trà hoa ủ lạnh và cà phê phin rang mộc giữa lòng trung tâm",
                "open_hours": "07:00 – 22:30",
                "cover_image": "store-tiem-tra-co-do.webp",
                "speed_factor": 0.92,
                "variance": 0.13,
                "staff_range": (2, 3),
                "dishes": [
                    {"dish": "Cà phê muối Đà Nẵng", "slug": "ca-phe-muoi-da-nang", "group": "do_uong", "base_time": 4.2, "image": "ca-phe-muoi-da-nang.webp"},
                    {"dish": "Cà phê sữa đá phin", "slug": "ca-phe-sua-da", "group": "do_uong", "base_time": 3.5, "image": "ca-phe-sua-da.webp"},
                    {"dish": "Trà đào cam sả", "slug": "tra-dao-cam-sa", "group": "do_uong", "base_time": 3.8, "image": "tra-dao-cam-sa.webp"},
                    {"dish": "Trà vải lài thơm", "slug": "tra-vai-lai-thom", "group": "do_uong", "base_time": 3.8, "image": "tra-vai-lai-thom.webp"},
                    {"dish": "Trà sen vàng kem mặn", "slug": "tra-sen-vang", "group": "do_uong", "base_time": 4.8, "image": "tra-sen-vang.webp"},
                ]
            },
        ]
    },
    {
        "id": "son_tra",
        "name": "Quận Sơn Trà",
        "emoji": "🌊",
        "tagline": "Ven sông biển lộng gió · Bún chả cá, bánh xèo & mì Quảng",
        "stores": [
            {
                "store": "Bún Chả Cá Biển Xanh",
                "district": "Quận Sơn Trà, Đà Nẵng",
                "emoji": "🐟",
                "description": "Bún chả cá ngọt thanh từ cá biển tươi cập bến mỗi sáng",
                "open_hours": "06:00 – 21:00",
                "cover_image": "store-bun-cha-ca.webp",
                "speed_factor": 0.90,
                "variance": 0.10,
                "staff_range": (1, 3),
                "dishes": [
                    {"dish": "Bún chả cá đặc biệt", "slug": "bun-cha-ca-dac-biet", "group": "nuoc_dung", "base_time": 6.5, "image": "bun-cha-ca-dac-biet.webp"},
                    {"dish": "Bún cá thu tươi", "slug": "bun-ca-thu-tuoi", "group": "nuoc_dung", "base_time": 7.0, "image": "bun-ca-thu-tuoi.webp"},
                    {"dish": "Bún riêu cua đồng", "slug": "bun-rieu-cua-dong", "group": "nuoc_dung", "base_time": 6.0, "image": "bun-rieu-cua-dong.webp"},
                    {"dish": "Bún chả cá viên chiên", "slug": "bun-cha-ca-vien-chien", "group": "nuoc_dung", "base_time": 5.5, "image": "bun-cha-ca-vien-chien.webp"},
                ]
            },
            {
                "store": "Bánh Xèo Sông Hàn",
                "district": "Quận Sơn Trà, Đà Nẵng",
                "emoji": "🥞",
                "description": "Bánh xèo chảo gang giòn rụm kèm nem lụi và nước lèo béo",
                "open_hours": "10:00 – 22:00",
                "cover_image": "store-banh-xeo.webp",
                "speed_factor": 1.18,
                "variance": 0.18,
                "staff_range": (2, 4),
                "dishes": [
                    {"dish": "Bánh xèo tôm nhảy giòn", "slug": "banh-xeo-tom-nhay-gion", "group": "chien_nuong", "base_time": 13.0, "image": "banh-xeo-tom-nhay-gion.webp"},
                    {"dish": "Bánh xèo thịt bò bắp", "slug": "banh-xeo-thit-bo-bap", "group": "chien_nuong", "base_time": 12.5, "image": "banh-xeo-thit-bo-bap.webp"},
                    {"dish": "Nem lụi nướng than hoa", "slug": "nem-lui-nuong-than-hoa", "group": "chien_nuong", "base_time": 10.0, "image": "nem-lui-nuong-than-hoa.webp"},
                    {"dish": "Bò cuộn lá lốt nướng", "slug": "bo-cuon-la-lot-nuong", "group": "chien_nuong", "base_time": 11.0, "image": "bo-cuon-la-lot-nuong.webp"},
                ]
            },
            {
                "store": "Mì Quảng Gió Biển",
                "district": "Quận Sơn Trà, Đà Nẵng",
                "emoji": "🍲",
                "description": "Mì Quảng nấu mẻ nhỏ, đậm vị hải sản và thịt quê ven biển",
                "open_hours": "06:30 – 21:00",
                "cover_image": "store-mi-quang-gio-bien.webp",
                "speed_factor": 1.14,
                "variance": 0.14,
                "staff_range": (1, 2),
                "dishes": [
                    {"dish": "Mì Quảng gà ta", "slug": "mi-quang-ga-ta", "group": "nuoc_dung", "base_time": 8.2, "image": "mi-quang-ga-ta.webp"},
                    {"dish": "Mì Quảng tôm thịt", "slug": "mi-quang-tom-thit", "group": "nuoc_dung", "base_time": 8.6, "image": "mi-quang-tom-thit.webp"},
                    {"dish": "Mì Quảng ếch đồng", "slug": "mi-quang-ech-dong", "group": "nuoc_dung", "base_time": 9.5, "image": "mi-quang-ech-dong.webp"},
                    {"dish": "Mì Quảng sườn non", "slug": "mi-quang-suon-non", "group": "nuoc_dung", "base_time": 8.8, "image": "mi-quang-suon-non.webp"},
                    {"dish": "Mì Quảng thập cẩm", "slug": "mi-quang-thap-cam", "group": "nuoc_dung", "base_time": 9.2, "image": "mi-quang-thap-cam.webp"},
                ]
            },
        ]
    },
    {
        "id": "ngu_hanh_son",
        "name": "Quận Ngũ Hành Sơn",
        "emoji": "⛰️",
        "tagline": "Cung đường biển thoáng đãng · Cà phê muối & bánh xèo làng đá",
        "stores": [
            {
                "store": "Trà & Cà Phê Muối Bãi Biển",
                "district": "Quận Ngũ Hành Sơn, Đà Nẵng",
                "emoji": "☕",
                "description": "Cà phê muối béo mặn đặc trưng và trà trái cây mát lành",
                "open_hours": "07:00 – 22:30",
                "cover_image": "store-cafe-muoi.webp",
                "speed_factor": 0.75,
                "variance": 0.14,
                "staff_range": (1, 2),
                "dishes": [
                    {"dish": "Cà phê muối Đà Nẵng", "slug": "ca-phe-muoi-da-nang", "group": "do_uong", "base_time": 3.5, "image": "ca-phe-muoi-da-nang.webp"},
                    {"dish": "Cà phê cốt dừa đá", "slug": "ca-phe-cot-dua-da", "group": "do_uong", "base_time": 4.0, "image": "ca-phe-cot-dua-da.webp"},
                    {"dish": "Trà đào cam sả", "slug": "tra-dao-cam-sa", "group": "do_uong", "base_time": 3.0, "image": "tra-dao-cam-sa.webp"},
                    {"dish": "Trà vải lài thơm", "slug": "tra-vai-lai-thom", "group": "do_uong", "base_time": 3.0, "image": "tra-vai-lai-thom.webp"},
                    {"dish": "Sinh tố bơ sáp dừa", "slug": "sinh-to-bo-sap-dua", "group": "do_uong", "base_time": 5.0, "image": "sinh-to-bo-sap-dua.webp"},
                ]
            },
            {
                "store": "Bánh Xèo Làng Đá",
                "district": "Quận Ngũ Hành Sơn, Đà Nẵng",
                "emoji": "🥟",
                "description": "Bánh xèo vỏ mỏng đổ nhanh, mực sữa giòn và rau sống sạch",
                "open_hours": "10:30 – 22:00",
                "cover_image": "store-banh-xeo-lang-da.webp",
                "speed_factor": 0.94,
                "variance": 0.15,
                "staff_range": (1, 3),
                "dishes": [
                    {"dish": "Bánh xèo tôm nhảy giòn", "slug": "banh-xeo-tom-nhay-gion", "group": "chien_nuong", "base_time": 11.0, "image": "banh-xeo-tom-nhay-gion.webp"},
                    {"dish": "Bánh xèo thịt bò bắp", "slug": "banh-xeo-thit-bo-bap", "group": "chien_nuong", "base_time": 10.5, "image": "banh-xeo-thit-bo-bap.webp"},
                    {"dish": "Bánh xèo mực sữa", "slug": "banh-xeo-muc-sua", "group": "chien_nuong", "base_time": 11.5, "image": "banh-xeo-muc-sua.webp"},
                    {"dish": "Nem lụi nướng than hoa", "slug": "nem-lui-nuong-than-hoa", "group": "chien_nuong", "base_time": 8.5, "image": "nem-lui-nuong-than-hoa.webp"},
                    {"dish": "Bò cuộn lá lốt nướng", "slug": "bo-cuon-la-lot-nuong", "group": "chien_nuong", "base_time": 9.5, "image": "bo-cuon-la-lot-nuong.webp"},
                ]
            },
        ]
    },
]

# Bảng tra cứu phẳng (Flat lookup) tự động xây từ REGIONS để các module ML & API dùng thuận tiện
STORES = {}
STORE_TO_REGION = {}
for _region in REGIONS:
    for _store_cfg in _region["stores"]:
        _s_name = _store_cfg["store"]
        _entry = dict(_store_cfg)
        _entry["region_id"] = _region["id"]
        _entry["region_name"] = _region["name"]
        STORES[_s_name] = _entry
        STORE_TO_REGION[_s_name] = _region["name"]

# Danh sách hình thức nhận món (Channel)
CHANNELS = ["tai_quan", "mang_di", "giao_app"]
CHANNEL_LABELS = {
    "tai_quan": "Ăn tại quán",
    "mang_di": "Mang đi",
    "giao_app": "Giao qua app"
}

# Hệ số ảnh hưởng của 2 đặc trưng (đặt ở 1 chỗ để dễ tinh chỉnh)
CHANNEL_EFFECTS = {
    "tai_quan": {"extra_minutes": 0.0, "noise_scale": 1.0},
    "mang_di": {"extra_minutes": 1.5, "noise_scale": 1.1},   # +1-2 phút đóng gói hộp, túi
    "giao_app": {"extra_minutes": 3.0, "noise_scale": 1.25}  # +2-4 phút gom đơn, giao tài xế, dao động lớn hơn
}

# Tương tác queue × staff: queue_effect / (STAFF_INTERACTION_BASE + STAFF_INTERACTION_SLOPE * staff)
STAFF_INTERACTION_BASE = 0.6
STAFF_INTERACTION_SLOPE = 0.4

# Tên các nhóm món (chỉ dùng nội bộ cho mô hình và thống kê, KHÔNG trả ra giao diện web)
DISH_GROUP_NAMES = {
    "nuoc_dung": "Món nước / Mì",
    "com": "Món cơm",
    "chien_nuong": "Chiên / Nướng",
    "do_uong": "Đồ uống / Cà phê"
}

# Danh sách features dùng để huấn luyện và dự đoán (giữ nguyên, KHÔNG thêm đặc trưng mới)
CATEGORICAL_FEATURES = ["store", "dish", "dish_group", "weekday", "channel"]
NUMERIC_FEATURES = ["qty", "hour", "is_weekend", "queue", "staff"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COL = "minutes"
