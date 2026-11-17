"""
prepare_images.py - Chuẩn hóa toàn bộ ảnh quán và ảnh món ăn về cùng tỉ lệ, kích thước cố định.
- Ảnh quán (store-*.webp): tỉ lệ 16:9 -> 800 x 450 px (WebP)
- Ảnh món ăn (*.webp):     tỉ lệ 4:3  -> 800 x 600 px (WebP)

Sử dụng Pillow ImageOps.fit (cắt giữa theo trọng tâm tùy chỉnh từng ảnh nếu cần + LANCZOS resize).
Kiểm tra đối chiếu với src/config.py và liệt kê các ảnh còn thiếu (nếu có).

Cách chạy:
    python scripts/prepare_images.py
    python scripts/prepare_images.py --input-dir path/to/raw_photos
"""
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import STORES, IMAGES_DIR

# Kích thước chuẩn cố định
STORE_SIZE = (800, 450)  # 16:9 cho ảnh bìa quán
DISH_SIZE = (800, 600)   # 4:3 cho ảnh món ăn
WEBP_QUALITY = 85

# Trọng tâm cắt tùy chỉnh (cx, cy) trong khoảng [0.0, 1.0], mặc định (0.5, 0.5) là cắt chính giữa.
# Có thể chỉnh từng ảnh nếu phần chính của món lệch lên trên (cy=0.35) hoặc xuống dưới (cy=0.65).
CROP_FOCUS: dict[str, tuple[float, float]] = {
    "ca-phe-muoi-da-nang.webp": (0.5, 0.45),
    "tra-dao-cam-sa.webp": (0.5, 0.55),
    "store-mi-quang.webp": (0.5, 0.55),
}

SUPPORTED_EXTS = {".webp", ".jpg", ".jpeg", ".png"}


def get_required_images() -> tuple[set[str], set[str]]:
    """Trả về (store_images, dish_images) khai báo trong src/config.py."""
    store_imgs = set()
    dish_imgs = set()
    for s_info in STORES.values():
        store_imgs.add(s_info["cover_image"])
        for d in s_info["dishes"]:
            dish_imgs.add(d["image"])
    return store_imgs, dish_imgs


def process_single_image(src_path: Path, dst_path: Path, target_size: tuple[int, int], focus: tuple[float, float] = (0.5, 0.5)):
    """Đọc ảnh nguồn, sửa xoay EXIF, cắt theo tỉ lệ chuẩn (center/focus crop) và lưu WebP."""
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        fitted = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS, centering=focus)
        fitted.save(dst_path, format="WEBP", quality=WEBP_QUALITY, method=6)


def main():
    parser = argparse.ArgumentParser(description="Chuẩn hóa kích thước ảnh quán (16:9) và ảnh món (4:3).")
    parser.add_argument("--input-dir", type=str, default=None, help="Thư mục chứa ảnh nguồn thô (.jpg/.png/.webp) cần đưa vào static/images/")
    args = parser.parse_args()

    store_imgs, dish_imgs = get_required_images()
    all_required = store_imgs | dish_imgs

    print("=" * 70)
    print("🖼️  CHUẨN HÓA ẢNH QUÁN (16:9 · 800x450) & ẢNH MÓN (4:3 · 800x600)")
    print("=" * 70)

    # Nếu có thư mục input-dir, xử lý và chuyển đổi sang app/static/images/<stem>.webp
    if args.input_dir:
        in_dir = Path(args.input_dir)
        if not in_dir.exists():
            print(f"❌ Không tìm thấy thư mục nguồn: {in_dir}")
            return
        for p in sorted(in_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                out_name = f"{p.stem}.webp"
                dst = IMAGES_DIR / out_name
                is_store = out_name.startswith("store-") or out_name in store_imgs
                size = STORE_SIZE if is_store else DISH_SIZE
                focus = CROP_FOCUS.get(out_name, (0.5, 0.5))
                process_single_image(p, dst, size, focus)
                print(f"  [Import] {p.name} -> {out_name} ({size[0]}x{size[1]})")

    # Chuẩn hóa mọi file ảnh đang nằm trong app/static/images/
    processed_count = 0
    for p in sorted(IMAGES_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        out_name = f"{p.stem}.webp"
        dst = IMAGES_DIR / out_name
        is_store = out_name.startswith("store-") or out_name in store_imgs
        size = STORE_SIZE if is_store else DISH_SIZE
        focus = CROP_FOCUS.get(out_name, (0.5, 0.5))

        # Kiểm tra nếu đã đúng định dạng webp và đúng kích thước thì bỏ qua hay vẫn chuẩn hóa
        needs_work = True
        if p.suffix.lower() == ".webp":
            with Image.open(p) as im:
                if im.size == size:
                    needs_work = False

        if needs_work:
            process_single_image(p, dst, size, focus)
            if p != dst and p.exists():
                p.unlink()  # Xóa file nguồn không phải webp sau khi đã convert
            processed_count += 1
            print(f"  ✅ Đã chuẩn hóa: {out_name:32s} -> {size[0]}x{size[1]} WebP")

    # Kiểm tra đối chiếu toàn bộ ảnh khai báo trong config.py
    existing_webp = {p.name for p in IMAGES_DIR.glob("*.webp")}
    missing_stores = sorted(store_imgs - existing_webp)
    missing_dishes = sorted(dish_imgs - existing_webp)

    print("-" * 70)
    print(f"📦 Tổng số ảnh khai báo trong config: {len(all_required)} ({len(store_imgs)} quán + {len(dish_imgs)} món)")
    print(f"✨ Đã xử lý / cắt tỉ lệ lại trong lần chạy này: {processed_count} ảnh")
    print(f"✅ Hiện có sẵn: {len(all_required & existing_webp)}/{len(all_required)} ảnh chuẩn kích thước")

    if missing_stores or missing_dishes:
        print("\n⚠️  DANH SÁCH ẢNH CÒN THIẾU (đặt đúng tên file rồi chạy lại script):")
        for m in missing_stores:
            print(f"   - [Quán 16:9] {m}")
        for m in missing_dishes:
            print(f"   - [Món  4:3 ] {m}")
    else:
        print("\n🎉 TẤT CẢ ẢNH QUÁN VÀ ẢNH MÓN ĐỀU ĐÃ ĐẦY ĐỦ VÀ ĐÚNG KÍCH THƯỚC CHUẨN!")
    print("=" * 70)


if __name__ == "__main__":
    main()
