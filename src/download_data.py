import os
import sys
import zipfile
import argparse
import urllib.request

# Base URL của Amazon S3 bucket chứa dữ liệu Citi Bike
BASE_URL = "https://s3.amazonaws.com/tripdata/"

# Danh sách chính xác các file dữ liệu Jersey City từ tháng 10/2025 đến tháng 10/2026
# (Quan sát từ ảnh giao diện S3: một số tháng có đuôi .csv.zip, một số tháng có đuôi .zip)
JC_FILES = [
    "JC-202510-citibike-tripdata.zip",
    "JC-202511-citibike-tripdata.csv.zip",
    "JC-202512-citibike-tripdata.csv.zip",
    "JC-202601-citibike-tripdata.zip",
    "JC-202602-citibike-tripdata.csv.zip",
    "JC-202603-citibike-tripdata.csv.zip",
    "JC-202604-citibike-tripdata.zip",
    "JC-202605-citibike-tripdata.csv.zip",
    "JC-202606-citibike-tripdata.csv.zip",
    "JC-202607-citibike-tripdata.csv.zip",
    "JC-202608-citibike-tripdata.csv.zip",
    "JC-202609-citibike-tripdata.csv.zip",
    "JC-202610-citibike-tripdata.zip",
]

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_and_extract(filename, dest_dir=RAW_DATA_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, filename)
    url = BASE_URL + filename

    # Kiểm tra xem file CSV tương ứng đã được giải nén chưa
    base_name = filename.replace(".csv.zip", "").replace(".zip", "")
    target_csv = os.path.join(dest_dir, f"{base_name}.csv")
    if os.path.exists(target_csv):
        print(f"[=] Đã tồn tại {base_name}.csv, bỏ qua bước tải.")
        return

    print(f"[*] Đang tải: {filename}...")
    try:
        # Tải file từ S3 với User-Agent chuẩn
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"    -> Đã tải xong ({len(data) / (1024 * 1024):.2f} MB)")

        # Giải nén lấy file CSV
        print(f"[*] Đang giải nén {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                # Bỏ qua các file ẩn của macOS (__MACOSX)
                if not member.startswith("__MACOSX") and member.endswith(".csv"):
                    zip_ref.extract(member, dest_dir)
                    print(f"    -> Trích xuất thành công: {member}")

    except Exception as e:
        print(f"[!] Lỗi khi tải/giải nén {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu Citi Bike Jersey City từ AWS S3")
    parser.add_argument(
        "--mode",
        choices=["prototype", "all"],
        default="prototype",
        help="Chế độ tải: 'prototype' (2 tháng gần nhất để test) hoặc 'all' (toàn bộ 12 tháng)",
    )
    args = parser.parse_args()

    files_to_download = JC_FILES[-2:] if args.mode == "prototype" else JC_FILES
    print(f"==================================================")
    print(f"Bắt đầu tải dữ liệu Citi Bike (Jersey City) - Chế độ: {args.mode.upper()}")
    print(f"Số lượng file sẽ tải: {len(files_to_download)}")
    print(f"Thư mục lưu: {os.path.abspath(RAW_DATA_DIR)}")
    print(f"==================================================")

    for idx, f in enumerate(files_to_download, 1):
        print(f"\n[{idx}/{len(files_to_download)}]")
        download_and_extract(f)

    print("\n[+] Hoàn thành toàn bộ quá trình tải dữ liệu!")


if __name__ == "__main__":
    main()
