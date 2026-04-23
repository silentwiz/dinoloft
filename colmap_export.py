"""
colmap_export.py
COLMAP .bin 결과물을 사람이 읽을 수 있는 JSON으로 변환.

출력:
  cameras.json   — 카메라 내부 파라미터
  images.json    — 이미지별 pose (qvec, tvec) + 메타
  points3D.json  — 3D 포인트 좌표 / 색상 / 재투영 오차

실행:
  python3 colmap_export.py
  python3 colmap_export.py --recon_dir sfm_output/sparse/1 --out_dir ./export
"""

import argparse
import json
import os
import struct
from pathlib import Path

from colmap_parser import read_cameras_bin, read_images_bin, read_points3d_bin


CAMERA_MODEL_NAMES = {
    0:  "SIMPLE_PINHOLE",
    1:  "PINHOLE",
    2:  "SIMPLE_RADIAL",
    3:  "RADIAL",
    4:  "OPENCV",
    5:  "OPENCV_FISHEYE",
    6:  "FULL_OPENCV",
    7:  "FOV",
    8:  "SIMPLE_RADIAL_FISHEYE",
    9:  "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

SPARSE_DIR = "sfm_output/sparse"
OUT_DIR    = "./colmap_export"


def _best_recon_dir(sparse_dir: str) -> str:
    """images.bin 기준 이미지 수가 가장 많은 reconstruction 폴더 반환."""
    best, best_count = None, 0
    for sub in sorted(os.listdir(sparse_dir)):
        p = os.path.join(sparse_dir, sub, "images.bin")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        if n > best_count:
            best, best_count = sub, n
    if best is None:
        raise FileNotFoundError(f"유효한 reconstruction이 없습니다: {sparse_dir}")
    return os.path.join(sparse_dir, best)


def export_cameras(cameras: dict, out_path: str):
    out = {}
    for cam_id, cam in cameras.items():
        out[str(cam_id)] = {
            "model":  CAMERA_MODEL_NAMES.get(cam["model_id"], f"UNKNOWN_{cam['model_id']}"),
            "width":  int(cam["width"]),
            "height": int(cam["height"]),
            "params": cam["params"].tolist(),
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}  ({len(out)}개 카메라)")


def export_images(images: dict, out_path: str):
    out = {}
    for img_id, img in images.items():
        out[str(img_id)] = {
            "name":      img["name"],
            "camera_id": int(img["camera_id"]),
            "qvec":      img["qvec"].tolist(),   # [qw, qx, qy, qz]
            "tvec":      img["tvec"].tolist(),   # [tx, ty, tz]
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}  ({len(out)}개 이미지)")


def export_points3d(points3d: dict, out_path: str):
    out = {}
    for pt_id, pt in points3d.items():
        out[str(pt_id)] = {
            "xyz":   pt["xyz"].tolist(),
            "rgb":   pt["rgb"].tolist(),
            "error": float(pt["error"]),
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}  ({len(out)}개 포인트)")


def main():
    parser = argparse.ArgumentParser(description="COLMAP .bin → JSON 변환")
    parser.add_argument("--recon_dir", default=None,
                        help="reconstruction 폴더 경로 (기본: sparse에서 가장 큰 폴더 자동 선택)")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help=f"출력 폴더 (기본: {OUT_DIR})")
    args = parser.parse_args()

    recon_dir = args.recon_dir or _best_recon_dir(SPARSE_DIR)
    print(f"[Export] reconstruction 폴더: {recon_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/3] cameras.bin 파싱...")
    cameras = read_cameras_bin(str(Path(recon_dir) / "cameras.bin"))

    print("[2/3] images.bin 파싱...")
    images = read_images_bin(str(Path(recon_dir) / "images.bin"))

    print("[3/3] points3D.bin 파싱...")
    points3d = read_points3d_bin(str(Path(recon_dir) / "points3D.bin"))

    print("\n[저장 중...]")
    export_cameras (cameras,  str(Path(args.out_dir) / "cameras.json"))
    export_images  (images,   str(Path(args.out_dir) / "images.json"))
    export_points3d(points3d, str(Path(args.out_dir) / "points3D.json"))

    print(f"\n완료! 출력 폴더: {args.out_dir}")


if __name__ == "__main__":
    main()
