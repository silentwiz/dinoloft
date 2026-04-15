"""
colmap_parser.py
COLMAP .bin 파일(cameras, images, points3D)을 읽어 파이썬 딕셔너리로 반환.
"""

import struct
import numpy as np
from pathlib import Path


# COLMAP 카메라 모델별 파라미터 개수
CAMERA_MODEL_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE : f, cx, cy
    1: 4,   # PINHOLE        : fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL  : f, cx, cy, k
    3: 5,   # RADIAL         : f, cx, cy, k1, k2
    4: 8,   # OPENCV         : fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


def read_cameras_bin(path: str) -> dict:
    """
    cameras.bin 파싱.

    반환값:
        {camera_id: {"model_id": int, "width": int, "height": int, "params": np.ndarray}}
    """
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id  = struct.unpack("<I", f.read(4))[0]
            model_id   = struct.unpack("<i", f.read(4))[0]
            width      = struct.unpack("<Q", f.read(8))[0]
            height     = struct.unpack("<Q", f.read(8))[0]
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_id, 0)
            params     = np.array(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[camera_id] = {
                "model_id": model_id,
                "width":    width,
                "height":   height,
                "params":   params,
            }
    return cameras


def read_images_bin(path: str) -> dict:
    """
    images.bin 파싱.

    반환값:
        {image_id: {
            "name":      str,
            "camera_id": int,
            "qvec":      np.ndarray([qw, qx, qy, qz]),
            "tvec":      np.ndarray([tx, ty, tz]),
            "xys":       np.ndarray([[x, y], ...]),   # 2D 관측 좌표
            "point3D_ids": np.ndarray([id, ...]),      # 대응 3D 포인트 ID (-1 = 없음)
        }}
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id  = struct.unpack("<I",  f.read(4))[0]
            qvec      = np.array(struct.unpack("<4d", f.read(32)))   # qw qx qy qz
            tvec      = np.array(struct.unpack("<3d", f.read(24)))   # tx ty tz
            camera_id = struct.unpack("<I",  f.read(4))[0]

            # null-terminated 문자열 읽기
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")

            num_pts2d = struct.unpack("<Q", f.read(8))[0]

            # COLMAP binary 형식: 포인트마다 [x:f64, y:f64, id:i64] 인터리브 저장
            # → structured array로 한 번에 파싱 (블록 분리 읽기는 잘못된 방식)
            dt  = np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            raw = np.frombuffer(f.read(24 * num_pts2d), dtype=dt)

            xys         = np.column_stack([raw["x"], raw["y"]])   # [N, 2]
            point3D_ids = raw["id"].astype(np.int64)               # [N]

            images[image_id] = {
                "name":        name,
                "camera_id":   camera_id,
                "qvec":        qvec,
                "tvec":        tvec,
                "xys":         xys,
                "point3D_ids": point3D_ids,
            }
    return images


def read_points3d_bin(path: str) -> dict:
    """
    points3D.bin 파싱.

    반환값:
        {point3D_id: {"xyz": np.ndarray([x, y, z]), "rgb": np.ndarray([r, g, b]), "error": float}}
    """
    points3d = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id   = struct.unpack("<Q", f.read(8))[0]
            xyz          = np.array(struct.unpack("<3d", f.read(24)))
            rgb          = np.array(struct.unpack("<3B", f.read(3)),  dtype=np.uint8)
            error        = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            # track = [(image_id, point2D_idx), ...] — 여기선 스킵
            f.read(8 * track_length)

            points3d[point3D_id] = {
                "xyz":   xyz,
                "rgb":   rgb,
                "error": error,
            }
    return points3d


def build_mapping(images: dict, points3d: dict) -> dict:
    """
    images + points3d → 이미지별 2D-3D 대응점 LUT 생성.

    반환값:
        {img_name: [{"2d": (x, y), "xyz": np.ndarray([x, y, z])}, ...]}
    """
    mapping = {}
    for img in images.values():
        name   = img["name"]
        pairs  = []
        for xy, p3d_id in zip(img["xys"], img["point3D_ids"]):
            if p3d_id == -1:
                continue
            if p3d_id not in points3d:
                continue
            pairs.append({
                "2d":  (float(xy[0]), float(xy[1])),
                "xyz": points3d[p3d_id]["xyz"],
            })
        mapping[name] = pairs
    return mapping


def load_colmap_model(sparse_dir: str) -> tuple[dict, dict, dict, dict]:
    """
    sparse_dir (예: sfm_output/sparse/0) 안의 .bin 파일 3종을 한 번에 로드.

    반환값: (cameras, images, points3d, mapping)
    """
    base = Path(sparse_dir)
    print(f"[colmap_parser] 파싱 경로: {base}")

    cameras  = read_cameras_bin(str(base / "cameras.bin"))
    print(f"  cameras  : {len(cameras)}개")

    images   = read_images_bin(str(base / "images.bin"))
    print(f"  images   : {len(images)}개")

    points3d = read_points3d_bin(str(base / "points3D.bin"))
    print(f"  points3D : {len(points3d)}개")

    mapping  = build_mapping(images, points3d)
    total_pairs = sum(len(v) for v in mapping.values())
    print(f"  2D-3D 대응점 합계: {total_pairs}개")

    return cameras, images, points3d, mapping
