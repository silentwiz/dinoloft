"""
fine_matching.py
Phase 3: Fine 매칭 모듈 (LoFTR 기반 픽셀 단위 2D-2D 매칭)

주요 기능:
  - kornia LoFTR 모델 로드 (indoor 사전학습 가중치)
  - 쿼리 이미지 ↔ Top-K DB 이미지 쌍별 매칭 실행
  - 매칭된 2D 좌표(query_pts, db_pts) + 신뢰도(confidence) 반환

EXIF 처리 방침:
  - 쿼리 이미지: EXIF 보정 후 LoFTR 입력 → 매칭 품질 향상
    출력 query_pts는 raw 픽셀 좌표로 역변환 (COLMAP/PnP 일관성 유지)
  - DB 이미지: EXIF 미적용 (COLMAP raw 좌표계와 일치)
  - 시각화용 query_pts_display: EXIF 보정 공간 좌표 (show_fine 전용)

사용 예시:
  from fine_matching import FineMatcher

  matcher = FineMatcher()
  matches = matcher.match("./query/IMG_3342.JPG",
                          ["./dataset/IMG_3342.JPG", "./dataset/IMG_3341.JPG"])
  # matches[0] = {
  #     "db_name"          : "IMG_3342.JPG",
  #     "query_pts"        : np.ndarray [M, 2],  # raw 픽셀 좌표 (PnP용)
  #     "query_pts_display": np.ndarray [M, 2],  # EXIF 보정 좌표 (시각화용)
  #     "db_pts"           : np.ndarray [M, 2],  # raw 픽셀 좌표
  #     "confidence"       : np.ndarray [M],
  # }
"""

import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image, ImageOps

try:
    from kornia.feature import LoFTR
except ImportError:
    raise ImportError("kornia가 설치되어 있지 않습니다.\n  pip install kornia")


# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
LOFTR_WEIGHTS  = "indoor"    # "indoor" | "outdoor"
CONF_THRESHOLD = 0.5         # 이 값 미만 매칭 제거
RESIZE_LONG    = 1024        # LoFTR 입력 긴 변 기준 리사이즈 (8의 배수)
                             # 640(기본) → 1024: 4K 입력 시 세부 특징 보존 향상
                             # 메모리 부족 시 832 또는 640으로 낮출 것


# ──────────────────────────────────────────────
# EXIF 방향 유틸 (모듈 레벨)
# ──────────────────────────────────────────────
def _get_exif_orientation(path: str) -> int:
    """
    이미지 파일의 EXIF Orientation 태그 값 반환.
    태그 없거나 오류 시 1 (정상) 반환.

    주요 값:
      1 = 정상 (변환 불필요)
      3 = 180° 회전
      6 = 90° CW 회전 필요 (iPhone 포트레이트 일반적)
      8 = 90° CCW 회전 필요
    """
    try:
        from PIL.ExifTags import TAGS
        img = Image.open(path)
        exif_data = img._getexif()
        if exif_data is None:
            return 1
        for tag_id, val in exif_data.items():
            if TAGS.get(tag_id) == "Orientation":
                return int(val)
    except Exception:
        pass
    return 1


def _pts_exif_to_raw(pts: np.ndarray, orientation: int,
                     exif_wh: tuple[int, int]) -> np.ndarray:
    """
    EXIF 보정 공간 좌표 → raw 픽셀 좌표 역변환.

    Args:
        pts          : [M, 2] EXIF 보정 좌표 (x, y)
        orientation  : EXIF orientation tag 값
        exif_wh      : EXIF 보정 이미지 크기 (W, H)

    Returns:
        [M, 2] raw 픽셀 좌표
    """
    if len(pts) == 0 or orientation == 1:
        return pts.copy()

    ex = pts[:, 0].copy()
    ey = pts[:, 1].copy()
    eW, eH = exif_wh

    if orientation == 3:    # 180° — raw와 exif 크기 동일
        return np.column_stack([eW - 1 - ex, eH - 1 - ey])

    elif orientation == 6:  # 90° CW 표시 (raw=landscape, exif=portrait)
        # forward: raw(x,y) → exif(H_raw-1-y, x),  exif size = (H_raw, W_raw)
        # inverse: exif(ex,ey) → raw(ey, eW-1-ex)   (eW = H_raw)
        return np.column_stack([ey, eW - 1 - ex])

    elif orientation == 8:  # 90° CCW 표시 (raw=landscape, exif=portrait)
        # forward: raw(x,y) → exif(y, W_raw-1-x),  exif size = (H_raw, W_raw)
        # inverse: exif(ex,ey) → raw(eH-1-ey, ex)  (eH = W_raw)
        return np.column_stack([eH - 1 - ey, ex])

    # 그 외 (flip 계열 등): 미지원, 원본 반환
    return pts.copy()


def _pts_raw_to_exif(pts: np.ndarray, orientation: int,
                     raw_wh: tuple[int, int]) -> np.ndarray:
    """
    raw 픽셀 좌표 → EXIF 보정 공간 좌표 순방향 변환 (시각화 전용).

    Args:
        pts         : [M, 2] raw 좌표 (x, y)
        orientation : EXIF orientation tag 값
        raw_wh      : raw 이미지 크기 (W, H)

    Returns:
        [M, 2] EXIF 보정 공간 좌표
    """
    if len(pts) == 0 or orientation == 1:
        return pts.copy()

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    W, H = raw_wh

    if orientation == 3:    # 180°
        return np.column_stack([W - 1 - x, H - 1 - y])
    elif orientation == 6:  # raw(x,y) → exif(H-1-y, x)
        return np.column_stack([H - 1 - y, x])
    elif orientation == 8:  # raw(x,y) → exif(y, W-1-x)
        return np.column_stack([y, W - 1 - x])

    return pts.copy()


class FineMatcher:
    """kornia LoFTR 기반 2D-2D 픽셀 매칭기."""

    def __init__(self,
                 weights: str = LOFTR_WEIGHTS,
                 conf_thr: float = CONF_THRESHOLD,
                 resize_long: int = RESIZE_LONG):
        # 디바이스 선택
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[FineMatcher] 디바이스: {self.device}")

        self.conf_thr   = conf_thr
        self.resize_long = resize_long

        # LoFTR 모델 로드
        self.matcher = LoFTR(pretrained=weights)
        self.matcher.eval().to(self.device)
        print(f"[FineMatcher] LoFTR ({weights}) 로드 완료")

    # ──────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────
    def _load_gray(self, img_path: str,
                   apply_exif: bool = False
                   ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """
        이미지를 그레이스케일 Tensor [1, 1, H, W]로 로드.
        긴 변을 RESIZE_LONG에 맞게 리사이즈 (8의 배수 보정).

        Args:
            apply_exif : True이면 EXIF 보정 후 리사이즈

        Returns:
            tensor       : [1, 1, H, W]
            corrected_hw : EXIF 보정 후 원본 크기 (H, W) — _scale_pts 기준
            raw_hw       : EXIF 미적용 원본 크기   (H, W) — K 스케일링 기준
        """
        img = Image.open(img_path)
        raw_w, raw_h = img.size   # raw 원본 크기

        if apply_exif:
            img = ImageOps.exif_transpose(img)

        img = img.convert("L")
        orig_w, orig_h = img.size  # EXIF 보정 후 크기 (apply_exif=False면 raw와 동일)

        # 긴 변 기준 스케일 계산 (업스케일 금지 — LoFTR은 업스케일 시 품질 저하)
        scale = min(1.0, self.resize_long / max(orig_w, orig_h))
        new_w = max(8, int(orig_w * scale) // 8 * 8)
        new_h = max(8, int(orig_h * scale) // 8 * 8)
        img   = img.resize((new_w, new_h), Image.BILINEAR)

        tensor = T.ToTensor()(img).unsqueeze(0).to(self.device)   # [1, 1, H, W]
        return tensor, (orig_h, orig_w), (raw_h, raw_w)

    def _scale_pts(self, pts: np.ndarray,
                   resized_hw: tuple[int, int],
                   orig_hw: tuple[int, int]) -> np.ndarray:
        """리사이즈된 좌표를 원본 이미지 좌표계로 역변환."""
        scale_x = orig_hw[1] / resized_hw[1]
        scale_y = orig_hw[0] / resized_hw[0]
        pts_orig = pts.copy().astype(np.float32)
        pts_orig[:, 0] *= scale_x
        pts_orig[:, 1] *= scale_y
        return pts_orig

    # ──────────────────────────────────────────
    # 메인 인터페이스
    # ──────────────────────────────────────────
    def match_pair(self, query_path: str,
                   db_path: str,
                   debug_mode: int = 0,
                   dataset_path: str = "./dataset") -> dict:
        """
        쿼리-DB 이미지 한 쌍 매칭.

        쿼리 이미지 EXIF 처리:
          - LoFTR 입력: EXIF 보정 이미지 사용 (바로 선 이미지로 더 좋은 매칭)
          - 출력 query_pts: raw 픽셀 좌표로 역변환 (COLMAP/PnP 일관성)
          - query_pts_display: EXIF 보정 좌표 (show_fine 시각화 전용)

        Args:
            debug_mode   : 1이면 매칭 선 시각화 출력
            dataset_path : DB 이미지 루트 폴더 (시각화 시 사용)

        Returns:
            {
                "db_name"          : str,
                "query_pts"        : np.ndarray [M, 2],  # raw 픽셀 좌표 (PnP/KD-Tree용)
                "query_pts_display": np.ndarray [M, 2],  # EXIF 보정 좌표 (시각화용)
                "db_pts"           : np.ndarray [M, 2],  # raw 픽셀 좌표
                "confidence"       : np.ndarray [M],
                "query_orig_wh"    : (W, H),             # raw 크기 (K 스케일링용)
                "query_exif_orient": int,                # EXIF orientation 값
            }
        """
        # 쿼리: EXIF 보정 적용 (매칭 품질 향상)
        q_orient = _get_exif_orientation(query_path)
        q_tensor, q_corr_hw, q_raw_hw = self._load_gray(query_path, apply_exif=(q_orient != 1))

        # DB: EXIF 미적용 (COLMAP raw 좌표계와 일치)
        d_tensor, d_corr_hw, d_raw_hw = self._load_gray(db_path, apply_exif=False)

        batch = {"image0": q_tensor, "image1": d_tensor}

        with torch.no_grad():
            out = self.matcher(batch)

        q_pts  = out["keypoints0"].cpu().numpy()    # [M, 2]  리사이즈 좌표
        db_pts = out["keypoints1"].cpu().numpy()    # [M, 2]
        conf   = out["confidence"].cpu().numpy()    # [M]

        # 신뢰도 필터링
        mask   = conf >= self.conf_thr
        q_pts  = q_pts[mask]
        db_pts = db_pts[mask]
        conf   = conf[mask]

        # 리사이즈 좌표 → 원본 좌표계 복원
        resized_qhw = (q_tensor.shape[2], q_tensor.shape[3])
        resized_dhw = (d_tensor.shape[2], d_tensor.shape[3])

        # 쿼리: EXIF 보정 공간 원본 좌표 (시각화용)
        q_pts_exif = self._scale_pts(q_pts, resized_qhw, q_corr_hw)

        # 쿼리: raw 픽셀 좌표로 역변환 (PnP/COLMAP용)
        q_pts_raw = _pts_exif_to_raw(q_pts_exif, q_orient, (q_corr_hw[1], q_corr_hw[0]))

        # DB: raw 픽셀 좌표 (EXIF 미적용이므로 그대로)
        db_pts_orig = self._scale_pts(db_pts, resized_dhw, d_raw_hw)

        if q_orient != 1:
            print(f"  [FineMatcher] 쿼리 EXIF orientation={q_orient} 보정 적용")

        result = {
            "db_name":           Path(db_path).name,
            "query_pts":         q_pts_raw,                        # raw 좌표 (PnP용)
            "query_pts_display": q_pts_exif,                       # EXIF 보정 좌표 (시각화용)
            "db_pts":            db_pts_orig,                      # raw 좌표
            "confidence":        conf,
            "query_orig_wh":     (q_raw_hw[1], q_raw_hw[0]),       # raw (W, H) — K 스케일링용
            "query_exif_orient": q_orient,
        }

        if debug_mode:
            from visualizer import show_fine
            show_fine(query_path, result, dataset_path=dataset_path)

        return result

    def match(self, query_path: str,
              db_paths: list[str],
              debug_mode: int = 0,
              dataset_path: str = "./dataset") -> list[dict]:
        """
        쿼리 ↔ Top-K DB 이미지 전체 매칭.

        Args:
            query_path   : 쿼리 이미지 경로
            db_paths     : DB 이미지 경로 리스트 (Top-K)
            debug_mode   : 1이면 각 쌍마다 매칭 시각화 출력
            dataset_path : DB 이미지 루트 폴더 (시각화 시 사용)

        Returns:
            [match_pair_result, ...] 입력 순서 유지
        """
        results = []
        for db_path in db_paths:
            res = self.match_pair(query_path, db_path,
                                  debug_mode=debug_mode,
                                  dataset_path=dataset_path)
            results.append(res)
            print(f"  [{Path(db_path).name}] 매칭 {len(res['query_pts'])}개 "
                  f"(conf≥{self.conf_thr})")
        return results


# ──────────────────────────────────────────────
# 단독 실행 (간단 테스트)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query_path = sys.argv[1] if len(sys.argv) > 1 else "./query/IMG_3342.JPG"
    db_path    = sys.argv[2] if len(sys.argv) > 2 else "./dataset/IMG_3342.JPG"

    matcher = FineMatcher()
    res     = matcher.match_pair(query_path, db_path)

    print(f"\n[결과]")
    print(f"  매칭 수  : {len(res['query_pts'])}")
    print(f"  conf 범위: {res['confidence'].min():.3f} ~ {res['confidence'].max():.3f}")
    if len(res['query_pts']) > 0:
        print(f"  예시 쌍  : query={res['query_pts'][0]}  db={res['db_pts'][0]}")
