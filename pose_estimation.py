"""
pose_estimation.py
Phase 4: 2D-3D 연결 및 PnP 최적화

주요 기능:
  - KD-Tree로 LoFTR DB 매칭점 ↔ COLMAP 2D-3D LUT 연결 (반경 2px)
  - cv2.solvePnPRansac() → rvec, tvec (6DoF)
  - 카메라 내부 파라미터 K를 COLMAP cameras 정보에서 자동 추출

사용 예시:
  from pose_estimation import PoseEstimator

  estimator = PoseEstimator(sfm_db_path="./sfm_output/sfm_db.pkl")
  pose = estimator.estimate(
      match_result,       # fine_matching.py의 match_pair() 결과 dict
      db_img_name,        # "IMG_3342.JPG"
      query_camera_id=1,  # COLMAP camera id (쿼리와 DB가 같은 카메라면 동일)
  )
  # pose = {
  #     "rvec"   : np.ndarray [3, 1],
  #     "tvec"   : np.ndarray [3, 1],
  #     "inliers": np.ndarray [K],        # RANSAC 생존 인덱스
  #     "kd_pts_query" : np.ndarray [N,2],  # KD-Tree 통과한 쿼리 2D
  #     "kd_pts_3d"    : np.ndarray [N,3],  # 대응 3D 좌표
  # }
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial import KDTree


# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
SFM_DB_PATH    = "./sfm_output/sfm_db.pkl"
KD_RADIUS_PX   = 4.0    # KD-Tree 탐색 반경 (픽셀) — LoFTR semi-dense 오차 여유
RANSAC_REPROJ  = 4.0    # PnP RANSAC reprojection error threshold (픽셀)
RANSAC_ITER    = 1000   # RANSAC 최대 반복 횟수

# match_method 선택
#   "kd_tree"     : 반경(kd_radius) 이내 COLMAP 점만 허용 (엄격, 저밀도 DB에서 낮은 통과율)
#   "brute_force" : 최근접 COLMAP 점을 반경 제한 없이 연결 → RANSAC이 outlier 처리
#                   semi-dense LoFTR + sparse COLMAP 조합에서 권장
MATCH_METHOD   = "brute_force"
MAX_NN_DIST_PX = None   # brute_force 수동 거리 임계값 (None=자동 계산)
NN_DIST_FACTOR = 2.0    # 자동 임계값 = factor × COLMAP 점간 median NN 거리
                        # 근거: Poisson 공간 과정에서 2× E[d_NN] 이상은
                        #       다른 구조물의 SIFT 점일 가능성이 높음

# ──────────────────────────────────────────────
# 쿼리 카메라 내부 파라미터 (직접 설정)
# ──────────────────────────────────────────────
# QUERY_CAMERA_K        : 기준 해상도에서의 3×3 K 행렬 (쿼리 카메라 전용)
# QUERY_CAMERA_K_WH     : K를 캘리브레이션한 기준 해상도 (width, height)
#
# ※ K 스케일링 구조:
#   LoFTR은 내부적으로 640px로 리사이즈 후 매칭 → 결과 좌표를 원본 해상도로 역스케일
#   → query_pts는 쿼리 원본 해상도(예: 4K) 기준 2D 좌표
#   → db_pts는 DB 원본 해상도(예: 1080p) 기준 → KD-Tree 탐색에만 사용, K와 무관
#   → PnP의 K는 query_pts 좌표계(쿼리 원본 해상도)와 일치해야 함
#   → 쿼리 해상도 ≠ 기준 해상도인 경우 fx/fy/cx/cy를 선형 스케일링
#
# ※ 4K 사진 사용 시: iPhone 동일 광학계 → 선형 스케일 가정 유효
#   더 정밀한 값이 필요하면 실제 ARKit currentFrame.camera.intrinsics (4K 모드)로 교체
#
# iPhone 13 Pro — ARKit currentFrame.camera.intrinsics 기준값
# 기준 해상도: 1920×1440 (ARKit 1440p 모드)
QUERY_CAMERA_K = np.array([
    [1450.0,    0.0,  960.0],
    [   0.0, 1450.0,  720.0],
    [   0.0,    0.0,    1.0],
], dtype=np.float64)
QUERY_CAMERA_K_WH = (1920, 1440)  # K를 보정한 기준 해상도 (width, height)


# ──────────────────────────────────────────────
# 카메라 모델 → K 행렬 변환
# ──────────────────────────────────────────────
def camera_to_K(camera: dict) -> np.ndarray:
    """
    COLMAP camera dict → 3×3 내부 파라미터 행렬 K.

    지원 모델:
      0 SIMPLE_PINHOLE : [f, cx, cy]
      1 PINHOLE        : [fx, fy, cx, cy]
      2 SIMPLE_RADIAL  : [f, cx, cy, k]   → k 무시
      3 RADIAL         : [f, cx, cy, k1, k2] → k 무시
      4 OPENCV         : [fx, fy, cx, cy, ...]
      기타             : fx=fy=params[0], cx=params[1], cy=params[2]
    """
    model_id = camera["model_id"]
    p        = camera["params"]

    if model_id == 0 or model_id == 2 or model_id == 3:
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f,  0, cx],
                      [0,  f, cy],
                      [0,  0,  1]], dtype=np.float64)
    elif model_id == 1 or model_id == 4:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        K = np.array([[fx,  0, cx],
                      [ 0, fy, cy],
                      [ 0,  0,  1]], dtype=np.float64)
    else:
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f,  0, cx],
                      [0,  f, cy],
                      [0,  0,  1]], dtype=np.float64)
    return K


class PoseEstimator:
    """KD-Tree + PnP RANSAC 기반 6DoF 포즈 추정기."""

    def __init__(self, sfm_db_path: str = SFM_DB_PATH):
        if not Path(sfm_db_path).exists():
            raise FileNotFoundError(
                f"SFM DB를 찾을 수 없습니다: {sfm_db_path}\n"
                "먼저 build_db_claude.py 를 실행해 주세요."
            )
        with open(sfm_db_path, "rb") as f:
            db = pickle.load(f)

        self.cameras  = db["cameras"]   # {camera_id: {...}}
        self.images   = db["images"]    # {image_id: {...}}
        self.mapping  = db["mapping"]   # {img_name: [{"2d":..., "xyz":...}]}
        self.points3d = db["points3d"]  # {point3d_id: {"xyz":..., "rgb":...}}
        print(f"[PoseEstimator] SFM DB 로드 완료 ({len(self.mapping)}개 이미지)")

    # ──────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────
    def _build_kd_tree(self, db_img_name: str) -> tuple[KDTree, list, float]:
        """
        DB 이미지의 COLMAP 2D-3D 맵핑 엔트리로 KD-Tree 구성.

        Returns:
            (tree, entries, median_nn_dist)
              entries        : [{"2d":(x,y), "xyz":array}, ...]
              median_nn_dist : COLMAP 점들 간 최근접 거리의 중앙값 (px)
                               brute_force 자동 임계값 계산에 사용
        """
        entries = self.mapping.get(db_img_name, [])
        if not entries:
            raise ValueError(f"매핑 데이터 없음: {db_img_name}")

        # NaN / Inf 포함된 엔트리 제거
        clean_entries = [
            e for e in entries
            if np.all(np.isfinite(e["2d"])) and np.all(np.isfinite(e["xyz"]))
        ]
        if not clean_entries:
            raise ValueError(f"유효한 2D-3D 대응점 없음: {db_img_name}")

        pts2d = np.array([e["2d"] for e in clean_entries], dtype=np.float32)
        tree  = KDTree(pts2d)

        # COLMAP 점들 간 실제 NN 거리 분포 계산
        # k=2: 자기 자신(거리=0)을 제외한 최근접 이웃
        nn_dists, _ = tree.query(pts2d, k=min(2, len(pts2d)))
        if nn_dists.ndim == 2:
            nn_dists = nn_dists[:, -1]   # k=2이면 두 번째 열 = 자신 제외 최근접
        median_nn_dist = float(np.median(nn_dists))

        return tree, clean_entries, median_nn_dist

    def _get_camera_K(self, db_img_name: str) -> np.ndarray:
        """DB 이미지 이름으로 대응 카메라 K 행렬 반환."""
        for img in self.images.values():
            if img["name"] == db_img_name:
                cam_id = img["camera_id"]
                return camera_to_K(self.cameras[cam_id])
        # fallback: 첫 번째 카메라 사용
        cam_id = list(self.cameras.keys())[0]
        return camera_to_K(self.cameras[cam_id])

    # ──────────────────────────────────────────
    # 메인 인터페이스
    # ──────────────────────────────────────────
    def estimate(self, match_result: dict,
                 kd_radius: float = KD_RADIUS_PX,
                 match_method: str = MATCH_METHOD,
                 max_nn_dist: float | None = MAX_NN_DIST_PX,
                 nn_dist_factor: float = NN_DIST_FACTOR,
                 ransac_reproj: float = RANSAC_REPROJ,
                 ransac_iter: int = RANSAC_ITER,
                 query_K: np.ndarray | None = None,
                 query_K_wh: tuple | None = None,
                 debug_mode: int = 0,
                 query_path: str = "",
                 dataset_path: str = "./dataset") -> dict | None:
        """
        LoFTR 매칭 결과 + COLMAP LUT → 6DoF 포즈 추정.

        Args:
            match_result : fine_matching.match_pair() 반환값
                           {"db_name", "query_pts", "db_pts", "confidence"}
            kd_radius    : match_method="kd_tree" 시 탐색 반경 (픽셀)
            match_method   : "kd_tree"     — 반경 이내 COLMAP 점만 허용 (엄격)
                             "brute_force" — 최근접 COLMAP 점 연결, 임계값으로 필터
                                             (LoFTR semi-dense + sparse COLMAP 조합 권장)
            max_nn_dist    : brute_force 수동 거리 임계값 (None=자동 계산)
            nn_dist_factor : 자동 임계값 배율. threshold = factor × median(COLMAP_NN_dist)
                             기본 2.0: Poisson 공간 과정에서 2× E[d_NN] 이상은
                             다른 구조물의 SIFT 점일 가능성이 높음
            debug_mode   : 1이면 KD-Tree/RANSAC 점 시각화 출력
            dataset_path : DB 이미지 루트 폴더 (시각화 시 사용)

        Returns:
            성공 시:
              {
                "rvec"         : np.ndarray [3,1],
                "tvec"         : np.ndarray [3,1],
                "inliers"      : np.ndarray [K],
                "kd_pts_query" : np.ndarray [N,2],
                "kd_pts_3d"    : np.ndarray [N,3],
                "num_kd"       : int,   # 2D-3D 연결 성공 개수
                "num_inliers"  : int,   # RANSAC 생존 개수
              }
            실패 시: None
        """
        db_name  = match_result["db_name"]
        q_pts    = match_result["query_pts"]   # [M, 2]  raw 좌표 (기본)
        db_pts   = match_result["db_pts"]      # [M, 2]

        if len(q_pts) < 4:
            print(f"  [PoseEstimator] 매칭 수 부족 ({len(q_pts)}개) → 스킵")
            return None

        # Step 1: KD-Tree 구성 및 필터링
        tree, entries, median_nn_dist = self._build_kd_tree(db_name)

        # 쿼리 카메라 K 결정
        #   - query_K가 명시적으로 전달된 경우: 해당 K를 쿼리 해상도에 맞게 스케일
        #   - 전달 안 된 경우: COLMAP이 DB에서 추정한 K를 쿼리 해상도에 맞게 스케일
        if query_K is not None:
            K_ref = query_K
            ref_wh = query_K_wh if query_K_wh is not None else QUERY_CAMERA_K_WH
            ref_w, ref_h = ref_wh
        else:
            K_ref = self._get_camera_K(db_name)
            db_cam_id = next(
                (img["camera_id"] for img in self.images.values() if img["name"] == db_name),
                list(self.cameras.keys())[0]
            )
            db_cam = self.cameras[db_cam_id]
            ref_w  = db_cam["width"]
            ref_h  = db_cam["height"]

        # ── EXIF orientation 처리 ───────────────────────────────────
        # EXIF=6(90° CW) / EXIF=8(90° CCW): raw는 landscape, COLMAP DB는 portrait
        # PnP 2D 좌표와 K 행렬을 같은 좌표계(portrait)로 통일해야 회전 오차 없음
        #
        # portrait K 스케일 원리 (EXIF=6 예시):
        #   raw: W=4032, H=3024   portrait: W=3024, H=4032
        #   sx = W_portrait / H_ref = 3024/1440 = 2.1   (ref의 H축으로 스케일)
        #   sy = H_portrait / W_ref = 4032/1920 = 2.1   (ref의 W축으로 스케일)
        #   → fx=fy=3045, cx=cy_ref×2.1=1512, cy=cx_ref×2.1=2016  ✓
        q_orient = match_result.get("query_exif_orient", 1)
        raw_w, raw_h = match_result.get("query_orig_wh", (ref_w, ref_h))

        if q_orient in (6, 8):
            # EXIF 보정 좌표 (portrait) 사용 → COLMAP portrait 좌표계와 일치
            q_pts   = match_result.get("query_pts_display", q_pts)
            query_w = raw_h   # portrait width  = raw height
            query_h = raw_w   # portrait height = raw width
            # K: ref의 W↔H 축 교환 후 스케일
            sx = query_w / ref_h
            sy = query_h / ref_w
            K_mat = K_ref.copy()
            K_mat[0, 0] = K_ref[1, 1] * sx   # fx_portrait = fy_ref × sx
            K_mat[1, 1] = K_ref[0, 0] * sy   # fy_portrait = fx_ref × sy
            K_mat[0, 2] = K_ref[1, 2] * sx   # cx_portrait = cy_ref × sx
            K_mat[1, 2] = K_ref[0, 2] * sy   # cy_portrait = cx_ref × sy
            orient_note = " [portrait보정]"
        else:
            query_w, query_h = raw_w, raw_h
            sx = query_w / ref_w
            sy = query_h / ref_h
            K_mat = K_ref.copy()
            K_mat[0, 0] *= sx   # fx
            K_mat[1, 1] *= sy   # fy
            K_mat[0, 2] *= sx   # cx
            K_mat[1, 2] *= sy   # cy
            orient_note = ""

        print(f"  [PoseEstimator] 쿼리 K (해상도 {query_w}×{query_h}{orient_note}): "
              f"fx={K_mat[0,0]:.1f} fy={K_mat[1,1]:.1f} "
              f"cx={K_mat[0,2]:.1f} cy={K_mat[1,2]:.1f}")

        pts_query_list = []
        pts_3d_list    = []
        nn_dist_list   = []   # 진단용: 각 대응의 nearest-neighbor 거리

        if match_method == "brute_force":
            # ── Brute Force: 자동 임계값으로 최근접 COLMAP 점 연결 ──────────
            # 임계값 = nn_dist_factor × median(COLMAP 점간 NN 거리)
            # 근거: Poisson 공간 과정에서 2× E[d_NN] 이상이면 다른 구조물의
            #       SIFT 점일 가능성이 높아 기하학적으로 무의미한 대응이 됨
            if max_nn_dist is not None:
                effective_threshold = max_nn_dist          # 수동 오버라이드
                thresh_note = f"수동 설정"
            else:
                effective_threshold = nn_dist_factor * median_nn_dist  # 자동 계산
                thresh_note = f"median_NN({median_nn_dist:.1f}px) × {nn_dist_factor}"
            print(f"  [PoseEstimator] NN 임계값: {effective_threshold:.1f}px  ({thresh_note})")

            for q_pt, db_pt in zip(q_pts, db_pts):
                dist, idx = tree.query(db_pt, k=1)
                if dist > effective_threshold:
                    continue
                pts_query_list.append(q_pt)
                pts_3d_list.append(entries[idx]["xyz"])
                nn_dist_list.append(dist)
        else:
            # ── KD-Tree: 반경 이내 COLMAP 점만 허용 ─────────────────────
            for q_pt, db_pt in zip(q_pts, db_pts):
                idxs = tree.query_ball_point(db_pt, r=kd_radius)
                if not idxs:
                    continue
                dists = [np.linalg.norm(np.array(entries[i]["2d"]) - db_pt) for i in idxs]
                best  = idxs[int(np.argmin(dists))]
                pts_query_list.append(q_pt)
                pts_3d_list.append(entries[best]["xyz"])
                nn_dist_list.append(dists[int(np.argmin(dists))])

        num_kd = len(pts_query_list)
        method_label = "BruteForce NN" if match_method == "brute_force" else "KD-Tree"
        print(f"  [PoseEstimator] {method_label} 연결: {num_kd}개 / {len(q_pts)}개", end="")
        if nn_dist_list:
            arr = np.array(nn_dist_list)
            print(f"  (NN 거리 median={np.median(arr):.1f}px  max={arr.max():.1f}px)")
        else:
            print()

        # ── 진단 출력 ──────────────────────────────────────────
        if num_kd == 0 or (match_method == "kd_tree" and num_kd < 10 and len(q_pts) > 50):
            colmap_pts = np.array([e["2d"] for e in entries], dtype=np.float32)
            print(f"  [진단] COLMAP 2D 점 수       : {len(entries)}")
            print(f"  [진단] COLMAP x 범위         : {colmap_pts[:,0].min():.1f} ~ {colmap_pts[:,0].max():.1f}")
            print(f"  [진단] COLMAP y 범위         : {colmap_pts[:,1].min():.1f} ~ {colmap_pts[:,1].max():.1f}")
            print(f"  [진단] LoFTR db_pts x 범위   : {db_pts[:,0].min():.1f} ~ {db_pts[:,0].max():.1f}")
            print(f"  [진단] LoFTR db_pts y 범위   : {db_pts[:,1].min():.1f} ~ {db_pts[:,1].max():.1f}")
            print(f"  [진단] KD-Tree 반경           : {kd_radius}px")
            print(f"  → match_method='brute_force' 로 전환하면 COLMAP 밀도 불일치 우회 가능")
        # ────────────────────────────────────────────────────────

        if num_kd < 4:
            print(f"  [PoseEstimator] KD-Tree 통과 수 부족 → PnP 스킵")
            return None

        pts_query = np.array(pts_query_list, dtype=np.float64)   # [N, 2]
        pts_3d    = np.array(pts_3d_list,    dtype=np.float64)   # [N, 3]

        # Step 2: PnP RANSAC
        dist_coeffs = np.zeros(4, dtype=np.float64)   # COLMAP은 왜곡 보정 완료
        pnp_kwargs = dict(
            objectPoints      = pts_3d,
            imagePoints       = pts_query,
            cameraMatrix      = K_mat,
            distCoeffs        = dist_coeffs,
            reprojectionError = ransac_reproj,
            iterationsCount   = ransac_iter,
        )

        # 1차: ITERATIVE (정밀도 우선)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            **pnp_kwargs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # 2차 폴백: EPNP (초기값 불필요, outlier 비율 높아도 안정적)
        if not success or inliers is None:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                **pnp_kwargs, flags=cv2.SOLVEPNP_EPNP
            )
            if success and inliers is not None:
                print(f"  [PoseEstimator] EPNP 폴백으로 성공")

        if not success or inliers is None:
            print(f"  [PoseEstimator] PnP RANSAC 실패 (ITERATIVE + EPNP 모두 실패)")
            return None

        inliers = inliers.flatten()
        print(f"  [PoseEstimator] RANSAC 생존: {len(inliers)}개 / {num_kd}개")

        result = {
            "rvec"         : rvec,
            "tvec"         : tvec,
            "inliers"      : inliers,
            "kd_pts_query" : pts_query,
            "kd_pts_3d"    : pts_3d,
            "num_kd"       : num_kd,
            "num_inliers"  : len(inliers),
        }

        if debug_mode:
            from visualizer import show_pnp
            show_pnp(query_path, result, dataset_path=dataset_path)

        return result

    # ──────────────────────────────────────────
    # 통합 PnP (Top-K 전체 합산)
    # ──────────────────────────────────────────
    def estimate_merged(self,
                        match_results: list[dict],
                        kd_radius: float = KD_RADIUS_PX,
                        match_method: str = MATCH_METHOD,
                        max_nn_dist: float | None = MAX_NN_DIST_PX,
                        nn_dist_factor: float = NN_DIST_FACTOR,
                        ransac_reproj: float = RANSAC_REPROJ,
                        ransac_iter: int = RANSAC_ITER,
                        query_K: np.ndarray | None = None,
                        query_K_wh: tuple | None = None,
                        debug_mode: int = 0,
                        query_path: str = "",
                        dataset_path: str = "./dataset") -> dict | None:
        """
        Top-K 전체 match_results의 2D-3D 대응쌍을 합산한 뒤 PnP RANSAC 1회 실행.

        per-image PnP 대비 장점:
          - 더 넓은 3D 포인트 분포 → RANSAC 안정성 향상
          - Top-K 이미지가 서로 다른 시점의 3D 점을 커버 → 기하학적 다양성 증가
          - 한 이미지에서 매칭이 적어도 다른 이미지가 보완

        Args:
            match_results : fine_matching.match() 반환값 리스트
            (나머지 파라미터는 estimate()와 동일)

        Returns:
            estimate()와 동일한 구조 + "db_names": 기여한 DB 이미지 목록
        """
        all_query_pts = []
        all_pts_3d    = []
        db_names_used = []
        K_mat         = None

        print(f"\n[PoseEstimator] 통합 PnP: {len(match_results)}개 이미지 대응쌍 합산 중...")

        for match_result in match_results:
            db_name = match_result["db_name"]
            q_pts   = match_result["query_pts"]
            db_pts  = match_result["db_pts"]

            if len(q_pts) < 1:
                continue

            try:
                tree, entries, median_nn_dist = self._build_kd_tree(db_name)
            except ValueError as e:
                print(f"  [skip] {db_name}: {e}")
                continue

            # K 행렬은 첫 번째 유효 이미지에서 한 번만 계산 (모든 이미지가 같은 쿼리 카메라)
            if K_mat is None:
                if query_K is not None:
                    K_ref = query_K
                    ref_w, ref_h = (query_K_wh if query_K_wh is not None
                                    else QUERY_CAMERA_K_WH)
                else:
                    K_ref  = self._get_camera_K(db_name)
                    db_cam_id = next(
                        (img["camera_id"] for img in self.images.values()
                         if img["name"] == db_name),
                        list(self.cameras.keys())[0]
                    )
                    db_cam = self.cameras[db_cam_id]
                    ref_w, ref_h = db_cam["width"], db_cam["height"]

                q_orient = match_result.get("query_exif_orient", 1)
                raw_w, raw_h = match_result.get("query_orig_wh", (ref_w, ref_h))

                if q_orient in (6, 8):
                    q_pts_for_pnp_key = "query_pts_display"
                    query_w, query_h  = raw_h, raw_w
                    sx = query_w / ref_h
                    sy = query_h / ref_w
                    K_mat = K_ref.copy()
                    K_mat[0, 0] = K_ref[1, 1] * sx
                    K_mat[1, 1] = K_ref[0, 0] * sy
                    K_mat[0, 2] = K_ref[1, 2] * sx
                    K_mat[1, 2] = K_ref[0, 2] * sy
                    orient_note = " [portrait보정]"
                else:
                    q_pts_for_pnp_key = "query_pts"
                    query_w, query_h  = raw_w, raw_h
                    sx = query_w / ref_w
                    sy = query_h / ref_h
                    K_mat = K_ref.copy()
                    K_mat[0, 0] *= sx; K_mat[1, 1] *= sy
                    K_mat[0, 2] *= sx; K_mat[1, 2] *= sy
                    orient_note = ""

                print(f"  [PoseEstimator] 쿼리 K (해상도 {query_w}×{query_h}{orient_note}): "
                      f"fx={K_mat[0,0]:.1f} fy={K_mat[1,1]:.1f} "
                      f"cx={K_mat[0,2]:.1f} cy={K_mat[1,2]:.1f}")
            else:
                q_orient = match_result.get("query_exif_orient", 1)
                q_pts_for_pnp_key = "query_pts_display" if q_orient in (6, 8) else "query_pts"

            q_pts = match_result.get(q_pts_for_pnp_key, match_result["query_pts"])

            # 2D-3D 연결
            pts_query_list, pts_3d_list, nn_dist_list = [], [], []

            if match_method == "brute_force":
                if max_nn_dist is not None:
                    effective_threshold = max_nn_dist
                else:
                    effective_threshold = nn_dist_factor * median_nn_dist

                for q_pt, db_pt in zip(q_pts, db_pts):
                    dist, idx = tree.query(db_pt, k=1)
                    if dist > effective_threshold:
                        continue
                    pts_query_list.append(q_pt)
                    pts_3d_list.append(entries[idx]["xyz"])
                    nn_dist_list.append(dist)
            else:
                for q_pt, db_pt in zip(q_pts, db_pts):
                    idxs = tree.query_ball_point(db_pt, r=kd_radius)
                    if not idxs:
                        continue
                    dists = [np.linalg.norm(np.array(entries[i]["2d"]) - db_pt)
                             for i in idxs]
                    best = idxs[int(np.argmin(dists))]
                    pts_query_list.append(q_pt)
                    pts_3d_list.append(entries[best]["xyz"])
                    nn_dist_list.append(dists[int(np.argmin(dists))])

            method_label = "BruteForce NN" if match_method == "brute_force" else "KD-Tree"
            n_matched = len(pts_query_list)
            print(f"  [{db_name}] {method_label}: {n_matched}개 / {len(q_pts)}개", end="")
            if nn_dist_list:
                arr = np.array(nn_dist_list)
                print(f"  (median={np.median(arr):.1f}px)")
            else:
                print()

            if n_matched > 0:
                all_query_pts.extend(pts_query_list)
                all_pts_3d.extend(pts_3d_list)
                db_names_used.append(db_name)

        total = len(all_query_pts)
        print(f"  [PoseEstimator] 합산 총 대응쌍: {total}개 ({len(db_names_used)}개 이미지)")

        if total < 4:
            print(f"  [PoseEstimator] 대응쌍 부족 ({total}개) → PnP 스킵")
            return None

        pts_query = np.array(all_query_pts, dtype=np.float64)
        pts_3d    = np.array(all_pts_3d,    dtype=np.float64)

        dist_coeffs = np.zeros(4, dtype=np.float64)
        pnp_kwargs = dict(
            objectPoints      = pts_3d,
            imagePoints       = pts_query,
            cameraMatrix      = K_mat,
            distCoeffs        = dist_coeffs,
            reprojectionError = ransac_reproj,
            iterationsCount   = ransac_iter,
        )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            **pnp_kwargs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success or inliers is None:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                **pnp_kwargs, flags=cv2.SOLVEPNP_EPNP
            )
            if success and inliers is not None:
                print(f"  [PoseEstimator] EPNP 폴백으로 성공")

        if not success or inliers is None:
            print(f"  [PoseEstimator] PnP RANSAC 실패 (ITERATIVE + EPNP 모두 실패)")
            return None

        inliers = inliers.flatten()
        print(f"  [PoseEstimator] RANSAC 생존: {len(inliers)}개 / {total}개")

        result = {
            "rvec"         : rvec,
            "tvec"         : tvec,
            "inliers"      : inliers,
            "kd_pts_query" : pts_query,
            "kd_pts_3d"    : pts_3d,
            "num_kd"       : total,
            "num_inliers"  : len(inliers),
            "db_names"     : db_names_used,
        }

        if debug_mode:
            from visualizer import show_pnp
            show_pnp(query_path, result, dataset_path=dataset_path)

        return result

    # ──────────────────────────────────────────
    # 평가 유틸
    # ──────────────────────────────────────────
    @staticmethod
    def _qvec_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """COLMAP qvec (qw, qx, qy, qz) → 3×3 rotation matrix (world-to-camera)."""
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        return R

    def evaluate(self, rvec: np.ndarray, tvec: np.ndarray,
                 query_img_name: str) -> dict | None:
        """
        추정 포즈와 COLMAP GT 포즈를 비교 평가.

        GT: self.images에서 name == query_img_name 인 항목.
        쿼리 파일명과 DB 이미지 파일명이 동일한 경우를 전제로 함.

        Returns:
            {
              "gt_name"   : str,
              "C_gt"      : np.ndarray [3],   # GT 카메라 월드 위치
              "C_est"     : np.ndarray [3],   # 추정 카메라 월드 위치
              "R_gt"      : np.ndarray [3,3],
              "t_gt"      : np.ndarray [3],
              "trans_err" : float (미터),
              "rot_err"   : float (도),
              "high"      : bool,   # < 0.25m AND < 2°
              "medium"    : bool,   # < 0.5m  AND < 5°
              "low"       : bool,   # < 5.0m  AND < 10°
            }
            GT를 찾지 못하면 None 반환.
        """
        # GT 이미지 탐색 (정확한 이름 매칭)
        gt_img = None
        for img in self.images.values():
            if img["name"] == query_img_name:
                gt_img = img
                break

        if gt_img is None:
            print(f"  [평가] GT 없음: '{query_img_name}' 이 COLMAP DB에 없습니다.")
            return None

        # GT 포즈
        qw, qx, qy, qz = gt_img["qvec"]
        R_gt = self._qvec_to_R(qw, qx, qy, qz)   # world-to-camera
        t_gt = gt_img["tvec"]                       # [3]

        # 카메라 월드 위치: C = -R^T @ t
        C_gt = -R_gt.T @ t_gt

        # 추정 포즈
        R_est, _ = cv2.Rodrigues(rvec)
        t_est    = tvec.flatten()
        C_est    = -R_est.T @ t_est

        # Translation error (미터)
        trans_err = float(np.linalg.norm(C_gt - C_est))

        # Rotation error (도)
        R_rel  = R_gt @ R_est.T
        trace  = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = float(np.degrees(np.arccos(trace)))

        return {
            "gt_name":   query_img_name,
            "C_gt":      C_gt,
            "C_est":     C_est,
            "R_gt":      R_gt,
            "t_gt":      t_gt,
            "trans_err": trans_err,   # COLMAP world units (임의 스케일, 미터 아님)
            "rot_err":   rot_err,     # 도(degree)
        }

    def get_all_camera_centers(self) -> list[np.ndarray]:
        """
        self.images 전체를 순회하여 각 DB 이미지의 카메라 월드 위치 반환.

        Returns:
            [C_0, C_1, ..., C_N]  각 C_i는 np.ndarray [3]
        """
        centers = []
        for img in self.images.values():
            qw, qx, qy, qz = img["qvec"]
            R = self._qvec_to_R(qw, qx, qy, qz)
            C = -R.T @ img["tvec"]
            centers.append(C)
        return centers

    def pose_to_euler(self, rvec: np.ndarray) -> tuple[float, float, float]:
        """rvec → (pitch, yaw, roll) 도(degree) 단위."""
        R, _ = cv2.Rodrigues(rvec)
        sy   = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            roll  = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            roll  = 0.0
        return pitch, yaw, roll


# ──────────────────────────────────────────────
# 단독 실행 (간단 테스트)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from fine_matching import FineMatcher

    query_path = sys.argv[1] if len(sys.argv) > 1 else "./query/IMG_3342.JPG"
    db_path    = sys.argv[2] if len(sys.argv) > 2 else "./dataset/IMG_3342.JPG"

    matcher   = FineMatcher()
    match_res = matcher.match_pair(query_path, db_path)

    estimator = PoseEstimator()
    pose      = estimator.estimate(match_res)

    if pose:
        tvec             = pose["tvec"].flatten()
        pitch, yaw, roll = estimator.pose_to_euler(pose["rvec"])
        print(f"\n[포즈 결과]")
        print(f"  위치 (X, Y, Z)       : {tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}")
        print(f"  자세 (Pitch/Yaw/Roll) : {pitch:.2f}°, {yaw:.2f}°, {roll:.2f}°")
        print(f"  KD-Tree 통과 / RANSAC 생존 : {pose['num_kd']} / {pose['num_inliers']}")
    else:
        print("[포즈 결과] 추정 실패")
