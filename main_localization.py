"""
main_localization.py
[위치추정] — 쿼리 이미지 한 장으로 6DoF 포즈 추정

사전 준비 (최초 1회):
  python build_db_claude.py
    → sfm_output/sfm_db.pkl      (2D-3D LUT)
    → sfm_output/dino_feats.pt   (DB DINOv2 벡터)

실행:
  python main_localization.py [쿼리_이미지_경로] [--topk N] [--debug 0|1] [--eval 0|1]

예시:
  python main_localization.py ./query/IMG_3342.JPG
  python main_localization.py ./query/IMG_3342.JPG --topk 3 --debug 0 --eval 1

파이프라인 흐름:
  쿼리 이미지
    │
    ▼ [Coarse]  DINOv2 코사인 유사도 → Top-K DB 이미지 선택
    │
    ▼ [Fine]    LoFTR 픽셀 매칭 → (query_pts, db_pts)
    │
    ▼ [PnP]     KD-Tree로 db_pts ↔ 3D 연결 → solvePnPRansac → rvec, tvec
    │
    ▼ 결과 출력 (위치 XYZ, 자세 Pitch/Yaw/Roll)
       + eval=1: GT 비교 평가 + PLY 저장
       + debug=1: 각 단계 시각화 팝업
"""

import argparse
import numpy as np
from pathlib import Path

from coarse_retrieval  import CoarseRetriever
from fine_matching     import FineMatcher
from pose_estimation   import PoseEstimator
from visualizer        import print_summary


# ═══════════════════════════════════════════════════════════════
#  파라미터 설정  ← 여기서 모든 조정 가능
#  CLI 인자(--topk 등)로 오버라이드 가능 / VSCode F5 직접 실행 시 이 값 사용
# ═══════════════════════════════════════════════════════════════

# ── 파일 경로 ─────────────────────────────────────────────────
DATASET_PATH  = "./dataset"
SFM_DB_PATH   = "./sfm_output/sfm_db.pkl"
DINO_PT_PATH  = "./sfm_output/dino_feats.pt"
DEFAULT_QUERY = "./query/IMG_3342.JPG"   # VSCode 직접 실행 시 기본 쿼리 이미지

# ── 실행 모드 ─────────────────────────────────────────────────
EVAL_MODE  = 1   # 1=GT 비교 평가 + PLY 저장,  0=생략
DEBUG_MODE = 1   # 1=각 단계 시각화 팝업 ON,   0=OFF

# ── Coarse Retrieval (DINOv2) ─────────────────────────────────
TOP_K = 1       # Top-K DB 이미지 검색 수 (클수록 후보 많아지나 속도 저하)

# ── Fine Matching (LoFTR) ─────────────────────────────────────
LOFTR_WEIGHTS        = "outdoor" # 사전학습 가중치: "outdoor" | "indoor"
                               #   outdoor: MegaDepth 학습 (야외 랜드마크, 원거리, 스케일 변화 강인)
                               #   indoor : ScanNet 학습  (실내 RGB-D, 단거리, 조명 안정)
LOFTR_RESIZE_LONG    = 640   # LoFTR 입력 긴 변 해상도 (8의 배수)
                               #   640  : 빠름, 저해상도 (기본값)
                               #   832  : 중간
                               #   1024 : 권장 (4K 쿼리 시 특징 보존)
                               #   ※ 메모리 부족 시 832 또는 640으로 낮출 것
LOFTR_CONF_THRESHOLD = 0.5    # LoFTR 신뢰도 임계값
                               #   높일수록 매칭 수 감소, 정밀도 향상
                               #   낮출수록 매칭 수 증가, 노이즈 증가

# ── 2D-3D 연결 방식 ──────────────────────────────────────────
MATCH_METHOD  = "brute_force"  # "brute_force" (권장) | "kd_tree"
                               #   brute_force: 반경 제한 없이 최근접 COLMAP 점 연결
                               #                LoFTR semi-dense + sparse COLMAP 조합에 적합
                               #                RANSAC이 outlier 처리
                               #   kd_tree    : KD_RADIUS_PX 이내 COLMAP 점만 허용
                               #                COLMAP 점 밀도가 충분히 높을 때 유리
MAX_NN_DIST_PX = None          # brute_force 수동 임계값 (None=자동 계산)
NN_DIST_FACTOR = 2.0           # 자동 임계값 배율: threshold = factor × median(COLMAP_NN_dist)
                               #   2.0 권장: Poisson 공간 과정 기준 2× E[d_NN] 이상은
                               #             다른 구조물의 SIFT 점일 가능성이 높음
                               #   낮출수록 엄격 (1.5), 높일수록 관대 (3.0)

# ── KD-Tree (match_method="kd_tree" 일 때만 사용) ─────────────
KD_RADIUS_PX = 5.0            # KD-Tree 탐색 반경 (픽셀)
                               #   작을수록 엄격한 매칭, 클수록 더 많은 후보 허용

# ── PnP RANSAC ────────────────────────────────────────────────
RANSAC_REPROJ = 10.0           # 재투영 오차 임계값 (픽셀): 이 이하만 inlier
                               #   작을수록 엄격, 클수록 inlier 수 증가
RANSAC_ITER   = 1000          # RANSAC 최대 반복 횟수 (클수록 정확하나 느림)

# ── 쿼리 카메라 K ─────────────────────────────────────────────
# 실제 쿼리 카메라의 캘리브레이션 값으로 교체하세요.
# 기준 해상도(QUERY_CAMERA_K_WH)와 실제 쿼리 해상도가 다르면
# fx/fy/cx/cy 가 자동 스케일됩니다.
#
# 아래는 iPhone 13 Pro — 1920×1440 (4:3) 모드 기준 예시값
QUERY_CAMERA_K = np.array([
    [1450.0,    0.0,  960.0],
    [   0.0, 1450.0,  720.0],
    [   0.0,    0.0,    1.0],
], dtype=np.float64)
QUERY_CAMERA_K_WH = (1920, 1440)  # K 캘리브레이션 기준 해상도 (width, height)

# ── 스케일 캘리브레이션 ───────────────────────────────────────
# MeshLab 측정: 7.70053 COLMAP units = 67.29m (Google Maps 기준)
# → 1 COLMAP unit = 67.29 / 7.70053 ≈ 8.7384m
COLMAP_UNIT_TO_METER = 67.29 / 7.70053   # m / COLMAP unit

# ═══════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
# 파이프라인
# ──────────────────────────────────────────────────────────────
def localize(query_path: str,
             top_k: int       = TOP_K,
             debug_mode: int  = DEBUG_MODE,
             eval_mode: int   = EVAL_MODE) -> dict | None:
    """
    쿼리 이미지 한 장의 6DoF 포즈를 추정하는 전체 파이프라인.

    Returns:
        성공 시: {"rvec", "tvec", "tvec_xyz", "euler_deg", "db_name", "eval", ...}
        실패 시: None
    """
    if not Path(query_path).exists():
        raise FileNotFoundError(f"쿼리 이미지를 찾을 수 없습니다: {query_path}")

    # ── 모듈 초기화 ───────────────────────────────
    print("\n[Pipeline] 모듈 초기화...")
    retriever = CoarseRetriever(dino_pt_path=DINO_PT_PATH)
    matcher   = FineMatcher(weights=LOFTR_WEIGHTS,
                            conf_thr=LOFTR_CONF_THRESHOLD,
                            resize_long=LOFTR_RESIZE_LONG)
    estimator = PoseEstimator(sfm_db_path=SFM_DB_PATH)

    # ── Phase 2: Coarse 검색 ──────────────────────
    print(f"\n[Phase 2] Coarse 검색 (Top-{top_k})...")
    top_k_results = retriever.retrieve(query_path, top_k=top_k,
                                       debug_mode=debug_mode,
                                       dataset_path=DATASET_PATH)

    # ── Phase 3: Fine 매칭 (LoFTR) ───────────────
    print("\n[Phase 3] Fine 매칭 (LoFTR)...")
    db_paths      = [str(Path(DATASET_PATH) / name) for name, _ in top_k_results]
    match_results = matcher.match(query_path, db_paths,
                                  debug_mode=debug_mode,
                                  dataset_path=DATASET_PATH)

    # ── Phase 4: PnP 포즈 추정 (Top-K 합산) ─────
    print("\n[Phase 4] PnP 포즈 추정...")
    best_pose = estimator.estimate_merged(
        match_results,
        kd_radius=KD_RADIUS_PX,
        match_method=MATCH_METHOD,
        max_nn_dist=MAX_NN_DIST_PX,
        nn_dist_factor=NN_DIST_FACTOR,
        ransac_reproj=RANSAC_REPROJ,
        ransac_iter=RANSAC_ITER,
        query_K=QUERY_CAMERA_K,
        query_K_wh=QUERY_CAMERA_K_WH,
        debug_mode=debug_mode,
        query_path=query_path,
        dataset_path=DATASET_PATH,
    )
    best_db_name = (best_pose["db_names"][0] if best_pose and best_pose["db_names"]
                    else None)

    # ── 평가 + PLY 출력 ───────────────────────────
    eval_result = None
    if eval_mode and best_pose is not None:
        query_name  = Path(query_path).name
        eval_result = estimator.evaluate(best_pose["rvec"], best_pose["tvec"],
                                         query_name)
        if eval_result:
            trans_m = eval_result['trans_err'] * COLMAP_UNIT_TO_METER
            print(f"\n[평가] GT: {eval_result['gt_name']}")
            print(f"[평가] 위치 오차: {eval_result['trans_err']:.4f} units × {COLMAP_UNIT_TO_METER:.4f} → {trans_m:.2f} m")
            print(f"[평가] 회전 오차: {eval_result['rot_err']:.2f} °")

        # PLY는 GT 유무와 무관하게 항상 저장
        import cv2
        R_est, _ = cv2.Rodrigues(best_pose["rvec"])
        est_center  = -R_est.T @ best_pose["tvec"].flatten()
        all_centers = estimator.get_all_camera_centers()
        inlier_3d   = best_pose["kd_pts_3d"][best_pose["inliers"]]
        from visualizer import save_result_ply
        save_result_ply(estimator.points3d, all_centers, eval_result,
                        inlier_pts_3d=inlier_3d,
                        est_camera_center=est_center)

    # ── 결과 요약 ─────────────────────────────────
    if debug_mode:
        print_summary(query_path, top_k_results, match_results, best_pose,
                      eval_result, unit_to_meter=COLMAP_UNIT_TO_METER)

    if best_pose is None:
        return None

    tvec = best_pose["tvec"].flatten()
    pitch, yaw, roll = estimator.pose_to_euler(best_pose["rvec"])

    return {
        **best_pose,
        "tvec_xyz":  (tvec[0], tvec[1], tvec[2]),
        "euler_deg": (pitch, yaw, roll),
        "db_name":   best_db_name,
        "eval":      eval_result,
    }


# ──────────────────────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 + LoFTR + PnP 로컬라이제이션 파이프라인"
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY,
                        help=f"쿼리 이미지 경로 (기본: {DEFAULT_QUERY})")
    parser.add_argument("--topk",  type=int, default=TOP_K,
                        help=f"Coarse Top-K 수 (기본 {TOP_K})")
    parser.add_argument("--debug", type=int, default=DEBUG_MODE, choices=[0, 1],
                        help=f"1=시각화 ON (기본 {DEBUG_MODE})")
    parser.add_argument("--eval",  type=int, default=EVAL_MODE,  choices=[0, 1],
                        help=f"1=GT 평가 + PLY 출력 (기본 {EVAL_MODE})")
    args = parser.parse_args()

    result = localize(args.query, top_k=args.topk,
                      debug_mode=args.debug, eval_mode=args.eval)

    if result:
        x, y, z          = result["tvec_xyz"]
        pitch, yaw, roll = result["euler_deg"]
        print(f"\n최종 포즈 추정 성공!")
        print(f"  위치 (X, Y, Z)        : {x:.4f}  {y:.4f}  {z:.4f}")
        print(f"  자세 (Pitch/Yaw/Roll) : {pitch:.2f}°  {yaw:.2f}°  {roll:.2f}°")
        print(f"  참조 DB 이미지        : {result['db_name']}")
    else:
        print("\n포즈 추정 실패 — 매칭 수가 부족하거나 DB와 겹치는 영역이 없습니다.")


if __name__ == "__main__":
    main()
