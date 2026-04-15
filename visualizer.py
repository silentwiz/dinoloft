"""
visualizer.py
Phase 5 보조 모듈: Matplotlib 기반 디버그 시각화

debug_mode == 1 일 때 main_localization.py 에서 호출됨.

시각화 3종:
  1. show_coarse()  : 쿼리 + Top-K 검색 결과 나란히
  2. show_fine()    : LoFTR 매칭 선 시각화
  3. show_pnp()     : DB 이미지 위에
                        초록 X = KD-Tree 통과 점
                        빨간 O = RANSAC 생존 점
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image, ImageOps

matplotlib.rcParams["figure.dpi"] = 120


# ──────────────────────────────────────────────
# 1. Coarse 시각화
# ──────────────────────────────────────────────
def show_coarse(query_path: str,
                top_k_results: list[tuple[str, float]],
                dataset_path: str = "./dataset") -> None:
    """
    쿼리 이미지와 Top-K 검색 결과를 나란히 표시.

    Args:
        query_path    : 쿼리 이미지 경로
        top_k_results : [(db_img_name, score), ...] CoarseRetriever.retrieve() 반환값
        dataset_path  : DB 이미지 루트 폴더
    """
    k    = len(top_k_results)
    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4))
    fig.suptitle("Phase 2 — Coarse Retrieval", fontsize=13, fontweight="bold")

    # 쿼리
    axes[0].imshow(ImageOps.exif_transpose(Image.open(query_path)).convert("RGB"))
    axes[0].set_title(f"Query\n{Path(query_path).name}", fontsize=9)
    axes[0].axis("off")

    # Top-K
    for i, (name, score) in enumerate(top_k_results, 1):
        db_img_path = str(Path(dataset_path) / name)
        axes[i].imshow(ImageOps.exif_transpose(Image.open(db_img_path)).convert("RGB"))
        axes[i].set_title(f"Top-{i}  (cos={score:.4f})\n{name}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 2. Fine 시각화
# ──────────────────────────────────────────────
def show_fine(query_path: str,
              match_result: dict,
              dataset_path: str = "./dataset",
              max_lines: int = 200) -> None:
    """
    쿼리 ↔ DB 이미지를 좌우 배치하고 매칭 선을 긋는다.

    Args:
        query_path   : 쿼리 이미지 경로
        match_result : fine_matching.match_pair() 반환값
        max_lines    : 시각화 선 최대 개수 (너무 많으면 느림)
    """
    db_name  = match_result["db_name"]
    # 시각화: EXIF 보정 좌표 우선 사용 (쿼리 이미지가 올바른 방향으로 표시됨)
    q_pts    = match_result.get("query_pts_display", match_result["query_pts"])
    db_pts   = match_result["db_pts"]
    conf     = match_result["confidence"]

    db_img_path = str(Path(dataset_path) / db_name)
    # 쿼리: EXIF 보정 표시 (query_pts_display와 좌표계 일치)
    # DB: raw 표시 (db_pts가 raw 좌표계)
    q_img  = np.array(ImageOps.exif_transpose(Image.open(query_path)).convert("RGB"))
    db_img = np.array(Image.open(db_img_path).convert("RGB"))

    # 좌우 이어붙이기
    h      = max(q_img.shape[0], db_img.shape[0])
    q_pad  = np.zeros((h, q_img.shape[1],  3), dtype=np.uint8)
    d_pad  = np.zeros((h, db_img.shape[1], 3), dtype=np.uint8)
    q_pad[:q_img.shape[0],  :] = q_img
    d_pad[:db_img.shape[0], :] = db_img

    canvas   = np.concatenate([q_pad, d_pad], axis=1)
    offset_x = q_img.shape[1]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Phase 3 — Fine Matching  ({len(q_pts)} matches)\n"
                 f"Query: {Path(query_path).name}  ↔  DB: {db_name}",
                 fontsize=11, fontweight="bold")
    ax.imshow(canvas)
    ax.axis("off")

    # 서브샘플링
    if len(q_pts) > max_lines:
        idx = np.random.choice(len(q_pts), max_lines, replace=False)
        q_pts_vis = q_pts[idx]
        db_pts_vis = db_pts[idx]
        conf_vis   = conf[idx]
    else:
        q_pts_vis, db_pts_vis, conf_vis = q_pts, db_pts, conf

    # 신뢰도에 따라 색 그라디언트 (낮음=파랑, 높음=노랑)
    cmap = plt.cm.plasma
    for qp, dp, c in zip(q_pts_vis, db_pts_vis, conf_vis):
        color = cmap(float(c))
        ax.plot([qp[0], dp[0] + offset_x], [qp[1], dp[1]],
                color=color, linewidth=0.6, alpha=0.7)
        ax.plot(qp[0], qp[1], "o", color=color, markersize=2)
        ax.plot(dp[0] + offset_x, dp[1], "o", color=color, markersize=2)

    if len(conf) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=conf.min(), vmax=1.0))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.02, label="Confidence")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 3. PnP 시각화
# ──────────────────────────────────────────────
def show_pnp(query_path: str,
             pose_result: dict,
             dataset_path: str = "./dataset") -> None:
    """
    쿼리 이미지 위에 KD-Tree 통과점(초록 X)과 RANSAC 생존점(빨간 O)을 덧그린다.

    Args:
        query_path  : 쿼리 이미지 경로
        pose_result : pose_estimation.estimate() 반환값
    """
    # EXIF 보정 이미지로 표시 + kd_pts_query (raw 좌표)를 EXIF 보정 공간으로 변환
    from fine_matching import _get_exif_orientation, _pts_raw_to_exif
    orient  = _get_exif_orientation(query_path)
    raw_pil = Image.open(query_path)
    raw_wh  = raw_pil.size                          # (W, H) raw
    img     = np.array(ImageOps.exif_transpose(raw_pil).convert("RGB"))

    kd_pts_raw = pose_result["kd_pts_query"]        # raw 좌표계
    kd_pts     = _pts_raw_to_exif(kd_pts_raw, orient, raw_wh)  # 표시용 변환
    inliers    = pose_result["inliers"]
    num_kd     = pose_result["num_kd"]
    num_in     = pose_result["num_inliers"]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Phase 4 — PnP Pose Estimation\n"
                 f"Query: {Path(query_path).name}   "
                 f"KD-Tree: {num_kd}pts   RANSAC inliers: {num_in}pts",
                 fontsize=11, fontweight="bold")
    ax.imshow(img)
    ax.axis("off")

    # KD-Tree 통과 전체 → 초록 X
    for pt in kd_pts:
        ax.plot(pt[0], pt[1], "gx", markersize=6, markeredgewidth=1.2, alpha=0.7)

    # RANSAC 생존 → 빨간 O (위에 덮어 그림)
    for idx in inliers:
        pt = kd_pts[idx]
        ax.plot(pt[0], pt[1], "ro", markersize=6, markeredgewidth=1,
                fillstyle="none", alpha=0.9)

    # 범례
    green_patch = mpatches.Patch(color="lime",  label=f"KD-Tree hit  ({num_kd})")
    red_patch   = mpatches.Patch(color="red",   label=f"RANSAC inlier ({num_in})")
    ax.legend(handles=[green_patch, red_patch],
              loc="upper right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 4. PLY 결과 저장
# ──────────────────────────────────────────────
def _sphere_cluster(center: np.ndarray, color: tuple,
                    r: float = 0.05) -> list[tuple]:
    """
    중심 center 주변에 구형 점 클러스터 생성 (27개: 3×3×3 격자).

    Returns:
        [(x, y, z, r, g, b), ...]
    """
    pts = []
    for dx in (-r, 0, r):
        for dy in (-r, 0, r):
            for dz in (-r, 0, r):
                p = center + np.array([dx, dy, dz])
                pts.append((p[0], p[1], p[2], color[0], color[1], color[2]))
    return pts


def save_result_ply(points3d: dict,
                    all_cam_centers: list,
                    eval_result: dict | None,
                    inlier_pts_3d: np.ndarray | None = None,
                    est_camera_center: np.ndarray | None = None,
                    output_path: str = "./sfm_output/result_pose.ply") -> None:
    """
    3D 포인트 클라우드 + DB 카메라 위치 + GT/추정 카메라 위치를 PLY로 저장.

    색상 코드:
      - 3D 포인트        : 원본 RGB
      - DB 카메라 (전체) : 회색 (150, 150, 150)
      - GT 카메라        : 초록 (0, 255, 0) — 구형 클러스터 (GT 있을 때만)
      - 추정 카메라      : 빨강 (255, 50, 50) — 구형 클러스터
      - PnP inlier 3D 점 : 노란색 (255, 220, 0)

    Args:
        points3d          : sfm_db["points3d"] = {id: {"xyz", "rgb"}}
        all_cam_centers   : estimator.get_all_camera_centers() 결과
        eval_result       : estimator.evaluate() 결과 (C_gt, C_est 포함). GT 없으면 None.
        inlier_pts_3d     : PnP RANSAC inlier 3D 좌표 [K, 3] (없으면 None)
        est_camera_center : GT 없을 때 추정 카메라 월드 위치 [3] (-R^T @ t)
        output_path       : 저장 경로
    """
    rows = []  # (x, y, z, r, g, b)

    # 3D 포인트 클라우드
    for pt in points3d.values():
        x, y, z = pt["xyz"]
        r, g, b = pt["rgb"]
        rows.append((x, y, z, int(r), int(g), int(b)))

    # DB 카메라 위치 (회색)
    for C in all_cam_centers:
        rows.append((C[0], C[1], C[2], 150, 150, 150))

    if eval_result is not None:
        # GT 카메라 (초록 구형 클러스터)
        rows.extend(_sphere_cluster(eval_result["C_gt"],  color=(0, 255, 0),   r=0.05))
        # 추정 카메라 (빨강 구형 클러스터)
        rows.extend(_sphere_cluster(eval_result["C_est"], color=(255, 50, 50), r=0.05))
    elif est_camera_center is not None:
        # GT 없을 때: 추정 카메라만 (빨강 구형 클러스터)
        rows.extend(_sphere_cluster(est_camera_center, color=(255, 50, 50), r=0.05))

    # PnP inlier 3D 점 (노란색)
    if inlier_pts_3d is not None and len(inlier_pts_3d) > 0:
        for pt in inlier_pts_3d:
            rows.append((pt[0], pt[1], pt[2], 255, 220, 0))

    # PLY ASCII 출력
    from pathlib import Path as _Path
    _Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for row in rows:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} "
                    f"{row[3]} {row[4]} {row[5]}\n")

    print(f"[PLY] 저장 완료: {output_path}  ({len(rows)}개 점)")


# ──────────────────────────────────────────────
# 5. 전체 결과 요약 출력
# ──────────────────────────────────────────────
def print_summary(query_path: str,
                  top_k_results: list[tuple[str, float]],
                  match_results: list[dict],
                  pose_result: dict | None,
                  eval_result: dict | None = None,
                  unit_to_meter: float | None = None) -> None:
    """터미널에 최종 결과 요약 출력."""
    print("\n" + "=" * 55)
    print("  LOCALIZATION RESULT SUMMARY")
    print("=" * 55)
    print(f"  Query : {Path(query_path).name}")

    print("\n  [Coarse]")
    for rank, (name, score) in enumerate(top_k_results, 1):
        print(f"    Top-{rank}: {name}  (cos={score:.4f})")

    print("\n  [Fine Matching]")
    for res in match_results:
        print(f"    {res['db_name']}: {len(res['query_pts'])} matches")

    print("\n  [Pose Estimation]")
    if pose_result is None:
        print("    ❌ 포즈 추정 실패")
    else:
        tvec = pose_result["tvec"].flatten()
        print(f"    위치 (X, Y, Z)  : {tvec[0]:.4f}  {tvec[1]:.4f}  {tvec[2]:.4f}")
        print(f"    KD통과 / RANSAC : {pose_result['num_kd']} / {pose_result['num_inliers']}")

    if eval_result is not None:
        print("\n  [평가 (GT 비교)]")
        print(f"    GT 이미지       : {eval_result['gt_name']}")
        if unit_to_meter is not None:
            trans_m = eval_result['trans_err'] * unit_to_meter
            print(f"    위치 오차       : {eval_result['trans_err']:.4f} units × {unit_to_meter:.4f} → {trans_m:.2f} m")
        else:
            print(f"    위치 오차       : {eval_result['trans_err']:.4f}  (COLMAP world units)")
        print(f"    회전 오차       : {eval_result['rot_err']:.2f} °")

    print("=" * 55 + "\n")
