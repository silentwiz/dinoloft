"""
filter_ply.py
PLY 포인트 클라우드에서 아웃라이어를 제거하고 정제된 PLY를 저장.

방법: Statistical Outlier Removal
  각 포인트에 대해 K개 최근접 이웃까지의 평균 거리를 계산.
  전체 평균 + N × 표준편차를 초과하는 포인트를 아웃라이어로 제거.

실행:
  python3 filter_ply.py
  python3 filter_ply.py --input sfm_output/result_pose.ply --k 20 --std 2.0
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree


INPUT_PLY  = "sfm_output/result_pose.ply"
OUTPUT_PLY = "sfm_output/result_pose_filtered.ply"
K          = 20    # 이웃 포인트 수 (클수록 글로벌 구조 반영)
STD_MULT   = 2.0   # 임계값 = 평균 + STD_MULT × 표준편차 (낮을수록 엄격)

# visualizer.py 가 사용하는 마커 색상 — SOR 대상에서 제외해 항상 보존
MARKER_COLORS = {
    (150, 150, 150),  # DB 카메라 (회색)
    (0,   255,   0),  # GT 카메라 구형 클러스터
    (0,   220,   0),  # GT 카메라 시선 화살표
    (255,  50,  50),  # 추정 카메라 구형 클러스터 + 시선 화살표
    (255, 220,   0),  # PnP inlier 3D 점 (노란색)
}


def read_ply(path: str):
    vertices = []
    with open(path, "r") as f:
        # 헤더 파싱
        props = []
        for line in f:
            line = line.strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        # 데이터 읽기
        for line in f:
            vals = line.split()
            if vals:
                vertices.append([float(v) for v in vals])
    return np.array(vertices), props, n_verts


def write_ply(path: str, vertices: np.ndarray, props: list):
    has_color = len(props) >= 6
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for v in vertices:
            if has_color:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                        f"{int(v[3])} {int(v[4])} {int(v[5])}\n")
            else:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def _find_dense_cluster_center(xyz: np.ndarray, k: int = 27) -> np.ndarray:
    """K 최근접 이웃이 가장 조밀한 점을 찾아 그 이웃들의 무게중심을 반환.
    구형 클러스터(조밀)와 시선 화살표(분산)가 같은 색을 쓸 때 클러스터 중심을 추정하는 데 사용."""
    k = min(k, len(xyz))
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=k)
    seed_idx = np.argmin(dists[:, -1])
    _, idx = tree.query(xyz[seed_idx], k=k)
    return xyz[idx].mean(axis=0)


def _expand_sphere_cluster(vertices: np.ndarray,
                            color: tuple,
                            new_r: float,
                            grid_n: int = 5) -> np.ndarray:
    """특정 색상 포인트 중 가장 조밀한 클러스터 중심에 grid_n^3 개의
    새 포인트를 추가해 구를 시각적으로 확대. 기존 포인트는 유지."""
    rgb = vertices[:, 3:6].astype(int)
    mask = np.array([tuple(c) == color for c in rgb])
    if not mask.any():
        return vertices

    xyz_colored = vertices[mask, :3]
    center = (xyz_colored.mean(axis=0) if len(xyz_colored) <= 27
              else _find_dense_cluster_center(xyz_colored, k=27))

    coords = np.linspace(-new_r, new_r, grid_n)
    new_pts = np.array([
        [center[0] + dx, center[1] + dy, center[2] + dz,
         color[0], color[1], color[2]]
        for dx in coords for dy in coords for dz in coords
    ])
    return np.vstack([vertices, new_pts])


def statistical_outlier_removal(vertices: np.ndarray, k: int, std_mult: float):
    """K 이웃 평균 거리 기반 아웃라이어 제거."""
    xyz = vertices[:, :3]
    tree = KDTree(xyz)

    print(f"  KD-Tree 구축 완료. 이웃 거리 계산 중 (k={k})...")
    # 자기 자신 포함이므로 k+1, 첫 번째(거리=0)는 자기 자신
    dists, _ = tree.query(xyz, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)   # 자기 자신 제외

    global_mean = mean_dists.mean()
    global_std  = mean_dists.std()
    threshold   = global_mean + std_mult * global_std

    mask = mean_dists <= threshold
    return mask, threshold, global_mean, global_std


def main():
    parser = argparse.ArgumentParser(description="PLY 아웃라이어 제거")
    parser.add_argument("--input",  default=INPUT_PLY)
    parser.add_argument("--output", default=OUTPUT_PLY)
    parser.add_argument("--k",   type=int,   default=K,
                        help=f"이웃 포인트 수 (기본 {K})")
    parser.add_argument("--std", type=float, default=STD_MULT,
                        help=f"표준편차 배율 (기본 {STD_MULT}, 낮을수록 엄격)")
    args = parser.parse_args()

    print(f"[1/4] PLY 읽는 중: {args.input}")
    vertices, props, _ = read_ply(args.input)
    print(f"  원본 포인트 수: {len(vertices):,}개")

    # 마커 점 분리 — SOR 에서 제외하고 항상 보존
    has_color = len(props) >= 6
    if has_color:
        rgb = vertices[:, 3:6].astype(int)
        is_marker = np.array([
            tuple(c) in MARKER_COLORS for c in rgb
        ])
        regular_verts = vertices[~is_marker]
        marker_verts  = vertices[is_marker]
        print(f"  마커 점 (보존): {is_marker.sum():,}개  "
              f"/ 일반 SfM 점 (필터링 대상): {(~is_marker).sum():,}개")
    else:
        regular_verts = vertices
        marker_verts  = np.empty((0, vertices.shape[1]))
        print("  색상 정보 없음 — 전체 포인트에 SOR 적용")

    print(f"\n[2/4] Statistical Outlier Removal (k={args.k}, std={args.std})")
    # 통계는 전체 점(마커 포함)으로 계산해 이전과 동일한 임계값 유지
    mask_all, threshold, mean, std = statistical_outlier_removal(vertices, args.k, args.std)

    # 마커 점은 SOR 결과와 무관하게 강제 보존
    if has_color:
        mask_all[is_marker] = True
    sor_mask     = mask_all[~is_marker] if has_color else mask_all
    removed      = (~sor_mask).sum()
    kept         = sor_mask.sum()

    print(f"  평균 이웃 거리: {mean:.4f}")
    print(f"  표준편차      : {std:.4f}")
    print(f"  임계값        : {threshold:.4f}")
    print(f"  제거됨        : {removed:,}개  ({removed/len(regular_verts)*100:.1f}%)")
    print(f"  유지됨 (SfM)  : {kept:,}개  ({kept/len(regular_verts)*100:.1f}%)")

    filtered = vertices[mask_all]
    print(f"  최종 포인트 수: {len(filtered):,}개  (SfM {kept:,} + 마커 {len(marker_verts):,})")

    # 카메라 마커 구형 클러스터 시인성 향상
    # GT 초록(0,255,0): 고유 색상이므로 단순 평균으로 중심 결정 → r=0.10
    # 추정 빨강(255,50,50): 시선 화살표와 색 공유 → 조밀 클러스터 중심 추정 → r=0.08
    if has_color:
        filtered = _expand_sphere_cluster(filtered, (0,   255,   0), new_r=0.10, grid_n=5)
        filtered = _expand_sphere_cluster(filtered, (255,  50,  50), new_r=0.08, grid_n=5)
        print(f"  카메라 마커 확대 완료: 최종 {len(filtered):,}개 포인트")

    print(f"\n[3/4] 필터링된 포인트 클라우드 저장: {args.output}")
    write_ply(args.output, filtered, props)

    # 필터링 후 범위 확인
    xyz_filtered = filtered[:, :3]
    print(f"\n[4/4] 필터링 후 공간 범위:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vmin, vmax = xyz_filtered[:, i].min(), xyz_filtered[:, i].max()
        print(f"  {axis}: [{vmin:.2f}, {vmax:.2f}]  범위={vmax-vmin:.2f}")

    print(f"\n완료! → {args.output}")
    print(f"MeshLab에서 열어 확인하세요.")


if __name__ == "__main__":
    main()
