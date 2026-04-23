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

    print(f"\n[2/4] Statistical Outlier Removal (k={args.k}, std={args.std})")
    mask, threshold, mean, std = statistical_outlier_removal(vertices, args.k, args.std)

    removed = (~mask).sum()
    kept    = mask.sum()
    print(f"  평균 이웃 거리: {mean:.4f}")
    print(f"  표준편차      : {std:.4f}")
    print(f"  임계값        : {threshold:.4f}")
    print(f"  제거됨        : {removed:,}개  ({removed/len(vertices)*100:.1f}%)")
    print(f"  유지됨        : {kept:,}개  ({kept/len(vertices)*100:.1f}%)")

    print(f"\n[3/4] 필터링된 포인트 클라우드 저장: {args.output}")
    write_ply(args.output, vertices[mask], props)

    # 필터링 후 범위 확인
    xyz_filtered = vertices[mask, :3]
    print(f"\n[4/4] 필터링 후 공간 범위:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        vmin, vmax = xyz_filtered[:, i].min(), xyz_filtered[:, i].max()
        print(f"  {axis}: [{vmin:.2f}, {vmax:.2f}]  범위={vmax-vmin:.2f}")

    print(f"\n완료! → {args.output}")
    print(f"MeshLab에서 열어 확인하세요.")


if __name__ == "__main__":
    main()
