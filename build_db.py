"""
build_db_claude.py
Phase 1: 오프라인 DB 구축 파이프라인

실행 순서:
  Step 1. COLMAP 실행 (feature_extractor → exhaustive_matcher → mapper → 결과는 .bin)
  Step 2. colmap_parser로 .bin 파일 파싱 → 2D-3D 맵핑 LUT
  Step 3. DB 이미지 전체를 DINOv2에 통과 → Global 평균 벡터 [384] → .pt 저장
  Step 4. LUT + 메타 정보를 pickle로 저장

출력물 (sfm_output/):
  - sparse/0/cameras.bin, images.bin, points3D.bin  (COLMAP 결과)
  - sfm_db.pkl                                       (2D-3D LUT + cameras + images 메타)
  - dino_feats.pt                                    (DB 이미지 DINOv2 벡터 딕셔너리)
"""

import os
import pickle
import subprocess
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

from colmap_parser import load_colmap_model


# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
DATASET_PATH = "./dataset"
OUTPUT_PATH  = "./sfm_output"
SPARSE_DIR   = os.path.join(OUTPUT_PATH, "sparse")
RECON_DIR    = os.path.join(SPARSE_DIR, "0")   # COLMAP mapper 기본 출력 서브폴더
DB_PATH      = os.path.join(OUTPUT_PATH, "database.db")
PKL_PATH     = os.path.join(OUTPUT_PATH, "sfm_db.pkl")
DINO_PT_PATH = os.path.join(OUTPUT_PATH, "dino_feats.pt")

DEBUG = True   # True면 진행 상황 출력


# ──────────────────────────────────────────────
# Step 1 : COLMAP 실행
# ──────────────────────────────────────────────
def run_colmap():
    """COLMAP 4단계 파이프라인 실행. 이미 .bin 결과물이 있으면 스킵."""
    bin_check = Path(RECON_DIR) / "images.bin"
    if bin_check.exists():
        print(f"[Step 1] COLMAP 결과물 발견 → 스킵 ({bin_check})")
        return

    print("[Step 1] COLMAP 파이프라인 시작...")
    os.makedirs(SPARSE_DIR, exist_ok=True)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"   # headless 환경 대응

    commands = [
        ["colmap", "feature_extractor",
            "--database_path",              DB_PATH,
            "--image_path",                 DATASET_PATH,
            "--FeatureExtraction.use_gpu",  "0"],   # Mac headless: OpenGL 비활성화
        ["colmap", "exhaustive_matcher",
            "--database_path",             DB_PATH,
            "--FeatureMatching.use_gpu",   "0"],    # Mac headless: OpenGL 비활성화
        ["colmap", "mapper",
            "--database_path", DB_PATH,
            "--image_path",    DATASET_PATH,
            "--output_path",   SPARSE_DIR],
    ]

    for cmd in commands:
        if DEBUG:
            print(f"  ▶ {' '.join(cmd)}")
        # capture_output 미사용 → COLMAP stdout/stderr 터미널에 실시간 출력
        result = subprocess.run(cmd, text=True, env=env)
        if result.returncode != 0:
            print(f"  ❌ 오류: {' '.join(cmd)}")
            raise RuntimeError("COLMAP 실행 실패")

    print("[Step 1] COLMAP 완료 ✓")


# ──────────────────────────────────────────────
# Step 2 : COLMAP .bin 파싱 + LUT 생성
# ──────────────────────────────────────────────
def parse_colmap() -> tuple[dict, dict, dict, dict]:
    """colmap_parser를 이용해 .bin 파일 파싱 후 (cameras, images, points3d, mapping) 반환."""
    print("[Step 2] COLMAP .bin 파싱...")
    cameras, images, points3d, mapping = load_colmap_model(RECON_DIR)
    print("[Step 2] 파싱 완료 ✓")
    return cameras, images, points3d, mapping


# ──────────────────────────────────────────────
# Step 3 : DINOv2 특징 추출
# ──────────────────────────────────────────────
def extract_dino_features(image_names: list[str]) -> dict:
    """
    DB 이미지 전체를 DINOv2(vits14, 384-dim)에 통과시켜
    CLS 토큰 기반 Global 벡터 [384]를 딕셔너리로 반환.

    반환값:
        {"IMG_XXXX.JPG": torch.Tensor([384]), ...}
    """
    print("[Step 3] DINOv2 특징 추출 시작...")

    # 디바이스 선택 (M-series Mac → mps, CUDA → cuda, 그 외 → cpu)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  사용 디바이스: {device}")

    # DINOv2 ViT-S/14 로드 (출력 차원 384)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)

    # 전처리 파이프라인 (DINOv2 공식 권장값)
    transform = T.Compose([
        T.Resize(518),               # 짧은 변 기준 리사이즈 (14의 배수)
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    feats = {}
    total = len(image_names)

    with torch.no_grad():
        for idx, name in enumerate(image_names, 1):
            img_path = os.path.join(DATASET_PATH, name)
            if not os.path.exists(img_path):
                print(f"  ⚠ 이미지 없음, 스킵: {img_path}")
                continue

            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)   # [1, 3, H, W]

            # forward → CLS 토큰 [1, 384]
            out  = model(tensor)                # dinov2는 기본적으로 CLS 토큰 반환
            vec  = out.squeeze(0).cpu()         # [384]
            feats[name] = vec

            if DEBUG and idx % 20 == 0:
                print(f"  [{idx}/{total}] 완료")

    print(f"[Step 3] DINOv2 특징 추출 완료 ({len(feats)}개) ✓")
    return feats


# ──────────────────────────────────────────────
# Step 4 : 저장
# ──────────────────────────────────────────────
def save_artifacts(cameras: dict, images: dict, points3d: dict,
                   mapping: dict, dino_feats: dict):
    """LUT(pkl)과 DINOv2 벡터(.pt)를 파일로 저장."""
    print("[Step 4] 결과물 저장...")

    # --- sfm_db.pkl ---
    db_data = {
        "cameras":  cameras,    # 카메라 내부 파라미터
        "images":   images,     # 이미지 메타 (pose, 2D 관측값)
        "points3d": points3d,   # 3D 포인트 좌표
        "mapping":  mapping,    # {img_name: [{"2d": (x,y), "xyz": np.array}, ...]}
    }
    with open(PKL_PATH, "wb") as f:
        pickle.dump(db_data, f)
    print(f"  저장 완료: {PKL_PATH}")

    # --- dino_feats.pt ---
    torch.save(dino_feats, DINO_PT_PATH)
    print(f"  저장 완료: {DINO_PT_PATH}")

    print("[Step 4] 저장 완료 ✓")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"데이터셋 폴더를 찾을 수 없습니다: {DATASET_PATH}")

    # Step 1: COLMAP
    run_colmap()

    # Step 2: 파싱
    cameras, images, points3d, mapping = parse_colmap()

    # Step 3: DINOv2 (COLMAP에 등록된 이미지 이름 기준)
    image_names = [img["name"] for img in images.values()]
    dino_feats  = extract_dino_features(image_names)

    # Step 4: 저장
    save_artifacts(cameras, images, points3d, mapping, dino_feats)

    print("\n🎉 Phase 1 완료! 생성된 파일:")
    print(f"  {PKL_PATH}")
    print(f"  {DINO_PT_PATH}")


if __name__ == "__main__":
    main()
