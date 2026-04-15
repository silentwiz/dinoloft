"""
coarse_retrieval.py
Phase 2: Coarse 검색 모듈 (DINOv2 기반 이미지 검색)

주요 기능:
  - 쿼리 이미지 → DINOv2 특징 추출 [384]
  - 사전 구축된 dino_feats.pt 와 코사인 유사도 연산 (MPS/CUDA/CPU)
  - 유사도 상위 Top-K DB 이미지 이름 + 점수 반환

사용 예시:
  from coarse_retrieval import CoarseRetriever

  retriever = CoarseRetriever()
  results   = retriever.retrieve("./query/IMG_3342.JPG", top_k=2)
  # results = [("IMG_3342.JPG", 0.981), ("IMG_3341.JPG", 0.963)]
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from PIL import Image, ImageOps


# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
DINO_PT_PATH = "./sfm_output/dino_feats.pt"


class CoarseRetriever:
    """DINOv2 코사인 유사도 기반 Coarse 검색기."""

    def __init__(self, dino_pt_path: str = DINO_PT_PATH):
        # 디바이스 선택
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[CoarseRetriever] 디바이스: {self.device}")

        # DINOv2 모델 로드
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval().to(self.device)

        # 전처리 파이프라인 (build_db_claude.py와 동일해야 함)
        self.transform = T.Compose([
            T.Resize(518),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        # DB 특징 벡터 로드 및 행렬로 변환
        self.db_names, self.db_matrix = self._load_db(dino_pt_path)
        print(f"[CoarseRetriever] DB 로드 완료: {len(self.db_names)}개 이미지")

    # ──────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────
    def _load_db(self, path: str) -> tuple[list[str], torch.Tensor]:
        """
        dino_feats.pt 로드 후
          - db_names  : ["IMG_3266.JPG", ...] 순서 고정 리스트
          - db_matrix : Tensor [N, 384], L2 정규화 완료 (코사인 유사도 = 내적)
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"DB 파일을 찾을 수 없습니다: {path}\n"
                "먼저 build_db_claude.py 를 실행해 주세요."
            )

        raw: dict = torch.load(path, map_location="cpu")   # {name: Tensor[384]}
        names  = list(raw.keys())
        matrix = torch.stack([raw[n] for n in names], dim=0)   # [N, 384]
        matrix = F.normalize(matrix, dim=1).to(self.device)    # L2 정규화
        return names, matrix

    def _extract_query_feat(self, img_path: str) -> torch.Tensor:
        """쿼리 이미지 → DINOv2 특징 벡터 [384], L2 정규화."""
        img    = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)   # [1, 3, H, W]
        with torch.no_grad():
            feat = self.model(tensor).squeeze(0)   # [384]
        return F.normalize(feat, dim=0)

    # ──────────────────────────────────────────
    # 메인 인터페이스
    # ──────────────────────────────────────────
    def retrieve(self, query_path: str,
                 top_k: int = 2,
                 debug_mode: int = 0,
                 dataset_path: str = "./dataset") -> list[tuple[str, float]]:
        """
        쿼리 이미지에 대해 Top-K DB 이미지를 반환.

        Args:
            query_path   : 쿼리 이미지 경로 (str)
            top_k        : 반환할 상위 이미지 수 (기본 2)
            debug_mode   : 1이면 Coarse 시각화 출력
            dataset_path : DB 이미지 루트 폴더 (시각화 시 사용)

        Returns:
            [(db_img_name, similarity_score), ...] 유사도 내림차순 정렬
        """
        query_feat = self._extract_query_feat(query_path)   # [384]

        # 코사인 유사도 = L2 정규화 벡터의 내적
        scores = self.db_matrix @ query_feat   # [N]

        # Top-K 추출
        k      = min(top_k, len(self.db_names))
        values, indices = torch.topk(scores, k)

        results = [
            (self.db_names[idx.item()], values[rank].item())
            for rank, idx in enumerate(indices)
        ]

        if debug_mode:
            from visualizer import show_coarse
            show_coarse(query_path, results, dataset_path=dataset_path)

        return results


# ──────────────────────────────────────────────
# 단독 실행 (간단 테스트)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    query_path = sys.argv[1] if len(sys.argv) > 1 else "./query/IMG_3342.JPG"
    top_k      = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    if not Path(query_path).exists():
        print(f"❌ 쿼리 이미지를 찾을 수 없습니다: {query_path}")
        sys.exit(1)

    retriever = CoarseRetriever()
    results   = retriever.retrieve(query_path, top_k=top_k)

    print(f"\n[결과] 쿼리: {query_path}")
    for rank, (name, score) in enumerate(results, 1):
        print(f"  Top-{rank}  {name}  (cosine={score:.4f})")
