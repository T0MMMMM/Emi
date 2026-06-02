import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


@dataclass
class Settings:
    dataset_id: str = "kuzheren/100k-random-memes"
    # 0 = tout le dataset ; >0 = sous-ensemble (dev rapide)
    index_limit: int = field(default_factory=lambda: _env_int("EMI_INDEX_LIMIT", 0))
    embedding_dim: int = 512
    image_model: str = "clip-ViT-B-32"
    text_model: str = "clip-ViT-B-32-multilingual-v1"
    ocr_languages: str = "eng+fra"
    index_dir: str = field(default_factory=lambda: os.environ.get("EMI_INDEX_DIR", "index"))
    # score cosinus au-dessus duquel on considère un "autre version" du même meme
    version_threshold: float = 0.90
    default_k: int = 24
