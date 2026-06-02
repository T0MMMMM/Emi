import os
from app.config import Settings


def test_defaults():
    s = Settings()
    assert s.dataset_id == "kuzheren/100k-random-memes"
    assert s.embedding_dim == 512
    assert s.index_dir.endswith("index")
    assert 0.0 < s.version_threshold < 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("EMI_INDEX_LIMIT", "500")
    s = Settings()
    assert s.index_limit == 500
