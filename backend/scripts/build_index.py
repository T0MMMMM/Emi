import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.embedder import CLIPEmbedder
from app.indexer import build_index
from app.ocr import extract_text
from app.vector_store import FaissStore


def main():
    parser = argparse.ArgumentParser(description="Construit l'index FAISS d'Emi")
    parser.add_argument("--limit", type=int, default=None,
                        help="nb de memes à indexer (0 = tout). Défaut: valeur de config.")
    args = parser.parse_args()

    settings = Settings()
    limit = settings.index_limit if args.limit is None else args.limit

    store = FaissStore(dim=settings.embedding_dim)
    embedder = CLIPEmbedder(settings)

    print(f"Indexation de '{settings.dataset_id}' (limit={limit or 'tout'})…")
    count = build_index(store=store, embedder=embedder, ocr=extract_text, limit=limit)

    store.save(settings.index_dir)
    print(f"Terminé : {count} memes indexés -> {settings.index_dir}/")


if __name__ == "__main__":
    main()
