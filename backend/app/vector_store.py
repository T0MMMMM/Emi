import json
import os
import faiss
import numpy as np


class FaissStore:
    """Index FAISS (produit scalaire = cosinus sur vecteurs normalisés) + métadonnées.

    Pour 100k vecteurs, IndexFlatIP est exact et assez rapide sur CPU.
    Levier d'échelle ultérieur : remplacer par un index HNSW/IVF (même interface).
    """

    INDEX_FILE = "index.faiss"
    META_FILE = "meta.json"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, ids, vectors, metadatas) -> None:
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"vecteurs attendus en (N, {self.dim}), reçu {matrix.shape}")
        self.index.add(matrix)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def search(self, vector, k: int) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query = np.asarray([vector], dtype="float32")
        scores, idxs = self.index.search(query, min(k, self.index.ntotal))
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            results.append(
                {"id": self.ids[i], "score": float(score), "metadata": self.metadatas[i]}
            )
        return results

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, self.INDEX_FILE))
        with open(os.path.join(directory, self.META_FILE), "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "ids": self.ids, "metadatas": self.metadatas}, f)

    @classmethod
    def load(cls, directory: str) -> "FaissStore":
        with open(os.path.join(directory, cls.META_FILE), encoding="utf-8") as f:
            data = json.load(f)
        store = cls(dim=data["dim"])
        store.index = faiss.read_index(os.path.join(directory, cls.INDEX_FILE))
        store.ids = data["ids"]
        store.metadatas = data["metadatas"]
        return store
