# Emi Backend (recherche de memes) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire le backend d'Emi : une API FastAPI qui retrouve des memes par description textuelle ou par image exemple, et renvoie le meme trouvé, ses autres versions et des memes similaires.

**Architecture:** Pipeline en composants isolés. Un *loader* lit le dataset Hugging Face, un *embedder* CLIP multilingue transforme images et textes en vecteurs dans un espace commun, un *OCR* extrait le texte écrit sur les memes (complément), un *vector store* FAISS stocke les vecteurs et fait la recherche des plus proches voisins, un *search engine* classe les résultats en catégories, et une *API FastAPI* expose le tout. L'indexation est faite une fois hors ligne via un script, puis l'index est rechargé au démarrage.

**Tech Stack:** Python 3.11, FastAPI + Uvicorn, sentence-transformers (`clip-ViT-B-32` pour les images, `clip-ViT-B-32-multilingual-v1` pour le texte), faiss-cpu, pytesseract (Tesseract OCR), datasets (Hugging Face), Pillow, pytest.

---

## File Structure

```
backend/
  requirements.txt          # dépendances Python
  packages.txt              # paquets système pour HF Space (tesseract)
  Dockerfile                # image du HF Space (Docker SDK)
  app/
    __init__.py
    config.py               # Settings : dataset, limite, modèles, chemins, seuils
    embedder.py             # _normalize() + CLIPEmbedder.embed_image/embed_text
    ocr.py                  # extract_text(image) -> str
    vector_store.py         # FaissStore : add / search / save / load
    dataset_loader.py       # iter_memes(limit) -> (id, PIL.Image, metadata)
    indexer.py              # build_index(loader, embedder, ocr, store, limit)
    search.py               # SearchEngine.search_text / search_image (catégorisé)
    api.py                  # app FastAPI : /health, /search/text, /search/image
  scripts/
    build_index.py          # CLI : construit et sauvegarde l'index FAISS
  tests/
    __init__.py
    conftest.py             # fixtures (images synthétiques, fake embedder/store)
    test_embedder.py
    test_ocr.py
    test_vector_store.py
    test_indexer.py
    test_search.py
    test_api.py
```

Chaque fichier a une seule responsabilité. Le `search engine` et l'`api` dépendent d'interfaces (embedder, store) injectées, donc testables avec des doublures rapides — aucun test n'a besoin de télécharger le dataset de 15 Go ni de charger les modèles, sauf un test d'intégration explicite de l'embedder.

**Conventions partagées (types utilisés dans tout le plan) :**
- Un **vecteur** = `numpy.ndarray` de forme `(512,)`, dtype `float32`, **normalisé L2** (norme = 1) → le produit scalaire FAISS = similarité cosinus.
- Un **résultat** = `dict` `{"id": str, "score": float, "metadata": dict}`.
- `metadata` contient au moins `{"image_ref": str, "ocr_text": str}`.
- La réponse de recherche = `dict` `{"found": result|None, "other_versions": [result], "similar": [result]}`.

---

### Task 0: Scaffold du projet

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py` (vide)
- Create: `backend/tests/__init__.py` (vide)
- Create: `backend/.gitignore`

- [ ] **Step 1: Créer `backend/requirements.txt`**

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
sentence-transformers==3.*
faiss-cpu==1.9.*
pillow==11.*
pytesseract==0.3.*
datasets==3.*
python-multipart==0.0.*
pytest==8.*
httpx==0.27.*
```

- [ ] **Step 2: Créer les fichiers `__init__.py` vides**

`backend/app/__init__.py` et `backend/tests/__init__.py` : fichiers vides.

- [ ] **Step 3: Créer `backend/.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
data/
index/
.venv/
```

- [ ] **Step 4: Créer l'environnement et installer**

Run (depuis `backend/`) :
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Expected: installation OK (peut prendre plusieurs minutes, télécharge torch).

- [ ] **Step 5: Vérifier que pytest tourne (aucun test encore)**

Run: `.venv\Scripts\pytest -q`
Expected: `no tests ran` (exit 0 ou 5), pas d'erreur d'import.

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "chore: scaffold backend Emi (deps, structure)"
```

---

### Task 1: Configuration

**Files:**
- Create: `backend/app/config.py`
- Test: `backend/tests/test_config.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_config.py
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
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/config.py
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
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py backend/tests/test_config.py
git commit -m "feat: config Settings du backend Emi"
```

---

### Task 2: Normalisation des vecteurs (fonction pure)

**Files:**
- Create: `backend/app/embedder.py`
- Test: `backend/tests/test_embedder.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_embedder.py
import numpy as np
from app.embedder import _normalize


def test_normalize_unit_norm():
    vec = np.array([3.0, 4.0], dtype="float32")
    out = _normalize(vec)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert out.dtype == np.float32


def test_normalize_zero_vector_is_safe():
    vec = np.zeros(4, dtype="float32")
    out = _normalize(vec)
    # pas de division par zéro : reste à zéro
    assert np.all(out == 0.0)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_embedder.py -v`
Expected: FAIL — `ImportError: cannot import name '_normalize'`.

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/embedder.py
import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise L2 un vecteur en float32. Vecteur nul -> reste nul (pas de NaN)."""
    vec = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype("float32")
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_embedder.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/embedder.py backend/tests/test_embedder.py
git commit -m "feat: _normalize pour les vecteurs CLIP"
```

---

### Task 3: CLIPEmbedder (chargement paresseux des modèles)

**Files:**
- Modify: `backend/app/embedder.py`
- Test: `backend/tests/test_embedder.py` (ajout)

- [ ] **Step 1: Ajouter le test d'intégration (télécharge les modèles une fois)**

```python
# backend/tests/test_embedder.py  (ajouter en bas)
import pytest
from PIL import Image
from app.embedder import CLIPEmbedder


@pytest.fixture(scope="module")
def embedder():
    return CLIPEmbedder()


@pytest.mark.integration
def test_embed_text_shape_and_norm(embedder):
    vec = embedder.embed_text("a pink hamster")
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


@pytest.mark.integration
def test_embed_image_shape_and_norm(embedder):
    img = Image.new("RGB", (64, 64), color=(255, 0, 128))
    vec = embedder.embed_image(img)
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


@pytest.mark.integration
def test_text_and_image_share_space(embedder):
    # une description doit être plus proche de l'image correspondante que d'une autre
    img = Image.new("RGB", (64, 64), color=(255, 0, 128))
    img_vec = embedder.embed_image(img)
    close = float(np.dot(embedder.embed_text("a pink square"), img_vec))
    far = float(np.dot(embedder.embed_text("a green forest landscape"), img_vec))
    assert close > far
```

- [ ] **Step 2: Enregistrer le marqueur `integration` et lancer (échec)**

Créer `backend/pytest.ini` :
```ini
[pytest]
markers =
    integration: tests qui chargent les modèles ou des ressources lourdes
```

Run: `.venv\Scripts\pytest tests/test_embedder.py -m integration -v`
Expected: FAIL — `ImportError: cannot import name 'CLIPEmbedder'`.

- [ ] **Step 3: Implémenter `CLIPEmbedder`**

```python
# backend/app/embedder.py  (ajouter en bas)
from functools import cached_property
from PIL import Image
from sentence_transformers import SentenceTransformer
from app.config import Settings


class CLIPEmbedder:
    """Encode images et textes dans le MÊME espace CLIP 512-D (cosinus).

    - images : modèle CLIP visuel
    - texte  : modèle CLIP texte multilingue, aligné sur le même espace
    Les modèles sont chargés à la première utilisation (lazy).
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()

    @cached_property
    def _image_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._settings.image_model)

    @cached_property
    def _text_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._settings.text_model)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        vec = self._image_model.encode(image.convert("RGB"))
        return _normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        vec = self._text_model.encode(text)
        return _normalize(vec)
```

- [ ] **Step 4: Lancer les tests d'intégration pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_embedder.py -m integration -v`
Expected: PASS (3 tests). Premier run lent (téléchargement des modèles).

- [ ] **Step 5: Commit**

```bash
git add backend/app/embedder.py backend/tests/test_embedder.py backend/pytest.ini
git commit -m "feat: CLIPEmbedder (image + texte multilingue, espace commun)"
```

---

### Task 4: OCR (texte écrit sur les memes)

**Files:**
- Create: `backend/app/ocr.py`
- Test: `backend/tests/test_ocr.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_ocr.py
from PIL import Image, ImageDraw
from app.ocr import extract_text


def _image_with_text(text: str) -> Image.Image:
    img = Image.new("RGB", (320, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill="black")
    return img


def test_extract_reads_written_text():
    img = _image_with_text("HELLO")
    assert "HELLO" in extract_text(img).upper()


def test_extract_blank_image_returns_empty():
    img = Image.new("RGB", (320, 80), color="white")
    assert extract_text(img).strip() == ""
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ocr'`.
(Prérequis : Tesseract installé. Windows : installer depuis github.com/UB-Mannheim/tesseract et l'ajouter au PATH. Sinon le test lèvera `TesseractNotFoundError` — c'est attendu tant que ce n'est pas installé.)

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/ocr.py
import pytesseract
from PIL import Image
from app.config import Settings

_SETTINGS = Settings()


def extract_text(image: Image.Image, languages: str | None = None) -> str:
    """Extrait le texte écrit sur une image. Renvoie '' si rien / en cas d'erreur OCR."""
    langs = languages or _SETTINGS.ocr_languages
    try:
        return pytesseract.image_to_string(image.convert("RGB"), lang=langs).strip()
    except pytesseract.TesseractError:
        return ""
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_ocr.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/ocr.py backend/tests/test_ocr.py
git commit -m "feat: extraction OCR du texte des memes"
```

---

### Task 5: Vector store FAISS

**Files:**
- Create: `backend/app/vector_store.py`
- Test: `backend/tests/test_vector_store.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_vector_store.py
import numpy as np
from app.vector_store import FaissStore


def _unit(*values) -> np.ndarray:
    v = np.array(values, dtype="float32")
    return v / np.linalg.norm(v)


def test_search_returns_nearest_first():
    store = FaissStore(dim=2)
    store.add(
        ids=["a", "b"],
        vectors=[_unit(1, 0), _unit(0, 1)],
        metadatas=[{"image_ref": "a.jpg"}, {"image_ref": "b.jpg"}],
    )
    results = store.search(_unit(1, 0), k=2)
    assert results[0]["id"] == "a"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["metadata"]["image_ref"] == "a.jpg"


def test_save_and_load_roundtrip(tmp_path):
    store = FaissStore(dim=2)
    store.add(["a"], [_unit(1, 0)], [{"image_ref": "a.jpg", "ocr_text": "hi"}])
    store.save(str(tmp_path))

    loaded = FaissStore.load(str(tmp_path))
    results = loaded.search(_unit(1, 0), k=1)
    assert results[0]["id"] == "a"
    assert results[0]["metadata"]["ocr_text"] == "hi"


def test_empty_store_search_returns_empty():
    store = FaissStore(dim=2)
    assert store.search(_unit(1, 0), k=5) == []
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_vector_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.vector_store'`.

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/vector_store.py
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
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_vector_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/vector_store.py backend/tests/test_vector_store.py
git commit -m "feat: FaissStore (add/search/save/load)"
```

---

### Task 6: Dataset loader

**Files:**
- Create: `backend/app/dataset_loader.py`
- Test: `backend/tests/test_dataset_loader.py`

Le loader expose une interface simple — un itérateur de `(id, PIL.Image, metadata)` — et isole le détail du dataset HF. Les tests utilisent un itérable synthétique injecté ; le vrai téléchargement n'est jamais déclenché en test.

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_dataset_loader.py
from PIL import Image
from app.dataset_loader import iter_memes


def _fake_rows():
    for i in range(5):
        yield {"image": Image.new("RGB", (8, 8)), "id": f"row-{i}"}


def test_iter_memes_applies_limit():
    items = list(iter_memes(limit=3, rows=_fake_rows()))
    assert len(items) == 3


def test_iter_memes_zero_limit_means_all():
    items = list(iter_memes(limit=0, rows=_fake_rows()))
    assert len(items) == 5


def test_iter_memes_yields_id_image_metadata():
    meme_id, image, metadata = next(iter(iter_memes(limit=1, rows=_fake_rows())))
    assert isinstance(meme_id, str)
    assert isinstance(image, Image.Image)
    assert "image_ref" in metadata
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_dataset_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.dataset_loader'`.

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/dataset_loader.py
from typing import Iterable, Iterator
from PIL import Image
from app.config import Settings


def _hf_rows(settings: Settings) -> Iterable[dict]:
    """Lit le dataset Hugging Face en streaming (téléchargement réel)."""
    from datasets import load_dataset

    return load_dataset(settings.dataset_id, split="train", streaming=True)


def iter_memes(
    limit: int = 0,
    rows: Iterable[dict] | None = None,
    settings: Settings | None = None,
) -> Iterator[tuple[str, Image.Image, dict]]:
    """Itère sur les memes en (id, image, metadata).

    `rows` permet d'injecter une source (tests). Sinon, lit le dataset HF.
    `limit=0` => tout ; `limit>0` => sous-ensemble.
    """
    settings = settings or Settings()
    source = rows if rows is not None else _hf_rows(settings)
    for i, row in enumerate(source):
        if limit and i >= limit:
            break
        image = row["image"]
        meme_id = str(row.get("id", i))
        metadata = {"image_ref": f"{settings.dataset_id}#{meme_id}"}
        yield meme_id, image, metadata
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_dataset_loader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/dataset_loader.py backend/tests/test_dataset_loader.py
git commit -m "feat: dataset_loader (iter_memes, source injectable)"
```

---

### Task 7: Indexer

**Files:**
- Create: `backend/app/indexer.py`
- Test: `backend/tests/test_indexer.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Écrire les doublures partagées dans `conftest.py`**

```python
# backend/tests/conftest.py
import numpy as np
import pytest
from PIL import Image


class FakeEmbedder:
    """Embedder déterministe sans modèle : couleur moyenne -> vecteur normalisé."""

    def embed_image(self, image: Image.Image) -> np.ndarray:
        r, g, b = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0)
        v = np.array([r, g, b, 1.0], dtype="float32")
        return v / np.linalg.norm(v)

    def embed_text(self, text: str) -> np.ndarray:
        # mappe quelques mots-clés vers une couleur pour des tests lisibles
        table = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
        r, g, b = table.get(text.strip().lower(), (0, 0, 0))
        v = np.array([r, g, b, 1.0], dtype="float32")
        return v / np.linalg.norm(v)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def color_image():
    def _make(rgb):
        return Image.new("RGB", (8, 8), color=rgb)
    return _make
```

- [ ] **Step 2: Écrire le test qui échoue**

```python
# backend/tests/test_indexer.py
from app.indexer import build_index
from app.vector_store import FaissStore


def test_build_index_populates_store(fake_embedder, color_image):
    rows = [
        {"image": color_image((255, 0, 0)), "id": "red"},
        {"image": color_image((0, 255, 0)), "id": "green"},
    ]
    store = FaissStore(dim=4)
    count = build_index(
        store=store,
        embedder=fake_embedder,
        ocr=lambda img: "",
        rows=rows,
        limit=0,
    )
    assert count == 2
    assert store.index.ntotal == 2
    assert store.metadatas[0]["image_ref"].endswith("#red")
```

- [ ] **Step 3: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.indexer'`.

- [ ] **Step 4: Écrire l'implémentation minimale**

```python
# backend/app/indexer.py
from typing import Callable, Iterable
from PIL import Image
from app.dataset_loader import iter_memes
from app.vector_store import FaissStore


def build_index(
    store: FaissStore,
    embedder,
    ocr: Callable[[Image.Image], str],
    rows: Iterable[dict] | None = None,
    limit: int = 0,
) -> int:
    """Indexe les memes : embedding image + OCR -> store. Renvoie le nb indexé.

    Les images illisibles sont ignorées (log via print) pour ne pas tout casser.
    """
    count = 0
    for meme_id, image, metadata in iter_memes(limit=limit, rows=rows):
        try:
            vector = embedder.embed_image(image)
            metadata = {**metadata, "ocr_text": ocr(image)}
            store.add([meme_id], [vector], [metadata])
            count += 1
        except Exception as exc:  # image corrompue / erreur d'encodage
            print(f"[indexer] meme {meme_id} ignoré : {exc}")
    return count
```

- [ ] **Step 5: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_indexer.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add backend/app/indexer.py backend/tests/test_indexer.py backend/tests/conftest.py
git commit -m "feat: indexer (embedding + OCR -> FaissStore)"
```

---

### Task 8: Search engine (résultats catégorisés)

**Files:**
- Create: `backend/app/search.py`
- Test: `backend/tests/test_search.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_search.py
from app.search import SearchEngine, categorize
from app.vector_store import FaissStore


def test_categorize_splits_found_versions_similar():
    results = [
        {"id": "a", "score": 0.99, "metadata": {}},
        {"id": "b", "score": 0.95, "metadata": {}},  # autre version
        {"id": "c", "score": 0.40, "metadata": {}},  # similaire
    ]
    out = categorize(results, version_threshold=0.90)
    assert out["found"]["id"] == "a"
    assert [r["id"] for r in out["other_versions"]] == ["b"]
    assert [r["id"] for r in out["similar"]] == ["c"]


def test_categorize_empty_results():
    out = categorize([], version_threshold=0.90)
    assert out == {"found": None, "other_versions": [], "similar": []}


def test_search_text_uses_text_embedding(fake_embedder, color_image):
    store = FaissStore(dim=4)
    store.add(["red"], [fake_embedder.embed_image(color_image((255, 0, 0)))], [{"image_ref": "r"}])
    store.add(["blue"], [fake_embedder.embed_image(color_image((0, 0, 255)))], [{"image_ref": "b"}])

    engine = SearchEngine(store=store, embedder=fake_embedder, version_threshold=0.90)
    out = engine.search_text("red", k=2)
    assert out["found"]["id"] == "red"


def test_search_image_uses_image_embedding(fake_embedder, color_image):
    store = FaissStore(dim=4)
    store.add(["green"], [fake_embedder.embed_image(color_image((0, 255, 0)))], [{"image_ref": "g"}])

    engine = SearchEngine(store=store, embedder=fake_embedder, version_threshold=0.90)
    out = engine.search_image(color_image((0, 255, 0)), k=1)
    assert out["found"]["id"] == "green"
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.search'`.

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/search.py
from PIL import Image


def categorize(results: list[dict], version_threshold: float) -> dict:
    """Classe les résultats triés par score décroissant en found / versions / similar."""
    if not results:
        return {"found": None, "other_versions": [], "similar": []}
    found, *rest = results
    other_versions = [r for r in rest if r["score"] >= version_threshold]
    similar = [r for r in rest if r["score"] < version_threshold]
    return {"found": found, "other_versions": other_versions, "similar": similar}


class SearchEngine:
    """Recherche par texte ou image -> résultats catégorisés."""

    def __init__(self, store, embedder, version_threshold: float = 0.90):
        self._store = store
        self._embedder = embedder
        self._version_threshold = version_threshold

    def search_text(self, query: str, k: int) -> dict:
        vector = self._embedder.embed_text(query)
        return categorize(self._store.search(vector, k), self._version_threshold)

    def search_image(self, image: Image.Image, k: int) -> dict:
        vector = self._embedder.embed_image(image)
        return categorize(self._store.search(vector, k), self._version_threshold)
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_search.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/search.py backend/tests/test_search.py
git commit -m "feat: SearchEngine + categorize (found/versions/similar)"
```

---

### Task 9: API FastAPI

**Files:**
- Create: `backend/app/api.py`
- Test: `backend/tests/test_api.py`

L'app charge l'index au démarrage. Pour les tests, on injecte un faux `SearchEngine` via `app.state` sans charger ni index ni modèle.

- [ ] **Step 1: Écrire le test qui échoue**

```python
# backend/tests/test_api.py
from fastapi.testclient import TestClient
from app.api import create_app


class StubEngine:
    def search_text(self, query, k):
        return {"found": {"id": "x", "score": 0.99, "metadata": {"image_ref": "x"}},
                "other_versions": [], "similar": []}

    def search_image(self, image, k):
        return {"found": {"id": "y", "score": 0.98, "metadata": {"image_ref": "y"}},
                "other_versions": [], "similar": []}


def _client():
    app = create_app(engine=StubEngine())
    return TestClient(app)


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_text():
    resp = _client().post("/search/text", json={"query": "pink hamster", "k": 10})
    assert resp.status_code == 200
    assert resp.json()["found"]["id"] == "x"


def test_search_text_requires_query():
    resp = _client().post("/search/text", json={})
    assert resp.status_code == 422


def test_search_image():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    resp = _client().post("/search/image", files={"file": ("m.png", buf, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["found"]["id"] == "y"
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api'`.

- [ ] **Step 3: Écrire l'implémentation minimale**

```python
# backend/app/api.py
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from app.config import Settings


class TextQuery(BaseModel):
    query: str
    k: int | None = None


def _load_engine() -> "object":
    """Charge l'index FAISS et le vrai SearchEngine (au démarrage de l'app)."""
    from app.embedder import CLIPEmbedder
    from app.vector_store import FaissStore
    from app.search import SearchEngine

    settings = Settings()
    store = FaissStore.load(settings.index_dir)
    return SearchEngine(
        store=store,
        embedder=CLIPEmbedder(settings),
        version_threshold=settings.version_threshold,
    )


def create_app(engine=None) -> FastAPI:
    settings = Settings()
    app = FastAPI(title="Emi — recherche de memes")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # MVP ; à restreindre au domaine Vercel ensuite
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine  # injecté en test ; sinon chargé au startup

    @app.on_event("startup")
    def _startup():
        if app.state.engine is None:
            app.state.engine = _load_engine()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/search/text")
    def search_text(body: TextQuery):
        k = body.k or settings.default_k
        return app.state.engine.search_text(body.query, k)

    @app.post("/search/image")
    async def search_image(file: UploadFile = File(...)):
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return app.state.engine.search_image(image, settings.default_k)

    return app


app = create_app()
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\pytest tests/test_api.py -v`
Expected: PASS (4 tests). (Le stub évite tout chargement de modèle/index.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/api.py backend/tests/test_api.py
git commit -m "feat: API FastAPI (/health, /search/text, /search/image)"
```

---

### Task 10: Script de construction de l'index (hors ligne)

**Files:**
- Create: `backend/scripts/build_index.py`

Ce script est un point d'entrée d'intégration (télécharge le dataset, charge les modèles). Pas de test unitaire — vérification manuelle sur un petit sous-ensemble.

- [ ] **Step 1: Écrire le script**

```python
# backend/scripts/build_index.py
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
```

- [ ] **Step 2: Vérification manuelle sur un petit sous-ensemble**

Run (depuis `backend/`) :
```bash
.venv\Scripts\python scripts/build_index.py --limit 50
```
Expected: affiche « Terminé : 50 memes indexés -> index/ », et le dossier `index/` contient `index.faiss` + `meta.json`.

- [ ] **Step 3: Vérifier la recherche de bout en bout (manuel)**

Run :
```bash
.venv\Scripts\uvicorn app.api:app --port 8000
```
Puis dans un autre terminal :
```bash
curl -X POST http://localhost:8000/search/text -H "Content-Type: application/json" -d "{\"query\": \"cat\", \"k\": 5}"
```
Expected: JSON avec `found`, `other_versions`, `similar`.

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/build_index.py
git commit -m "feat: script build_index (indexation hors ligne)"
```

---

### Task 11: Packaging Hugging Face Space (Docker)

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/packages.txt`
- Create: `backend/README.md`

- [ ] **Step 1: Créer `backend/packages.txt`**

```
tesseract-ocr
tesseract-ocr-eng
tesseract-ocr-fra
```

- [ ] **Step 2: Créer `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY index ./index

EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 3: Créer `backend/README.md` (front matter HF Space)**

```markdown
---
title: Emi Backend
emoji: 🔎
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
---

# Emi — backend de recherche de memes

API FastAPI. Endpoints : `/health`, `/search/text`, `/search/image`.

L'index FAISS (`index/`) doit être construit au préalable via
`python scripts/build_index.py` puis commité (ou monté) avec l'image.
```

- [ ] **Step 4: Vérifier le build Docker localement (optionnel mais recommandé)**

Run (depuis `backend/`, nécessite un `index/` déjà construit) :
```bash
docker build -t emi-backend .
docker run -p 7860:7860 emi-backend
```
Expected: l'API démarre ; `curl http://localhost:7860/health` renvoie `{"status":"ok"}`.

- [ ] **Step 5: Commit**

```bash
git add backend/Dockerfile backend/packages.txt backend/README.md
git commit -m "chore: packaging HF Space (Docker + tesseract)"
```

---

### Task 12: Suite complète & verrouillage

**Files:** aucun nouveau.

- [ ] **Step 1: Lancer toute la suite hors intégration**

Run (depuis `backend/`) :
```bash
.venv\Scripts\pytest -m "not integration" -v
```
Expected: tous les tests PASS (config, embedder/_normalize, ocr, vector_store, dataset_loader, indexer, search, api).

- [ ] **Step 2: Lancer les tests d'intégration (modèles)**

Run: `.venv\Scripts\pytest -m integration -v`
Expected: les 3 tests de l'embedder PASS.

- [ ] **Step 3: Commit final (si changements)**

```bash
git add -A
git commit -m "test: suite backend complète au vert"
```

---

## Self-Review

**Couverture de la spec :**
- Recherche par description → Task 8/9 (`search_text`). ✅
- Recherche par image exemple → Task 8/9 (`search_image`). ✅
- Meme trouvé + autres versions + similaires → `categorize` (Task 8). ✅
- Priorité visuelle (memes sans texte) → CLIP image embedding prioritaire, OCR en métadonnée complémentaire (Task 3/7). ✅
- Multilingue (fr/en) → `clip-ViT-B-32-multilingual-v1` + OCR `eng+fra` (Task 3/4). ✅
- Dataset `kuzheren/100k-random-memes`, sous-ensemble configurable → Task 1 (`index_limit`) + Task 6 + Task 10 (`--limit`). ✅
- FAISS → Task 5. ✅
- FastAPI → Task 9. ✅
- Pas de stockage des images sur le Space (réf seulement) → `metadata.image_ref` (Task 6). ✅
- Indexation hors ligne une fois puis rechargée → Task 10 + `_load_engine` (Task 9). ✅
- Gestion d'erreurs (image corrompue ignorée, OCR vide toléré) → Task 7 + Task 4. ✅
- Hébergement HF Space → Task 11. ✅
- Hors périmètre (collecte externe, modération) → absents du plan, conformes à la phase 2. ✅

**Scan placeholders :** aucun TODO/TBD ; chaque step de code contient le code complet. ✅

**Cohérence des types :** `result` = `{id, score, metadata}` partout (vector_store → search → api) ; `search_text`/`search_image` cohérents entre SearchEngine, API et stubs de test ; `FaissStore(dim=...)`, `.add(ids, vectors, metadatas)`, `.search(vector, k)`, `.save/.load` cohérents entre tasks 5/7/8/9/10. ✅

**Note frontend :** le frontend Next.js/Vercel fera l'objet d'un second plan (consomme `/search/text` et `/search/image`). Le CORS est déjà ouvert côté API (Task 9) pour permettre l'appel depuis Vercel.
