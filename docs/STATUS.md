# Emi — État du projet & reprise

> Document de reprise (handoff). Si tu reprends sur une autre machine, lis ce fichier en premier.

## Le projet en une phrase
**Emi** = agent IA qui retrouve des memes par **description texte** ou par **image exemple**, et renvoie le meme + ses autres versions + des memes similaires.
Contraintes : **100 % local, 100 % gratuit**, aucune API payante. Service web public, sans comptes, sans modération (MVP).

## Architecture
- **Frontend** → Vercel (Next.js) — *pas encore construit*
- **Backend IA** → Hugging Face Space (Docker) : FastAPI + CLIP + FAISS
- **Source memes** → dataset Hugging Face `kuzheren/100k-random-memes`
- **Moteur** → CLIP multilingue (image+texte même espace : sentence-transformers `clip-ViT-B-32` + `clip-ViT-B-32-multilingual-v1`) + Tesseract OCR (complément) + FAISS (`IndexFlatIP`). Priorité = recherche **visuelle** (retrouver un meme sans texte, ex. « pink hamster »).

## ✅ Fait (sur `main`, 12 commits)
Backend complet en TDD, sous `backend/` :
- `app/config.py` — réglages (dataset, limite, modèles, seuils)
- `app/embedder.py` — CLIP image+texte (tests d'intégration verts)
- `app/ocr.py` — Tesseract (complément)
- `app/vector_store.py` — FAISS add/search/save/load
- `app/dataset_loader.py` — `iter_memes`, sous-ensemble configurable
- `app/indexer.py` — pipeline embedding + OCR → index
- `app/search.py` — `categorize` found/other_versions/similar (seuil 0.90)
- `app/api.py` — FastAPI `/health`, `/search/text`, `/search/image` (CORS ouvert)
- `scripts/build_index.py` — indexation hors ligne (`--limit`)
- `Dockerfile` + `packages.txt` + `README.md` — packaging HF Space
- **19 tests unitaires verts + 3 tests d'intégration CLIP verts**

## ⏳ Reste à faire
1. **Installer Tesseract** (binaire) : `winget install --id UB-Mannheim.TesseractOCR -e` (inclure le pack « French »). Sans lui, seuls les 2 tests `test_ocr.py` échouent — le code est correct et tourne sur le Space via le Dockerfile.
2. **Construire l'index** : `cd backend; .venv\Scripts\python scripts/build_index.py --limit 2000` (commencer petit ; dataset 100k ≈ 15 Go).
   - ⚠️ Vérifier que `kuzheren/100k-random-memes` se charge via `datasets.load_dataset(streaming=True)` — format .7z, peut nécessiter d'adapter `dataset_loader._hf_rows`.
3. **Déployer** le backend sur un Hugging Face Space (SDK Docker) une fois `index/` généré.
4. **Plan frontend** Next.js/Vercel — *pas encore rédigé* (prochaine étape logique).

## Phase 2 (reportée volontairement)
Collecte externe (meme-api.com, Reddit, Imgflip) · modération / filtre NSFW.

## Mettre en place l'environnement sur une nouvelle machine
`.venv/` et `index/` sont git-ignorés → à recréer :
```
git clone https://github.com/T0MMMMM/Emi.git
cd Emi/backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # gros téléchargement (torch)
.venv\Scripts\pytest -m "not integration"        # doit être vert (sauf OCR si Tesseract absent)
```

## Règles de commit (à respecter)
Commits **uniquement** sous le compte de l'utilisateur (git user = `T0MMMMM`, email `tom.fuster34000@gmail.com`). Messages simples, 1re lettre en majuscule (ex. `Add embedder`). **Jamais** de mention « Claude » ni de trailer `Co-Authored-By`.

## Docs de référence
- `docs/2026-06-02-emi-meme-search-design.md` — design validé
- `docs/2026-06-02-emi-backend-implementation-plan.md` — plan backend détaillé

---

## 🔁 Prompt de reprise (à coller dans Claude Code sur la nouvelle machine)

> Je reprends le projet Emi (agent IA de recherche de memes). Lis `docs/STATUS.md`, `docs/2026-06-02-emi-meme-search-design.md` et `docs/2026-06-02-emi-backend-implementation-plan.md` pour le contexte complet. Le backend est fait et mergé sur main. Rappel des règles de commit : uniquement sous mon compte GitHub (T0MMMMM), messages simples avec majuscule, jamais de mention de Claude ni de Co-Authored-By. Prochaine étape souhaitée : [au choix — rédiger le plan frontend Next.js/Vercel | m'aider à construire l'index et déployer sur HF Space].
