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
