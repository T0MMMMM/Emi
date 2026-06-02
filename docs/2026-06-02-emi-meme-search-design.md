# Emi — Moteur de recherche de memes (Design MVP)

**Date :** 2026-06-02
**Statut :** Design validé — prêt pour le plan d'implémentation

## Vision

Emi est un agent web qui retrouve des memes de deux façons :
- **par description textuelle** (ex. « pink hamster », « chien dans une maison en feu »),
- **par photo exemple** (l'utilisateur importe une image).

Pour chaque recherche, Emi renvoie le meme correspondant, ses **autres versions** et des
**memes similaires**. La priorité est donnée à la **recherche visuelle** : retrouver un meme
décrit par son contenu visuel (« hamster rose ») même quand l'image ne contient **aucun texte**.

Contraintes fondatrices : **100 % local, 100 % gratuit**, aucune API payante.

## Périmètre MVP

Inclus :
- Une seule source : le **dataset Hugging Face** `kuzheren/100k-random-memes` (~100 000 memes,
  archive ~15 Go). Le choix du dataset est isolé pour être remplaçable facilement.
- **Indexation par sous-ensemble configurable** : objectif 100k, mais on peut indexer un
  échantillon (ex. 5–10k) pour itérer vite en développement. Le code reste identique.
- Indexation des memes : embedding CLIP (visuel, prioritaire) + OCR (texte du meme, complément).
- Recherche par **texte** et par **image**.
- Résultats classés : meme trouvé · autres versions · memes similaires.
- Interface web simple : barre de recherche, import d'image, grille de résultats.

Explicitement hors périmètre (phase 2) :
- Collecte de memes externes (Reddit, meme-api, Imgflip).
- Modération / filtre NSFW.
- Comptes utilisateurs, favoris, collections privées.
- Couche LLM payante.

## Capacités vues par l'utilisateur

1. **Recherche par description** → grille de memes classés du plus au moins pertinent.
2. **Recherche par image** → 3 zones : le meme trouvé · ses autres versions · memes ressemblants.
3. Affichage en grille responsive.

## Architecture

Séparation en deux hébergements, chaque couche ayant une seule responsabilité :

| Couche | Hébergement | Rôle |
|---|---|---|
| **Frontend** | **Vercel** (Next.js/React) | Interface : recherche, upload, grille. Appelle l'API du backend. |
| **Backend IA** | **Hugging Face Space** | Modèle CLIP + OCR + base vectorielle + endpoints de recherche. |

### Composants

1. **Loader de dataset** — charge `kuzheren/100k-random-memes` via la lib `datasets`, expose les
   images + métadonnées. Limite configurable (sous-ensemble). Interface isolée pour changer de
   dataset sans toucher au reste.
2. **Pipeline d'indexation** — pour chaque image : (a) embedding CLIP multilingue,
   (b) OCR Tesseract (texte écrit sur le meme), (c) métadonnées. Produit l'index.
3. **Base vectorielle locale (FAISS)** — stocke les embeddings + métadonnées via un index ANN
   (HNSW/IVF) pour des recherches quasi instantanées à l'échelle de 100k. On **ne stocke pas les
   images** sur le Space, seulement vecteurs + métadonnées + référence vers chaque image
   (~100k vecteurs ≈ 300 Mo). L'index est calculé une fois puis sauvegardé/rechargé.
4. **Moteur de recherche** — convertit la requête (texte ou image) en embedding CLIP, trouve
   les K plus proches voisins, combine avec une correspondance OCR optionnelle, renvoie les
   résultats classés. *Autres versions* = similarité très élevée ; *similaires* = moyenne.
5. **API backend (FastAPI)** — expose `search_text` et `search_image` sur le Space.
6. **Frontend Vercel** — consomme l'API.

## Flux de données

- **Indexation (hors ligne, une seule fois)** : dataset HF (tout ou sous-ensemble) → pipeline
  → index FAISS sauvegardé sur disque. Long sur CPU pour 100k → fait une fois puis rechargé.
- **Recherche texte** : requête → embedding texte CLIP → K plus proches → grille.
- **Recherche image** : upload → embedding image CLIP → meme + autres versions + similaires.

## Stack technique (tout gratuit / libre)

- **Python** côté backend IA.
- **CLIP multilingue** (OpenCLIP ou multilingual-CLIP / SigLIP) — embeddings image **et** texte
  dans le même espace, pour la recherche cross-modale.
- **Tesseract OCR** — extraction du texte des memes (complément).
- **FAISS** — base vectorielle locale (index ANN HNSW/IVF) pour la recherche des plus proches voisins.
- **FastAPI** pour exposer la recherche (`search_text`, `search_image`).
- **Next.js / React** sur **Vercel** pour le front.
- **Hugging Face Space** (CPU gratuit) pour le backend.

## Gestion d'erreurs

- Image corrompue / illisible à l'indexation → on l'ignore + log, on continue.
- OCR vide → on se repose entièrement sur l'embedding visuel (cas nominal pour les memes
  sans texte, ex. « hamster rose »).
- Aucune correspondance pertinente → message clair à l'utilisateur.
- Backend IA injoignable depuis le front → message d'erreur explicite côté Vercel.

## Tests

- Tests unitaires du pipeline : l'embedding et l'OCR produisent une sortie pour une image type.
- Test du loader : le dataset se charge et expose des images valides.
- Test du moteur de recherche sur un petit jeu connu : une requête visuelle (« pink hamster »)
  doit faire remonter l'image attendue dans le top des résultats.
- Test de l'API : `search_text` et `search_image` renvoient une liste classée.

## Évolutions prévues (phase 2+)

- Collecte externe : meme-api.com (sans clé), Reddit JSON, Imgflip (templates → détection des
  « autres versions »).
- Modération : filtre NSFW léger, liste de sources sûres, retrait manuel.
- Plus de sources, recherche web à la demande quand la base ne contient rien.
