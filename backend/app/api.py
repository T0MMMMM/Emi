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
