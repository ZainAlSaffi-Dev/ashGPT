"""Phase 1 tests: configuration, embeddings, PDF extraction, and ChromaDB indexing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    VLM_MODEL,
)


# ── Config sanity checks ──────────────────────────────────────────────────────


class TestConfig:
    def test_data_dir_exists(self) -> None:
        assert DATA_DIR.exists(), f"{DATA_DIR} does not exist"

    def test_data_dir_has_weeks(self) -> None:
        weeks = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("week")]
        assert len(weeks) > 0, "No week_* directories found in data/"

    def test_chunk_params_sensible(self) -> None:
        assert CHUNK_SIZE > CHUNK_OVERLAP, "CHUNK_SIZE must exceed CHUNK_OVERLAP"
        assert CHUNK_SIZE >= 500, "CHUNK_SIZE too small for legal text"

    def test_embedding_model_set(self) -> None:
        assert EMBEDDING_MODEL == "zembed-1"
        assert EMBEDDING_DIMENSIONS > 0

    def test_vlm_model_set(self) -> None:
        assert VLM_MODEL, "VLM_MODEL must not be empty"

    def test_chroma_collection_set(self) -> None:
        assert CHROMA_COLLECTION, "CHROMA_COLLECTION must not be empty"


# ── PDF extraction ─────────────────────────────────────────────────────────────


class TestPDFExtraction:
    def test_extract_text_returns_content(self) -> None:
        from src.indexing.build_index import extract_text_from_pdf

        pdfs = list(DATA_DIR.rglob("*.pdf"))
        assert len(pdfs) > 0, "No PDFs found in data/"

        text = extract_text_from_pdf(pdfs[0])
        assert len(text.strip()) > 0, f"Empty text extracted from {pdfs[0].name}"

    def test_extract_text_all_pdfs_non_empty(self) -> None:
        from src.indexing.build_index import extract_text_from_pdf

        pdfs = list(DATA_DIR.rglob("*.pdf"))
        for pdf_path in pdfs:
            text = extract_text_from_pdf(pdf_path)
            assert len(text.strip()) > 0, f"Empty text from {pdf_path}"


# ── Embeddings ─────────────────────────────────────────────────────────────────


class TestZeroEntropyEmbeddings:
    @pytest.mark.integration
    def test_embed_query_returns_correct_dims(self) -> None:
        from src.embeddings import ZeroEntropyEmbeddings

        emb = ZeroEntropyEmbeddings()
        result = emb.embed_query("What is adverse possession?")
        assert len(result) == EMBEDDING_DIMENSIONS

    @pytest.mark.integration
    def test_embed_documents_batch(self) -> None:
        from src.embeddings import ZeroEntropyEmbeddings

        emb = ZeroEntropyEmbeddings()
        texts = ["First document.", "Second document.", "Third document."]
        results = emb.embed_documents(texts)
        assert len(results) == 3
        assert all(len(r) == EMBEDDING_DIMENSIONS for r in results)

    @pytest.mark.integration
    def test_embed_documents_empty_list(self) -> None:
        from src.embeddings import ZeroEntropyEmbeddings

        emb = ZeroEntropyEmbeddings()
        results = emb.embed_documents([])
        assert results == []

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZEMBED_API_KEY", raising=False)

        from src.embeddings import ZeroEntropyEmbeddings

        with pytest.raises(ValueError, match="ZEMBED_API_KEY"):
            ZeroEntropyEmbeddings(api_key="")


# ── ChromaDB index integrity ──────────────────────────────────────────────────


class TestChromaDBIndex:
    @pytest.mark.integration
    def test_chroma_dir_exists(self) -> None:
        assert CHROMA_DIR.exists(), (
            f"{CHROMA_DIR} not found — run 'python -m src.indexing.build_index' first"
        )

    @pytest.mark.integration
    def test_collection_has_documents(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(CHROMA_COLLECTION)
        count = collection.count()
        assert count > 0, "ChromaDB collection is empty"

    @pytest.mark.integration
    def test_collection_has_text_and_slides(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(CHROMA_COLLECTION)

        text_results = collection.get(where={"type": {"$in": ["reading", "tutorial"]}})
        slide_results = collection.get(where={"type": "lecture_slide"})

        assert len(text_results["ids"]) > 0, "No text documents in collection"
        assert len(slide_results["ids"]) > 0, "No slide documents in collection"

    @pytest.mark.integration
    def test_metadata_has_required_fields(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(CHROMA_COLLECTION)

        sample = collection.peek(limit=5)
        for meta in sample["metadatas"]:
            assert "week" in meta, f"Missing 'week' metadata: {meta}"
            assert "type" in meta, f"Missing 'type' metadata: {meta}"
            assert "source" in meta, f"Missing 'source' metadata: {meta}"

    @pytest.mark.integration
    def test_slides_have_image_path(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(CHROMA_COLLECTION)

        slides = collection.get(where={"type": "lecture_slide"}, limit=5)
        for meta in slides["metadatas"]:
            assert "image_path" in meta, f"Slide missing 'image_path': {meta}"
            assert meta["image_path"], "image_path is empty"
