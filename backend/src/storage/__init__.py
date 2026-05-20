"""Storage adapters (relational DB, vector store, blob store).

Each adapter has a local-dev impl and a Cloudflare-prod impl behind the same
interface, selected via ``get_settings().*_backend``. Importing this package is
side-effect-free; concrete backends are constructed via ``get_db()``,
``get_vector_store()`` and ``get_blob_store()``.
"""
