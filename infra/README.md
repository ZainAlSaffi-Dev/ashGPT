# infra/

Deployment + local-orchestration config.

| File | Role | Activated in |
|---|---|---|
| `Containerfile` | Backend container (Python 3.12 + FastAPI). Used by `docker-compose` locally and Cloudflare Containers in prod. | Phase 0 build, Phase 1 runs |
| `wrangler.toml` | Cloudflare Workers / Containers bindings (D1, R2, Vectorize). | Phase 6 |
| `postgres/init.sql` | Enables `pgvector` for local dev parity with Vectorize. | Phase 0 |

## Local

```bash
docker compose up postgres     # Phase 0 — just the DB
docker compose up              # Phase 1+ — DB + backend
docker compose --profile frontend up   # Phase 3+ — also Next.js
```

## Production

Phase 6 wires `wrangler.toml` bindings, runs `wrangler deploy` for the edge Worker and `wrangler containers deploy` for the backend container. Pages auto-deploys from the `main` branch.
