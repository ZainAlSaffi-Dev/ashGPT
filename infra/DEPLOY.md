# Deploy LawGPT to Cloudflare

End-to-end recipe to take a green `main` branch to a live `lawgpt.<your-domain>`.

## Prereqs

- Cloudflare account with Workers, R2, D1, Vectorize, Containers, Pages enabled
- `wrangler` CLI (`npm i -g wrangler`)
- Clerk project (https://dashboard.clerk.com)

## One-time setup

```bash
# 1. R2 bucket for raw uploads
wrangler r2 bucket create lawgpt-uploads

# 2. D1 for relational metadata
wrangler d1 create lawgpt-prod
# Copy the database_id printed here into infra/wrangler.toml
wrangler d1 execute lawgpt-prod --file=infra/d1-schema.sql

# 3. Vectorize index (2560 dims = ZeroEntropy zembed-1)
wrangler vectorize create lawgpt-vectors --dimensions=2560 --metric=cosine

# 4. R2 S3-API token (for the Worker's presign path)
#    Cloudflare dashboard → R2 → Manage R2 API Tokens → Create.
#    Save Access Key ID + Secret Access Key.

# 5. Worker secrets
cd infra/worker
wrangler secret put CLERK_SECRET_KEY
wrangler secret put R2_ACCESS_KEY_ID
wrangler secret put R2_SECRET_ACCESS_KEY
wrangler secret put R2_ACCOUNT_ID
wrangler secret put OPENAI_API_KEY
wrangler secret put GOOGLE_API_KEY
wrangler secret put ZEMBED_API_KEY
wrangler secret put COHERE_API_KEY

# 6. Build + deploy the backend container + Worker
wrangler deploy

# 7. Frontend (Cloudflare Pages, connected to the GitHub repo)
cd ../../frontend
wrangler pages project create lawgpt --production-branch main
wrangler pages secret put NEXT_PUBLIC_API_BASE          # e.g. https://api.lawgpt.<your-domain>
wrangler pages secret put NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
wrangler pages secret put CLERK_SECRET_KEY
wrangler pages deploy .next --project-name lawgpt
```

## GitHub Actions

Repo secrets to add:

- `CLOUDFLARE_API_TOKEN` (Account + Workers + Pages + D1 + R2 + Vectorize edit)
- `CLOUDFLARE_ACCOUNT_ID`

Worker routes/custom domains are created once in the Cloudflare dashboard and
are not managed by CI. If `wrangler.toml` is changed to manage routes again,
also grant the token `Zone: Read` and `Workers Routes: Edit` for `ashgpt.xyz`.

Frontend deploys must use `wrangler pages deploy .vercel/output/static`.
Plain `wrangler deploy` is for Workers, not Pages; from `frontend/` it fails
with "Missing entry-point to Worker script or to assets directory".
The frontend Pages config pins `NODE_VERSION=20` in `frontend/wrangler.toml`
so dashboard builds do not drift to newer build-image Node defaults than CI.

Workflows:

- `.github/workflows/ci.yml` — runs backend pytest, frontend vitest + tsc, Worker tsc on every push and PR.
- `.github/workflows/deploy.yml` — on push to `main`, deploys the frontend (Pages) and Worker+Container (wrangler).

## Bindings recap (wrangler.toml)

| Binding   | What                  | Used by                                  |
|-----------|-----------------------|------------------------------------------|
| `UPLOADS` | R2 bucket             | Worker presign + Container reads         |
| `VECTORS` | Vectorize index       | Container (via REST) for upsert + query  |
| `DB`      | D1 database           | Container (via REST) for relational data |
| `BACKEND` | Durable Object        | Worker proxy to FastAPI container        |

## Verification

After deploy:

```bash
# Health
curl https://api.lawgpt.<your-domain>/health
# Authed (replace with a real Clerk token via the frontend devtools network tab)
curl -H "Authorization: Bearer $TOKEN" https://api.lawgpt.<your-domain>/sessions
```

Then sign in on the Pages URL, drop a PDF in `/library`, ask a question in `/chat`,
generate an exam in `/exam`. End-to-end smoke.
