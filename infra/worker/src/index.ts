/**
 * LawGPT edge Worker.
 *
 * Responsibilities (kept thin):
 *   1. Verify the Clerk JWT on every authenticated request.
 *   2. Receive authenticated browser uploads and stream them into R2.
 *   3. Proxy everything else to the FastAPI container.
 *
 * The container holds the LangGraph pipeline and DB writes. Keeping the
 * Worker thin means cold-start is low and the heavy Python runtime only
 * runs when needed.
 */

import { Container } from '@cloudflare/containers';
import { createRemoteJWKSet, jwtVerify, type JWTPayload } from 'jose';

export interface Env {
  BACKEND: DurableObjectNamespace;
  UPLOADS: R2Bucket;
  VECTORS: VectorizeIndex; // eslint-disable-line @typescript-eslint/no-explicit-any
  DB: D1Database;
  CLERK_ISSUER: string;
  CLERK_SECRET_KEY: string;
  CLERK_ACCEPTED_ISSUERS?: string;
  CLERK_FAPI?: string;
  CLERK_JWKS_URL?: string;
  CLERK_PROXY_URL?: string;
  CORS_ORIGINS?: string;
  // Backend container secrets — forwarded into the container via envVars.
  DATABASE_URL?: string;
  OPENAI_API_KEY?: string;
  GOOGLE_API_KEY?: string;
  ZEMBED_API_KEY?: string;
  COHERE_API_KEY?: string;
  // R2 S3-API credentials forwarded to the backend container for HEAD/read.
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
  R2_ACCOUNT_ID?: string;
  R2_BUCKET?: string;
}

const CLERK_PROXY_PATH = '/__clerk';
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
function matchesClerkProxyPath(pathname: string): boolean {
  return pathname === CLERK_PROXY_PATH || pathname.startsWith(`${CLERK_PROXY_PATH}/`);
}

function corsHeaders(env: Env, origin: string | null): Record<string, string> {
  const allowed = (env.CORS_ORIGINS ?? '').split(',').map((s) => s.trim()).filter(Boolean);
  const allow = origin && allowed.includes(origin) ? origin : '';
  return {
    'Access-Control-Allow-Origin': allow,
    'Vary': 'Origin',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
    'Access-Control-Allow-Headers': 'authorization,content-type,x-dev-user',
    'Access-Control-Max-Age': '86400',
  };
}

function withCors(res: Response, env: Env, origin: string | null): Response {
  const headers = new Headers(res.headers);
  for (const [k, v] of Object.entries(corsHeaders(env, origin))) {
    if (v) headers.set(k, v);
  }
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

let _jwks: ReturnType<typeof createRemoteJWKSet> | null = null;

function getJwks(env: Env) {
  if (_jwks) return _jwks;
  const url = new URL(
    env.CLERK_JWKS_URL || '.well-known/jwks.json',
    env.CLERK_ISSUER.replace(/\/$/, '') + '/',
  );
  _jwks = createRemoteJWKSet(url);
  return _jwks;
}

function getAcceptedIssuers(env: Env): string[] {
  return (env.CLERK_ACCEPTED_ISSUERS ?? env.CLERK_ISSUER)
    .split(',')
    .map((issuer) => issuer.trim().replace(/\/$/, ''))
    .filter(Boolean);
}

async function verifyClerk(req: Request, env: Env): Promise<JWTPayload | null> {
  const auth = req.headers.get('authorization') ?? '';
  const [scheme, token] = auth.split(/\s+/);
  if (scheme?.toLowerCase() !== 'bearer' || !token) {
    const hdrNames = Array.from(req.headers.keys()).sort().join(',');
    console.error(`verifyClerk: no Bearer | auth.len=${auth.length} | headers=[${hdrNames}]`);
    return null;
  }
  try {
    const { payload } = await jwtVerify(token, getJwks(env), {
      issuer: getAcceptedIssuers(env),
    });
    return payload;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // Decode header + claims unverified for diagnostics (NOT for trust).
    let preview = '';
    try {
      const parts = token.split('.');
      if (parts.length === 3) {
        const claims = JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));
        preview = `iss=${claims.iss} aud=${claims.aud} azp=${claims.azp} exp=${claims.exp}`;
      }
    } catch {}
    console.error(`verifyClerk failed: ${msg} | ${preview} | accepted=${getAcceptedIssuers(env).join(',')}`);
    return null;
  }
}

function safeUploadName(name: string): string {
  return name.replace(/[^\w.\-()]/g, '_') || 'upload';
}

function r2ObjectKey(userId: string, name: string): string {
  return `${userId}/${crypto.randomUUID()}/${safeUploadName(name)}`;
}

function workerUploadUrl(origin: string, key: string): string {
  const uploadUrl = new URL('/uploads/blob', origin);
  uploadUrl.searchParams.set('key', key);
  return uploadUrl.toString();
}

async function receiveUpload(req: Request, env: Env, userId: string, url: URL): Promise<Response> {
  const key = url.searchParams.get('key') ?? '';
  if (!key) return new Response('missing upload key', { status: 400 });
  if (!key.startsWith(`${userId}/`)) return new Response('key does not belong to user', { status: 403 });
  if (key.includes('..')) return new Response('invalid upload key', { status: 400 });

  const contentLength = req.headers.get('content-length');
  if (contentLength && Number(contentLength) > MAX_UPLOAD_BYTES) {
    return new Response('file too large', { status: 413 });
  }
  if (!req.body) return new Response('missing upload body', { status: 400 });

  try {
    await env.UPLOADS.put(key, req.body, {
      httpMetadata: {
        contentType: req.headers.get('content-type') ?? 'application/octet-stream',
      },
    });
  } catch (e) {
    console.error(`R2 upload failed key=${key}: ${(e as Error).message}`);
    return new Response('upload storage failed', { status: 502 });
  }
  return new Response(null, { status: 204 });
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const origin = req.headers.get('origin');

    if (matchesClerkProxyPath(url.pathname)) {
      return proxyToClerkFrontendApi(req, env);
    }

    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(env, origin) });
    }

    if (url.pathname === '/health') {
      return withCors(Response.json({ status: 'ok', layer: 'edge' }), env, origin);
    }

    // Keep-warm endpoint — same path the cron handler hits, exposed so we
    // can curl it manually. Anyone can call it; the response is harmless.
    if (url.pathname === '/internal/warm') {
      const probe = await warmContainer(env);
      return withCors(Response.json(probe), env, origin);
    }

    const claims = await verifyClerk(req, env);
    if (!claims) {
      return withCors(new Response('unauthorized', { status: 401 }), env, origin);
    }
    const userId = String(claims.sub);

    // Edge-only fast path: authenticated same-origin upload into R2. This
    // deliberately avoids browser → R2 direct PUTs so production uploads do
    // not depend on bucket CORS or S3 presign header matching.
    if (url.pathname === '/uploads/blob' && req.method === 'PUT') {
      return withCors(await receiveUpload(req, env, userId, url), env, origin);
    }

    // Edge-only fast path: register a file row and return a Worker upload URL.
    if (url.pathname === '/uploads/presign' && req.method === 'POST') {
      // Clone before .json() — the original ``req`` is forwarded to the
      // container further down and the underlying body stream can only be
      // consumed once (otherwise: "Cannot reconstruct a Request with a used
      // body").
      const body = await req.clone().json<{ name: string; mime: string; doc_type?: string; week?: string | null }>();
      try {
        const blob_key = r2ObjectKey(userId, body.name);
        const upload_url = workerUploadUrl(url.origin, blob_key);
        // Forward to the container so it records the File row in Postgres;
        // container returns the file_id we then echo with the Worker upload URL.
        const containerResp = await proxyToBackend(env, req, userId, { presigned: { upload_url, blob_key } });
        if (!containerResp.ok) {
          return withCors(
            new Response(await containerResp.text().catch(() => ''), {
              status: containerResp.status,
              statusText: containerResp.statusText,
            }),
            env,
            origin,
          );
        }
        const containerBody = await containerResp.json<{ file_id: string }>();
        return withCors(Response.json({
          file_id: containerBody.file_id,
          upload_url,
          blob_key,
          method: 'PUT',
        }), env, origin);
      } catch (e) {
        return withCors(new Response(`presign failed: ${(e as Error).message}`, { status: 500 }), env, origin);
      }
    }

    const backendResp = await proxyToBackend(env, req, userId);
    return withCors(backendResp, env, origin);
  },

  // CF cron triggers call ``scheduled`` (not ``fetch``). The wrangler.toml
  // ``[triggers] crons`` block is the schedule; this handler is the work.
  async scheduled(_event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(
      (async () => {
        try {
          const probe = await warmContainer(env);
          console.log(`cron warm: ${JSON.stringify(probe)}`);
        } catch (e) {
          console.error(`cron warm failed: ${(e as Error).message}`);
        }
      })(),
    );
  },
};

function getClientIp(req: Request): string {
  return (
    req.headers.get('CF-Connecting-IP') ??
    req.headers.get('X-Real-IP') ??
    req.headers.get('X-Forwarded-For')?.split(',')[0]?.trim() ??
    ''
  );
}

async function proxyToClerkFrontendApi(req: Request, env: Env): Promise<Response> {
  if (!env.CLERK_SECRET_KEY) {
    return Response.json({ error: 'missing Clerk proxy secret' }, { status: 500 });
  }

  const requestUrl = new URL(req.url);
  const fapiBase = env.CLERK_FAPI ?? 'https://frontend-api.clerk.dev';
  const proxyUrl = (env.CLERK_PROXY_URL ?? `${requestUrl.origin}${CLERK_PROXY_PATH}`).replace(/\/$/, '');
  const targetUrl = req.url.replace(proxyUrl, fapiBase);
  const proxyReq = new Request(req, {
    redirect: 'manual',
  });
  proxyReq.headers.set('Clerk-Proxy-Url', proxyUrl);
  proxyReq.headers.set('Clerk-Secret-Key', env.CLERK_SECRET_KEY);
  proxyReq.headers.set('X-Forwarded-For', getClientIp(req));

  return fetch(targetUrl, proxyReq);
}

async function warmContainer(env: Env): Promise<unknown> {
  // Forward an unauthenticated GET to the container's /internal/warm route.
  // The container only ever sees requests via the BACKEND DO binding, so
  // there's no public path to this endpoint — safe to skip JWT here.
  const id = env.BACKEND.idFromName('global');
  const stub = env.BACKEND.get(id);
  const req = new Request('https://internal/internal/warm', { method: 'GET' });
  const resp = await stub.fetch(req);
  let body: unknown = null;
  try {
    body = await resp.json();
  } catch {
    body = { status: resp.status };
  }
  return body;
}

async function proxyToBackend(
  env: Env,
  req: Request,
  userId: string,
  extra?: { presigned?: { upload_url: string; blob_key: string } },
): Promise<Response> {
  const id = env.BACKEND.idFromName('global');
  const stub = env.BACKEND.get(id);
  const forwarded = new Request(req, {
    headers: new Headers(req.headers),
  });
  forwarded.headers.set('X-User-Id', userId);
  if (extra?.presigned) {
    forwarded.headers.set('X-Presigned-Blob-Key', extra.presigned.blob_key);
  }
  return stub.fetch(forwarded);
}

// Durable Object backed by a Cloudflare Container running FastAPI on port 8000.
// Worker secrets are surfaced to the container as env vars via ``envVars``.
export class LawgptBackend extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = '10m';

  override envVars = {
    DATABASE_URL: this.env.DATABASE_URL ?? '',
    OPENAI_API_KEY: this.env.OPENAI_API_KEY ?? '',
    GOOGLE_API_KEY: this.env.GOOGLE_API_KEY ?? '',
    ZEMBED_API_KEY: this.env.ZEMBED_API_KEY ?? '',
    COHERE_API_KEY: this.env.COHERE_API_KEY ?? '',
    CLERK_JWKS_URL: this.env.CLERK_JWKS_URL ?? '',
    CLERK_ISSUER: this.env.CLERK_ISSUER ?? '',
    CLERK_SECRET_KEY: this.env.CLERK_SECRET_KEY ?? '',
    R2_ACCESS_KEY_ID: this.env.R2_ACCESS_KEY_ID ?? '',
    R2_SECRET_ACCESS_KEY: this.env.R2_SECRET_ACCESS_KEY ?? '',
    R2_ACCOUNT_ID: this.env.R2_ACCOUNT_ID ?? '',
    R2_BUCKET: this.env.R2_BUCKET ?? '',
    VECTOR_BACKEND: 'pgvector',
    BLOB_BACKEND: 'r2',
  };
}
