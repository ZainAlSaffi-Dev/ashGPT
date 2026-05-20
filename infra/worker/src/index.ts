/**
 * LawGPT edge Worker.
 *
 * Responsibilities (kept thin):
 *   1. Verify the Clerk JWT on every authenticated request.
 *   2. Issue R2 presigned PUT URLs for direct browser → R2 uploads.
 *   3. Proxy everything else to the FastAPI container.
 *
 * The container holds the LangGraph pipeline and DB writes. Keeping the
 * Worker thin means cold-start is low and the heavy Python runtime only
 * runs when needed.
 */

import { AwsClient } from 'aws4fetch';
import { Container } from '@cloudflare/containers';
import { createRemoteJWKSet, jwtVerify, type JWTPayload } from 'jose';

export interface Env {
  BACKEND: DurableObjectNamespace;
  UPLOADS: R2Bucket;
  VECTORS: VectorizeIndex; // eslint-disable-line @typescript-eslint/no-explicit-any
  DB: D1Database;
  CLERK_ISSUER: string;
  CLERK_SECRET_KEY: string;
  CORS_ORIGINS?: string;
  // Backend container secrets — forwarded into the container via envVars.
  DATABASE_URL?: string;
  OPENAI_API_KEY?: string;
  GOOGLE_API_KEY?: string;
  ZEMBED_API_KEY?: string;
  COHERE_API_KEY?: string;
  // R2 S3-API credentials for presigning (set via wrangler secret put):
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
  R2_ACCOUNT_ID?: string;
  R2_BUCKET?: string;
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
  const url = new URL('.well-known/jwks.json', env.CLERK_ISSUER.replace(/\/$/, '') + '/');
  _jwks = createRemoteJWKSet(url);
  return _jwks;
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
      issuer: env.CLERK_ISSUER,
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
    console.error(`verifyClerk failed: ${msg} | ${preview} | CLERK_ISSUER=${env.CLERK_ISSUER}`);
    return null;
  }
}

async function r2PresignPut(env: Env, body: { name: string; mime: string; user_id: string }) {
  if (!env.R2_ACCESS_KEY_ID || !env.R2_SECRET_ACCESS_KEY || !env.R2_ACCOUNT_ID) {
    throw new Error('R2 presign requires R2_* secrets');
  }
  const aws = new AwsClient({
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    region: 'auto',
    service: 's3',
  });
  const bucket = env.R2_BUCKET ?? 'lawgpt-uploads';
  const key = `${body.user_id}/${crypto.randomUUID()}/${body.name.replace(/[^\w.\-()]/g, '_')}`;
  // 15-minute expiry encoded as a query param so it is included in the
  // signature; aws4fetch has no `expiresIn` option.
  const endpoint = `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/${bucket}/${key}?X-Amz-Expires=900`;
  const signed = await aws.sign(new Request(endpoint, { method: 'PUT', headers: { 'content-type': body.mime } }), {
    aws: { signQuery: true },
  });
  return { upload_url: signed.url, blob_key: key };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const origin = req.headers.get('origin');

    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(env, origin) });
    }

    if (url.pathname === '/health') {
      return withCors(Response.json({ status: 'ok', layer: 'edge' }), env, origin);
    }

    const claims = await verifyClerk(req, env);
    if (!claims) {
      return withCors(new Response('unauthorized', { status: 401 }), env, origin);
    }
    const userId = String(claims.sub);

    // Edge-only fast path: presigned PUT URL for direct R2 upload.
    if (url.pathname === '/uploads/presign' && req.method === 'POST') {
      // Clone before .json() — the original ``req`` is forwarded to the
      // container further down and the underlying body stream can only be
      // consumed once (otherwise: "Cannot reconstruct a Request with a used
      // body").
      const body = await req.clone().json<{ name: string; mime: string; doc_type?: string; week?: string | null }>();
      try {
        const { upload_url, blob_key } = await r2PresignPut(env, {
          name: body.name,
          mime: body.mime,
          user_id: userId,
        });
        // Forward to the container so it records the File row in D1; container
        // returns the file_id we then echo with the presigned URL.
        const containerResp = await proxyToBackend(env, req, userId, { presigned: { upload_url, blob_key } });
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
};

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
