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
import { createRemoteJWKSet, jwtVerify, type JWTPayload } from 'jose';

export interface Env {
  BACKEND: DurableObjectNamespace;
  UPLOADS: R2Bucket;
  VECTORS: VectorizeIndex; // eslint-disable-line @typescript-eslint/no-explicit-any
  DB: D1Database;
  CLERK_ISSUER: string;
  CLERK_SECRET_KEY: string;
  // R2 S3-API credentials for presigning (set via wrangler secret put):
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
  R2_ACCOUNT_ID?: string;
  R2_BUCKET?: string;
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
  if (scheme?.toLowerCase() !== 'bearer' || !token) return null;
  try {
    const { payload } = await jwtVerify(token, getJwks(env), {
      issuer: env.CLERK_ISSUER,
    });
    return payload;
  } catch {
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
  const endpoint = `https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/${bucket}/${key}`;
  const signed = await aws.sign(new Request(endpoint, { method: 'PUT', headers: { 'content-type': body.mime } }), {
    aws: { signQuery: true },
    expiresIn: 900,
  });
  return { upload_url: signed.url, blob_key: key };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);

    if (url.pathname === '/health') {
      return Response.json({ status: 'ok', layer: 'edge' });
    }

    // Bypass auth on a tiny set of public routes (none right now besides health).
    const claims = await verifyClerk(req, env);
    if (!claims) {
      return new Response('unauthorized', { status: 401 });
    }
    const userId = String(claims.sub);

    // Edge-only fast path: presigned PUT URL for direct R2 upload.
    if (url.pathname === '/uploads/presign' && req.method === 'POST') {
      const body = await req.json<{ name: string; mime: string; doc_type?: string; week?: string | null }>();
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
        return Response.json({
          file_id: containerBody.file_id,
          upload_url,
          blob_key,
          method: 'PUT',
        });
      } catch (e) {
        return new Response(`presign failed: ${(e as Error).message}`, { status: 500 });
      }
    }

    // Everything else proxies through to the Container.
    return proxyToBackend(env, req, userId);
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

// Minimal Durable Object wrapping the Container endpoint. The Container Image
// is configured in wrangler.toml; this DO just routes to the container instance.
export class LawgptBackend {
  constructor(private state: DurableObjectState, _env: Env) {}

  async fetch(req: Request): Promise<Response> {
    const container = (this.state as unknown as { container?: { fetch: typeof fetch } }).container;
    if (!container) {
      return new Response('container binding not initialised', { status: 503 });
    }
    return container.fetch(req);
  }
}
