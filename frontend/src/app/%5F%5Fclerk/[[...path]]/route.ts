import type { NextRequest } from 'next/server';

export const runtime = 'edge';

const CLERK_FRONTEND_API = 'https://frontend-api.clerk.dev';
const DEFAULT_PROXY_URL = 'https://ashgpt.xyz/__clerk/';

function proxyUrl(): string {
  return (process.env.NEXT_PUBLIC_CLERK_PROXY_URL || DEFAULT_PROXY_URL).replace(/\/$/, '');
}

async function proxyClerkFrontendApi(req: NextRequest): Promise<Response> {
  const secretKey = process.env.CLERK_SECRET_KEY;
  if (!secretKey) {
    return new Response('Clerk proxy missing CLERK_SECRET_KEY', { status: 500 });
  }

  const incoming = new URL(req.url);
  const upstreamPath = incoming.pathname.replace(/^\/__clerk\/?/, '/');
  const upstream = new URL(upstreamPath + incoming.search, CLERK_FRONTEND_API);

  const headers = new Headers(req.headers);
  headers.delete('host');
  headers.set('Clerk-Proxy-Url', proxyUrl());
  headers.set('Clerk-Secret-Key', secretKey);
  headers.set(
    'X-Forwarded-For',
    req.headers.get('CF-Connecting-IP') || req.headers.get('X-Forwarded-For') || '',
  );

  return fetch(upstream, {
    method: req.method,
    headers,
    body: req.method === 'GET' || req.method === 'HEAD' ? undefined : req.body,
    redirect: 'manual',
  });
}

export const GET = proxyClerkFrontendApi;
export const POST = proxyClerkFrontendApi;
export const PUT = proxyClerkFrontendApi;
export const PATCH = proxyClerkFrontendApi;
export const DELETE = proxyClerkFrontendApi;
export const OPTIONS = proxyClerkFrontendApi;
