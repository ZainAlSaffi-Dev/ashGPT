import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

const isPublic = createRouteMatcher([
  '/',
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/api/health',
  '/__clerk(.*)',
]);

export default clerkMiddleware(
  async (auth, req) => {
    if (!isPublic(req)) {
      const authState = await auth();
      const isPageNavigation =
        req.headers.get('sec-fetch-dest') === 'document' ||
        req.headers.get('accept')?.includes('text/html');
      if (!authState.userId && isPageNavigation) {
        const signInUrl = new URL('/', req.url);
        signInUrl.searchParams.set('redirect_url', req.nextUrl.href);
        return NextResponse.redirect(signInUrl);
      }
      await auth.protect();
    }
  },
  {
    // Protected deep links should return to our same-origin landing page,
    // whose Clerk modal buttons own the sign-in/sign-up flow. Without this,
    // Clerk falls back to the account portal host (accounts.ashgpt.xyz), which
    // is not a reliable production route for ashGPT.
    signInUrl: '/',
    signUpUrl: '/',
    frontendApiProxy: {
      enabled: true,
    },
  },
);

export const config = {
  matcher: [
    '/((?!_next|.*\\..*).*)',
    '/(api|trpc)(.*)',
    '/__clerk/(.*)',
  ],
};
