# ashGPT optimization handoff

Use this file to start the next session focused purely on speed, stability, caching, and UX determinism.

## Current production state

- Production URL: `https://ashgpt.xyz`.
- Edge API: `https://api.ashgpt.xyz`.
- Clerk Frontend API is proxied through `https://ashgpt.xyz/__clerk`.
- Production auth boot has been verified in Chrome: the landing page leaves `Loading...`, Clerk client/environment requests return OK, and the sign-in modal opens.
- Phone chat requests reached the Worker as authenticated after the auth fixes. If a chat still shows `401`, inspect the chat stream/backend provider path before changing Clerk again.
- Avoid using `*.ashgpt.pages.dev` preview URLs for auth testing unless Clerk preview origins are configured. A preview tab showed Clerk 400 errors because Clerk could not attribute that origin to the production instance.

## User goal

Make ashGPT feel fast, stable, and deliberate:

- No random refreshes.
- No wrong-page redirects after sign-in, sign-up, or chat send.
- Guided tour appears quickly but does not block the chat shell.
- Chat send feels reliable even on mobile.
- History and session switching are fast.
- Caching is clear, predictable, and easy to reason about.

## Known risk areas

- `frontend/src/lib/useChat.ts`: streaming state, optimistic messages, session creation/navigation, and token refresh.
- `frontend/src/lib/streaming.ts`: SSE error handling, reconnect/retry behavior, and how stream errors surface to the UI.
- `frontend/src/lib/queries.ts`: TanStack Query stale times, invalidations, focus refetches, and cache keys.
- `frontend/src/components/ChatSurface.tsx`: route/session orchestration, busy/idle transitions, and message list hydration.
- `frontend/src/app/(app)/chat/[sessionId]/page.tsx`: historical chat reload and cache hydration.
- Onboarding/guided tour components: load timing, blocking behavior, and first-interaction delay.
- Clerk redirects in `frontend/src/app/layout.tsx` and landing-page sign-in/sign-up buttons.
- Backend `backend/api/routes_chat.py`: stream errors are emitted inside an HTTP 200 SSE response, so a visible `401` can be an upstream provider failure rather than an edge auth failure.

## Suggested testing matrix

Run these on desktop and mobile viewport, ideally with browser automation and production-like local env:

- Fresh unauthenticated visit to `/`.
- Sign in, then verify redirect lands on `/chat` once and only once.
- Sign up with a new account, then verify user is automatically signed in and routed to `/chat`.
- Send first chat in a fresh account.
- Send a chat from an existing session.
- Create a new session while another stream is idle.
- Reload during an idle chat.
- Reload during an active stream.
- Navigate back/forward between `/chat` and `/chat/[sessionId]`.
- Switch sessions rapidly.
- Open chat history after cold load.
- Lose network mid-stream, then recover.
- Expired/stale Clerk token: verify one refresh/retry, then a helpful error if still unauthorized.

## Metrics to capture

- Time to first usable chat shell.
- Time until guided tour is available.
- Time from pressing send to first streamed token/event.
- Time to load session list.
- Time to load message history for a session.
- Number of route changes during sign-in/sign-up/chat send.
- Number of duplicated network requests on focus/reload.
- Bundle size and largest client chunks for chat/onboarding.

## Guardrails

- Preserve production Clerk proxy behavior unless logs prove it is the failing layer.
- Do not commit secrets.
- Do not default to adding Redis or new infrastructure. Profile first; optimize TanStack Query, local optimistic state, backend indexes, and payload shape before adding another service.
- Keep changes small, tested, committed, and pushed.
- Update `AGENTS.md` when a new gotcha or shipped optimization matters for future sessions.

## Prompt for the next session

```text
You are working in /Users/zer0/Documents/projects/ashGPT. Read AGENTS.md first, then docs/optimization_handoff.md.

Focus only on performance, caching, route stability, and user-flow reliability for ashGPT production readiness. The auth recovery work is mostly done: production Clerk calls go through https://ashgpt.xyz/__clerk, production landing leaves Loading, and signed-in phone chat requests reached the Worker as authenticated. Do not restart the auth investigation unless current logs prove auth is failing.

Primary objective: make the app feel fast and deterministic. Eliminate random refreshes, wrong-page redirects, route jumps after chat send, slow guided-tour startup, and sluggish chat history/session loading.

Work plan:
1. Map the current chat/auth/navigation/cache flow in the frontend.
2. Instrument or measure the slow paths: initial shell, guided tour, session list, message history, send-to-first-token, and route transitions.
3. Build browser automation tests for the critical flows: fresh visit, sign-in/sign-up redirect, first chat, existing chat, reload during idle, reload during stream, back/forward navigation, rapid session switch, mobile viewport.
4. Fix the highest-impact bugs first. Prefer deterministic route state, robust optimistic chat state, lean TanStack Query invalidations, and lazy-loaded noncritical onboarding/tour code.
5. Stress test endpoints and UI flows after each meaningful change.
6. Only consider Redis or new infrastructure after profiling proves local/client/query/backend optimization is not enough.
7. Keep AGENTS.md updated with durable gotchas and progress. Commit and push small, tested changes.

Use subagents in parallel where useful:
- One explorer maps chat/session/cache/navigation code.
- One explorer maps guided-tour/onboarding bundle and load timing.
- One worker can add browser tests once the key flows are identified.
- One reviewer can inspect the diff for regressions before commit.

Deliverables:
- Code fixes and tests.
- A short performance/stability report with before/after measurements where possible.
- Updated AGENTS.md with shipped changes and remaining known risks.
```
