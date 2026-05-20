# frontend/

Next.js 15 + Tailwind + Clerk + shadcn/ui app. Built during Phase 3 of the productization plan, on the `frontend` branch.

## Initialise (when starting Phase 3)

```bash
cd frontend
rm package.json README.md
pnpm create next-app@latest . --typescript --tailwind --app --src-dir --import-alias '@/*' --use-pnpm
pnpm add @clerk/nextjs @tanstack/react-query mermaid react-dropzone eventsource-parser
pnpm add -D @types/node
```

## Routes (planned)

- `/` — landing → redirect to `/chat` when signed in
- `/chat` — streaming chat, IRAC accordion, Mermaid render, source panel
- `/library` — drag-drop file upload, file list with tagging
- `/exam` — generate + take + grade MCQ / short-answer / past-paper
- `/settings` — model prefs, token usage
