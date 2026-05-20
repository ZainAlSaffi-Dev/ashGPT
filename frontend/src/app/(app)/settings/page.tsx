'use client';

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-8">
      <h1 className="font-serif text-2xl text-ink">Settings</h1>
      <p className="mt-2 text-sm text-ink-muted">
        Model preferences and token usage will live here. Wired up after Phase 5.
      </p>
      <ul className="mt-6 space-y-2 text-sm text-ink-muted">
        <li>· Per-user token budget</li>
        <li>· Synthesis model preference (mini vs full)</li>
        <li>· Confidence-gated escalation threshold</li>
        <li>· Hybrid retrieval on/off</li>
      </ul>
    </div>
  );
}
