import LiveCoaching from '../components/LiveCoaching';

export default function LiveCoachingPage() {
  return (
    <div className="min-h-screen">
      <header className="bg-surface border-b border-border px-6 py-4">
        <h1 className="text-xl font-semibold text-text-primary">Live Coaching</h1>
        <p className="text-text-muted text-sm">Real-time match analysis and coaching</p>
      </header>
      <main className="p-6">
        <LiveCoaching />
      </main>
    </div>
  );
}
