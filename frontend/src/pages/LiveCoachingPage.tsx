import LiveCoaching from '../components/LiveCoaching';

export default function LiveCoachingPage() {
  return (
    <div className="min-h-screen">
      <header className="bg-surface border-b border-slate-700/50 px-6 py-4">
        <h1 className="text-xl font-semibold text-white">Live Coaching</h1>
        <p className="text-slate-400 text-sm">Real-time match analysis and coaching</p>
      </header>
      <main className="p-6">
        <LiveCoaching />
      </main>
    </div>
  );
}
