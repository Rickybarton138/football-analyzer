import { Outlet, Link, useLocation } from 'react-router-dom';
import { Home, Upload, Search, Users, Clapperboard } from 'lucide-react';

const nav = [
  { to: '/', label: 'Dashboard', icon: Home },
  { to: '/upload', label: 'Upload Match', icon: Upload },
  { to: '/search', label: 'Search', icon: Search },
  { to: '/squad', label: 'Squad', icon: Users },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 text-emerald-400 font-bold text-xl">
            <Clapperboard className="w-6 h-6" />
            Manager Mentor
          </Link>
          <nav className="flex items-center gap-1">
            {nav.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  location.pathname === to
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
