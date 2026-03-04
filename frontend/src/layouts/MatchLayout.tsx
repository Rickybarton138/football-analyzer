import { useEffect } from 'react';
import { Outlet, useParams, useNavigate, useLocation, Link } from 'react-router-dom';
import { useMatchStore } from '../stores/matchStore';
import { useUIStore } from '../stores/uiStore';
import { Spinner } from '../components/ui/Spinner';
import { formatDuration, cn } from '../lib/utils';
import {
  LayoutDashboard, Brain, Target, Shield, BarChart3, Users, ChevronLeft
} from 'lucide-react';

const MATCH_TABS = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'coaching', label: 'AI Coach', icon: Brain },
  { id: 'training', label: 'Training', icon: Target },
  { id: 'tactical', label: 'Tactical', icon: Shield },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'players', label: 'Players', icon: Users },
];

export default function MatchLayout() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { analysis, loading, error, loadAnalysis } = useMatchStore();
  const { selectedPeriod, setSelectedPeriod } = useUIStore();

  useEffect(() => {
    if (id && !analysis) {
      loadAnalysis(id);
    }
  }, [id]);

  // Redirect to overview if at bare /match/:id
  useEffect(() => {
    if (location.pathname === `/match/${id}`) {
      navigate(`/match/${id}/overview`, { replace: true });
    }
  }, [location.pathname, id]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Spinner size="lg" label="Loading match analysis..." />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-white text-xl font-semibold mb-2">Match Not Found</h2>
          <p className="text-slate-400 mb-4">{error || 'Could not load match analysis.'}</p>
          <Link to="/" className="text-emerald-400 hover:text-emerald-300">Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  const activeTab = location.pathname.split('/').pop() || 'overview';
  const matchTitle = analysis.video_path?.split(/[/\\]/).pop() || 'Match Analysis';

  return (
    <div className="min-h-screen">
      {/* Match Header */}
      <header className="bg-surface border-b border-slate-700/50">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Link
                to="/"
                className="text-slate-400 hover:text-emerald-400 transition-colors"
              >
                <ChevronLeft className="w-5 h-5" />
              </Link>
              <div>
                <h1 className="text-lg font-semibold text-white">{matchTitle}</h1>
                <p className="text-slate-400 text-xs">{formatDuration(analysis.duration_seconds)} duration</p>
              </div>
            </div>

            {/* Period Selector */}
            <div className="flex items-center gap-1 bg-slate-800/50 rounded-lg p-1">
              {(['full', '1st', '2nd'] as const).map(period => (
                <button
                  key={period}
                  onClick={() => setSelectedPeriod(period)}
                  className={cn(
                    'px-3 py-1.5 text-xs font-medium rounded-md transition-all',
                    selectedPeriod === period
                      ? 'bg-emerald-500 text-white'
                      : 'text-slate-400 hover:text-white'
                  )}
                >
                  {period === 'full' ? 'Full Match' : `${period} Half`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Sub-nav tabs */}
        <div className="px-6">
          <nav className="flex gap-1">
            {MATCH_TABS.map(tab => (
              <Link
                key={tab.id}
                to={`/match/${id}/${tab.id}`}
                className={cn(
                  'px-4 py-2.5 text-sm font-medium transition-all border-b-2 flex items-center gap-2',
                  activeTab === tab.id
                    ? 'text-emerald-400 border-emerald-400'
                    : 'text-slate-400 border-transparent hover:text-slate-200'
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      {/* Page content */}
      <main className="p-6">
        <Outlet />
      </main>
    </div>
  );
}
