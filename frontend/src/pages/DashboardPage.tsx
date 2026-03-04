import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useMatchStore } from '../stores/matchStore';
import { Upload, BarChart3, Brain, Video } from 'lucide-react';
import { Spinner } from '../components/ui/Spinner';
import { EmptyState } from '../components/ui/EmptyState';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

export default function DashboardPage() {
  const { matches, loadingMatches, loadMatches } = useMatchStore();

  useEffect(() => { loadMatches(); }, []);

  return (
    <div className="min-h-screen">
      {/* Top bar */}
      <header className="bg-surface border-b border-slate-700/50 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-white">Dashboard</h1>
            <p className="text-slate-400 text-sm">AI Coaching Intelligence</p>
          </div>
          <Link to="/upload">
            <Button>
              <Upload className="w-4 h-4" />
              New Match
            </Button>
          </Link>
        </div>
      </header>

      <div className="p-6 max-w-5xl mx-auto">
        {/* Welcome */}
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-white mb-3">
            Welcome to <span className="text-emerald-400">Dugout</span><span className="text-cyan-400">IQ</span>
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Upload match footage and get AI-powered coaching insights including tactical analysis,
            training recommendations, and real-time coaching intelligence.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <Link
            to="/upload"
            className="bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 border border-emerald-500/30 rounded-xl p-6 text-left hover:border-emerald-500/50 transition-all group"
          >
            <div className="w-12 h-12 bg-emerald-500/20 rounded-lg flex items-center justify-center mb-4 group-hover:bg-emerald-500/30 transition-colors">
              <Upload className="w-6 h-6 text-emerald-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1">Upload New Match</h3>
            <p className="text-slate-400 text-sm">Start analyzing a new match video with AI</p>
          </Link>

          <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-6">
            <div className="w-12 h-12 bg-slate-700/50 rounded-lg flex items-center justify-center mb-4">
              <BarChart3 className="w-6 h-6 text-slate-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1">Match Analysis</h3>
            <p className="text-slate-400 text-sm">View possession, passes, shots, and more</p>
          </div>

          <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-6">
            <div className="w-12 h-12 bg-slate-700/50 rounded-lg flex items-center justify-center mb-4">
              <Brain className="w-6 h-6 text-slate-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1">AI Coach</h3>
            <p className="text-slate-400 text-sm">Get tactical insights and recommendations</p>
          </div>
        </div>

        {/* Recent Matches */}
        <div>
          <h3 className="text-xl font-semibold text-white mb-4">Recent Matches</h3>

          {loadingMatches ? (
            <Spinner label="Loading matches..." className="py-12" />
          ) : matches.length > 0 ? (
            <div className="grid gap-3">
              {matches.map(match => (
                <Link
                  key={match.id}
                  to={`/match/${match.id}/overview`}
                  className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 text-left hover:border-emerald-500/30 hover:bg-slate-800/70 transition-all flex items-center gap-4"
                >
                  <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center">
                    <Video className="w-6 h-6 text-slate-300" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-white font-medium truncate">{match.name}</h4>
                    <p className="text-slate-400 text-sm">{match.date} &middot; {match.duration}</p>
                  </div>
                  <Badge variant={match.status === 'ready' ? 'success' : match.status === 'processing' ? 'medium' : 'critical'}>
                    {match.status === 'ready' ? 'Ready' : match.status === 'processing' ? 'Processing' : 'Failed'}
                  </Badge>
                </Link>
              ))}
            </div>
          ) : (
            <EmptyState
              icon={<Video className="w-8 h-8 text-slate-500" />}
              title="No matches yet"
              description="Upload your first match video to get started"
              action={
                <Link to="/upload">
                  <Button variant="secondary" size="sm">Upload Match</Button>
                </Link>
              }
            />
          )}
        </div>
      </div>
    </div>
  );
}
