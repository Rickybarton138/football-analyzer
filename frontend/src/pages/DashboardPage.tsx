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
      <header className="bg-surface border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-text-primary">Dashboard</h1>
            <p className="text-text-muted text-sm">AI Coaching Intelligence</p>
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
          <h2 className="text-3xl font-bold text-text-primary mb-3 font-display">
            Welcome to <span className="text-pitch">Coach</span><span className="text-pitch-deep">Mentor</span>
          </h2>
          <p className="text-text-secondary max-w-2xl mx-auto">
            Upload match footage and get AI-powered coaching insights including tactical analysis,
            training recommendations, and personalised session plans.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <Link
            to="/upload"
            className="bg-pitch-light border border-pitch/20 rounded-card p-6 text-left hover:border-pitch/40 hover:shadow-card-hover transition-all group"
          >
            <div className="w-12 h-12 bg-pitch/10 rounded-lg flex items-center justify-center mb-4 group-hover:bg-pitch/20 transition-colors">
              <Upload className="w-6 h-6 text-pitch" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-1">Upload New Match</h3>
            <p className="text-text-secondary text-sm">Start analysing a new match video with AI</p>
          </Link>

          <div className="bg-surface border border-border rounded-card p-6 shadow-card">
            <div className="w-12 h-12 bg-sky-light rounded-lg flex items-center justify-center mb-4">
              <BarChart3 className="w-6 h-6 text-sky" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-1">Match Analysis</h3>
            <p className="text-text-secondary text-sm">View possession, passes, shots, and more</p>
          </div>

          <div className="bg-surface border border-border rounded-card p-6 shadow-card">
            <div className="w-12 h-12 bg-gold-light rounded-lg flex items-center justify-center mb-4">
              <Brain className="w-6 h-6 text-gold-dark" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-1">AI Coach</h3>
            <p className="text-text-secondary text-sm">Get tactical insights and recommendations</p>
          </div>
        </div>

        {/* Recent Matches */}
        <div>
          <h3 className="text-xl font-semibold text-text-primary mb-4">Recent Matches</h3>

          {loadingMatches ? (
            <Spinner label="Loading matches..." className="py-12" />
          ) : matches.length > 0 ? (
            <div className="grid gap-3">
              {matches.map(match => (
                <Link
                  key={match.id}
                  to={`/match/${match.id}/overview`}
                  className="bg-surface border border-border rounded-card p-4 text-left hover:border-pitch/30 hover:shadow-card-hover transition-all flex items-center gap-4"
                >
                  <div className="w-12 h-12 bg-surface-alt rounded-lg flex items-center justify-center">
                    <Video className="w-6 h-6 text-text-muted" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-text-primary font-medium truncate">{match.name}</h4>
                    <p className="text-text-muted text-sm">{match.date} &middot; {match.duration}</p>
                  </div>
                  <Badge variant={match.status === 'ready' ? 'success' : match.status === 'processing' ? 'medium' : 'critical'}>
                    {match.status === 'ready' ? 'Ready' : match.status === 'processing' ? 'Processing' : 'Failed'}
                  </Badge>
                </Link>
              ))}
            </div>
          ) : (
            <EmptyState
              icon={<Video className="w-8 h-8 text-text-muted" />}
              title="No matches yet"
              description="Upload your first match video to get started with AI analysis"
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
