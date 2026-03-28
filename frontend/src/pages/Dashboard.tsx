import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';
import { Plus, Clock, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const statusIcon: Record<string, any> = {
  ready: <CheckCircle className="w-4 h-4 text-emerald-400" />,
  processing: <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />,
  indexing: <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />,
  analysing: <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />,
  uploading: <Clock className="w-4 h-4 text-zinc-400" />,
  failed: <AlertCircle className="w-4 h-4 text-red-400" />,
};

export default function Dashboard() {
  const { data: matches, isLoading } = useQuery({
    queryKey: ['matches'],
    queryFn: api.listMatches,
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Matches</h1>
          <p className="text-zinc-400 mt-1">Upload match footage and get AI-powered coaching insights</p>
        </div>
        <Link
          to="/upload"
          className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2.5 rounded-lg font-medium transition-colors"
        >
          <Plus className="w-4 h-4" />
          Upload Match
        </Link>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
        </div>
      ) : !matches?.length ? (
        <div className="text-center py-20 border border-dashed border-zinc-700 rounded-xl">
          <h3 className="text-lg font-medium text-zinc-300">No matches yet</h3>
          <p className="text-zinc-500 mt-1">Upload your first match to get started</p>
          <Link
            to="/upload"
            className="inline-flex items-center gap-2 mt-4 bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            <Plus className="w-4 h-4" />
            Upload Match
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {matches.map((match: any) => (
            <Link
              key={match.id}
              to={`/match/${match.id}`}
              className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden hover:border-zinc-700 transition-colors group"
            >
              {match.thumbnail_url ? (
                <img
                  src={match.thumbnail_url}
                  alt={match.title}
                  className="w-full h-40 object-cover group-hover:opacity-90 transition-opacity"
                />
              ) : (
                <div className="w-full h-40 bg-zinc-800 flex items-center justify-center">
                  <span className="text-zinc-600 text-sm">No thumbnail</span>
                </div>
              )}
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-zinc-100 truncate">{match.title}</h3>
                  {statusIcon[match.status] || statusIcon.uploading}
                </div>
                {match.opponent && (
                  <p className="text-sm text-zinc-400 mt-1">vs {match.opponent}</p>
                )}
                <div className="flex items-center gap-3 mt-2 text-xs text-zinc-500">
                  <span className="capitalize">{match.status}</span>
                  {match.duration_seconds && (
                    <span>{Math.round(match.duration_seconds / 60)} min</span>
                  )}
                  {match.match_date && (
                    <span>{new Date(match.match_date).toLocaleDateString()}</span>
                  )}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
