import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';
import { Search as SearchIcon, Loader2, Play } from 'lucide-react';

export default function Search() {
  const [query, setQuery] = useState('');

  const search = useMutation({
    mutationFn: (q: string) => api.searchFootage(q),
  });

  const results = (search.data as any)?.results || [];

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Search Match Footage</h1>
      <p className="text-zinc-400 mb-8">Ask anything about your matches in plain English</p>

      {/* Search bar */}
      <div className="flex gap-2 mb-8">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && query && search.mutate(query)}
            placeholder="e.g. counterattacks from the left side, defensive errors, pressing triggers..."
            className="w-full bg-zinc-900 border border-zinc-700 rounded-xl pl-12 pr-4 py-3.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none text-lg"
          />
        </div>
        <button
          onClick={() => query && search.mutate(query)}
          disabled={!query || search.isPending}
          className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 text-white px-6 rounded-xl font-medium transition-colors"
        >
          {search.isPending ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Search'}
        </button>
      </div>

      {/* Suggestions */}
      {!search.data && !search.isPending && (
        <div className="grid grid-cols-2 gap-3">
          {[
            'Show me all counterattacks',
            'Defensive transitions in the second half',
            'Build-up play from the back',
            'Pressing triggers and high press moments',
            'Set piece delivery and runs',
            'Moments where we lost possession in midfield',
          ].map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => { setQuery(suggestion); search.mutate(suggestion); }}
              className="text-left bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-sm text-zinc-400 hover:text-zinc-200 hover:border-zinc-700 transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <p className="text-sm text-zinc-500">{results.length} moments found</p>
          {results.map((result: any, i: number) => (
            <div
              key={i}
              className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden flex"
            >
              {result.thumbnail_url ? (
                <img src={result.thumbnail_url} alt="" className="w-48 h-28 object-cover flex-shrink-0" />
              ) : (
                <div className="w-48 h-28 bg-zinc-800 flex-shrink-0" />
              )}
              <div className="p-4 flex-1">
                <div className="flex items-center justify-between">
                  <Link to={`/match/${result.match_id}`} className="font-medium text-zinc-100 hover:text-emerald-400">
                    {result.match_title}
                  </Link>
                  <span className="text-xs text-zinc-500">
                    {Math.round(result.confidence * 100)}% match
                  </span>
                </div>
                <p className="text-sm text-zinc-400 mt-1">
                  {formatTime(result.start_time)} - {formatTime(result.end_time)}
                </p>
                {result.clip_url && (
                  <a
                    href={result.clip_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-2 text-xs text-emerald-400 hover:text-emerald-300"
                  >
                    <Play className="w-3 h-3" />
                    Watch clip
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {search.data && results.length === 0 && (
        <p className="text-center text-zinc-500 py-12">No moments found. Try a different search.</p>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
