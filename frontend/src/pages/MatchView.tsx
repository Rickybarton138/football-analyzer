import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import { Play, Brain, MessageSquare, ClipboardList, Loader2, Send } from 'lucide-react';

export default function MatchView() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<{ q: string; a: string }[]>([]);

  const { data: match, isLoading } = useQuery({
    queryKey: ['match', id],
    queryFn: () => api.getMatch(id!),
    refetchInterval: (query) => {
      const m = query.state.data as any;
      return m?.status === 'ready' || m?.status === 'failed' ? false : 3000;
    },
  });

  const { data: analyses } = useQuery({
    queryKey: ['analyses', id],
    queryFn: () => api.getMatchAnalyses(id!),
    enabled: (match as any)?.status === 'ready',
  });

  const runAnalysis = useMutation({
    mutationFn: (type: string) => api.runAnalysis({ match_id: id!, analysis_type: type }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['analyses', id] }),
  });

  const askMutation = useMutation({
    mutationFn: (q: string) => api.askQuestion(id!, q),
    onSuccess: (data: any, q) => {
      setChatHistory((prev) => [...prev, { q, a: data.answer }]);
      setQuestion('');
    },
  });

  const sessionPlan = useMutation({
    mutationFn: () => api.generateSessionPlan(id!),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  const m = match as any;
  const latestAnalysis = (analyses as any[])?.[0];

  return (
    <div className="space-y-8">
      {/* Video Player */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
        <div className="aspect-video bg-black">
          {m?.mux_playback_id ? (
            <iframe
              src={`https://stream.mux.com/${m.mux_playback_id}.m3u8`}
              className="w-full h-full"
              allow="autoplay; fullscreen"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-zinc-600">
              {m?.status !== 'ready' ? (
                <div className="text-center">
                  <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                  <p className="capitalize">{m?.status}...</p>
                </div>
              ) : (
                'Video unavailable'
              )}
            </div>
          )}
        </div>
        <div className="p-4">
          <h1 className="text-2xl font-bold">{m?.title}</h1>
          <div className="flex items-center gap-4 mt-1 text-sm text-zinc-400">
            {m?.opponent && <span>vs {m.opponent}</span>}
            {m?.formation && <span>{m.formation}</span>}
            {m?.duration_seconds && <span>{Math.round(m.duration_seconds / 60)} min</span>}
          </div>
        </div>
      </div>

      {/* Analysis Actions */}
      {m?.status === 'ready' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <button
            onClick={() => runAnalysis.mutate('full')}
            disabled={runAnalysis.isPending}
            className="flex items-center justify-center gap-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 py-3 rounded-lg font-medium transition-colors"
          >
            <Brain className="w-4 h-4" />
            Full Analysis
          </button>
          <button
            onClick={() => runAnalysis.mutate('highlights')}
            disabled={runAnalysis.isPending}
            className="flex items-center justify-center gap-2 bg-amber-500/10 border border-amber-500/20 text-amber-400 hover:bg-amber-500/20 py-3 rounded-lg font-medium transition-colors"
          >
            <Play className="w-4 h-4" />
            Highlights
          </button>
          <button
            onClick={() => runAnalysis.mutate('tactical')}
            disabled={runAnalysis.isPending}
            className="flex items-center justify-center gap-2 bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:bg-blue-500/20 py-3 rounded-lg font-medium transition-colors"
          >
            <Brain className="w-4 h-4" />
            Tactical
          </button>
          <button
            onClick={() => sessionPlan.mutate()}
            disabled={sessionPlan.isPending || !latestAnalysis}
            className="flex items-center justify-center gap-2 bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20 py-3 rounded-lg font-medium transition-colors disabled:opacity-40"
          >
            <ClipboardList className="w-4 h-4" />
            Session Plan
          </button>
        </div>
      )}

      {/* Analysis Results */}
      {latestAnalysis && (
        <div className="space-y-6">
          {latestAnalysis.coaching_advice && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-emerald-400 mb-3">Coaching Insights</h2>
              <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
                {latestAnalysis.coaching_advice}
              </div>
            </div>
          )}

          {latestAnalysis.tactical_raw && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-blue-400 mb-3">Tactical Analysis</h2>
              <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
                {latestAnalysis.tactical_raw}
              </div>
            </div>
          )}

          {latestAnalysis.highlights_raw && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-amber-400 mb-3">Key Moments</h2>
              <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
                {latestAnalysis.highlights_raw}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Session Plan */}
      {sessionPlan.data && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-purple-400 mb-3">Training Session Plan</h2>
          <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
            {(sessionPlan.data as any).session_plan}
          </div>
        </div>
      )}

      {/* Ask a Question */}
      {m?.status === 'ready' && latestAnalysis && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-zinc-200 mb-4 flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Ask About This Match
          </h2>

          {chatHistory.map((chat, i) => (
            <div key={i} className="mb-4">
              <p className="text-sm text-emerald-400 font-medium">{chat.q}</p>
              <p className="text-sm text-zinc-300 mt-1 whitespace-pre-wrap">{chat.a}</p>
            </div>
          ))}

          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && question && askMutation.mutate(question)}
              placeholder="e.g. Why did our shape collapse after 30 minutes?"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
            />
            <button
              onClick={() => question && askMutation.mutate(question)}
              disabled={!question || askMutation.isPending}
              className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 text-white px-4 rounded-lg transition-colors"
            >
              {askMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
