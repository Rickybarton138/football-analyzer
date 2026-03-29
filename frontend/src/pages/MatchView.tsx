import { useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import { Play, Brain, ClipboardList, Loader2, Users, AlertTriangle, Bot } from 'lucide-react';
import MuxPlayer from '@mux/mux-player-react';
import CoachChat from '../components/CoachChat';

export default function MatchView() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const [chatOpen, setChatOpen] = useState(false);
  const playerRef = useRef<any>(null);

  const { data: match, isLoading } = useQuery({
    queryKey: ['match', id],
    queryFn: () => api.getMatchStatus(id!),
    refetchInterval: (query) => {
      const m = query.state.data as any;
      return m?.status === 'ready' || m?.status === 'failed' ? false : 3000;
    },
  });

  const { data: analyses } = useQuery({
    queryKey: ['analyses', id],
    queryFn: () => api.getMatchAnalyses(id!),
    enabled: (match as any)?.status === 'ready',
    refetchInterval: (query) => {
      const list = query.state.data as any[];
      if (!list?.length) return false;
      const hasProcessing = list.some((a: any) => a.status === 'processing');
      return hasProcessing ? 5000 : false;
    },
  });

  const runAnalysis = useMutation({
    mutationFn: (type: string) => api.runAnalysis({ match_id: id!, analysis_type: type }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['analyses', id] }),
  });

  const sessionPlan = useMutation({
    mutationFn: () => api.generateSessionPlan(id!),
  });

  const handleSeekTo = (time: number) => {
    const el = playerRef.current;
    if (el) {
      // MuxPlayer exposes the media element
      const media = el.media?.nativeEl || el;
      if (media && typeof media.currentTime !== 'undefined') {
        media.currentTime = time;
        media.play?.();
      }
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  const m = match as any;
  const analysisList = (analyses as any[]) || [];
  const latestAnalysis = analysisList.find((a: any) => a.status === 'complete');
  const processingAnalysis = analysisList.find((a: any) => a.status === 'processing');

  return (
    <>
      <div className={`space-y-8 transition-all duration-300 ${chatOpen ? 'mr-[420px]' : ''}`}>
        {/* Video Player */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          <div className="aspect-video bg-black">
            {/* TODO: Generate chapters WebVTT from TwelveLabs timestamps */}
            {m?.mux_playback_id ? (
              <MuxPlayer
                ref={playerRef}
                playbackId={m.mux_playback_id}
                streamType="on-demand"
                accentColor="#10b981"
                className="w-full h-full"
                storyboard-src
                playbackRates={[0.25, 0.5, 1, 1.5, 2]}
                pip
                forwardSeekOffset={10}
                backwardSeekOffset={10}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-zinc-600">
                {m?.status === 'failed' ? (
                  <div className="text-center">
                    <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
                    <p className="text-red-400">Processing failed</p>
                    {m?.error_message && (
                      <p className="text-xs text-zinc-500 mt-1 max-w-md">{m.error_message}</p>
                    )}
                  </div>
                ) : m?.status !== 'ready' ? (
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
          <div className="p-4 flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold">{m?.title}</h1>
              <div className="flex items-center gap-4 mt-1 text-sm text-zinc-400">
                {m?.opponent && <span>vs {m.opponent}</span>}
                {m?.formation && <span>{m.formation}</span>}
                {m?.duration_seconds && <span>{Math.round(m.duration_seconds / 60)} min</span>}
              </div>
            </div>
            {m?.status === 'ready' && (
              <button
                onClick={() => setChatOpen(!chatOpen)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                  chatOpen
                    ? 'bg-emerald-500 text-white'
                    : 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20'
                }`}
              >
                <Bot className="w-4 h-4" />
                {chatOpen ? 'Close Coach' : 'Talk to Coach'}
              </button>
            )}
          </div>
        </div>

        {/* Analysis Actions */}
        {m?.status === 'ready' && (
          <div>
            {processingAnalysis && (
              <div className="flex items-center gap-2 text-sm text-amber-400 mb-3">
                <Loader2 className="w-4 h-4 animate-spin" />
                Analysis running...
              </div>
            )}
            {runAnalysis.isError && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-3 py-2 rounded-lg mb-3 text-sm">
                {(runAnalysis.error as Error).message}
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
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
                onClick={() => runAnalysis.mutate('player_spotlight')}
                disabled={runAnalysis.isPending}
                className="flex items-center justify-center gap-2 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20 py-3 rounded-lg font-medium transition-colors"
              >
                <Users className="w-4 h-4" />
                Players
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

            {latestAnalysis.player_analysis_raw && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
                <h2 className="text-lg font-semibold text-cyan-400 mb-3">Player Analysis</h2>
                <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
                  {latestAnalysis.player_analysis_raw}
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
      </div>

      {/* Floating Coach Chat Panel */}
      <CoachChat
        matchId={id!}
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        onSeekTo={handleSeekTo}
      />
    </>
  );
}
