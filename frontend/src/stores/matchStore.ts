import { create } from 'zustand';
import { api } from '../lib/api';
import { formatDuration } from '../lib/utils';
import type { MatchAnalysis, AvailableMatch } from '../types/match';

interface MatchState {
  // Data
  analysis: MatchAnalysis | null;
  matches: AvailableMatch[];
  currentMatchId: string | null;

  // Loading states
  loading: boolean;
  loadingMatches: boolean;
  error: string | null;

  // Actions
  loadMatches: () => Promise<void>;
  loadAnalysis: (matchId?: string) => Promise<void>;
  clearAnalysis: () => void;
  setError: (error: string | null) => void;
}

export const useMatchStore = create<MatchState>((set) => ({
  analysis: null,
  matches: [],
  currentMatchId: null,
  loading: false,
  loadingMatches: false,
  error: null,

  loadMatches: async () => {
    set({ loadingMatches: true });
    const matches: AvailableMatch[] = [];

    try {
      const data = await api.get<{ matches: AvailableMatch[] }>('/api/matches/list');
      if (data.matches) matches.push(...data.matches);
    } catch {
      // Endpoint may not exist yet
    }

    try {
      const data = await api.get<Record<string, unknown>>('/api/video/full-analysis');
      if (data && ((data as any).frame_analyses || (data as any).frames)) {
        const current: AvailableMatch = {
          id: 'current',
          name: (data as any).video_path?.split(/[/\\]/).pop() || 'Current Analysis',
          date: (data as any).start_time || new Date().toISOString().split('T')[0],
          duration: (data as any).duration_seconds ? formatDuration((data as any).duration_seconds) : 'Unknown',
          status: 'ready',
        };
        if (!matches.find(m => m.id === 'current')) matches.unshift(current);
      }
    } catch {
      // No current analysis
    }

    set({ matches, loadingMatches: false });
  },

  loadAnalysis: async (matchId?: string) => {
    set({ loading: true, error: null });
    try {
      const url = matchId && matchId !== 'current'
        ? `/api/matches/${matchId}/analysis`
        : '/api/video/full-analysis';

      const data = await api.get<Record<string, unknown>>(url);

      let analysis: MatchAnalysis;
      if ((data as any).frame_analyses) {
        analysis = {
          video_path: (data as any).video_path || '',
          duration_seconds: (data as any).duration_seconds || 0,
          total_frames: (data as any).total_frames || (data as any).frame_analyses.length,
          analyzed_frames: (data as any).analyzed_frames || (data as any).frame_analyses.length,
          fps_analyzed: (data as any).fps_analyzed || 3,
          avg_home_players: (data as any).avg_home_players || 0,
          avg_away_players: (data as any).avg_away_players || 0,
          frames: (data as any).frame_analyses.map((f: any) => ({
            frame_number: f.frame_number,
            timestamp: f.timestamp,
            player_count: f.player_count,
            home_players: f.home_players,
            away_players: f.away_players,
            ball_position: f.ball_position,
            detections: f.detections || [],
          })),
          pass_stats: (data as any).pass_stats,
          formation_stats: (data as any).formation_stats,
          xg_data: (data as any).xg_data,
          tactical_summary: (data as any).tactical_summary,
        };
      } else if ((data as any).frames) {
        analysis = data as unknown as MatchAnalysis;
      } else {
        throw new Error('Invalid analysis format');
      }

      set({ analysis, currentMatchId: matchId || 'current', loading: false });
    } catch (err) {
      set({ error: 'Could not load match analysis.', loading: false });
    }
  },

  clearAnalysis: () => set({ analysis: null, currentMatchId: null }),
  setError: (error) => set({ error }),
}));
