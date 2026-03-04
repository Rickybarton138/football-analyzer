export interface MatchAnalysis {
  video_path: string;
  duration_seconds: number;
  total_frames: number;
  analyzed_frames: number;
  fps_analyzed: number;
  avg_home_players: number;
  avg_away_players: number;
  frames: FrameData[];
  // Phase 1+ enrichments
  pass_stats?: Record<string, unknown>;
  formation_stats?: Record<string, unknown>;
  xg_data?: Record<string, unknown>;
  tactical_summary?: Record<string, unknown>;
}

export interface FrameData {
  frame_number: number;
  timestamp: number;
  player_count: number;
  home_players: number;
  away_players: number;
  ball_position: [number, number] | null;
  ball_pitch_x?: number;
  ball_pitch_y?: number;
  detections: Detection[];
}

export interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
  team: 'home' | 'away' | 'referee' | 'unknown';
  pitch_x?: number;
  pitch_y?: number;
}

export interface MatchMetadata {
  homeTeam: string;
  awayTeam: string;
  isHomeTeam: boolean;
  homeJerseyColor: string;
  awayJerseyColor: string;
  homeFormation: string;
  awayFormation: string;
  matchDate: string;
  competition: string;
  venue: string;
}

export interface AvailableMatch {
  id: string;
  name: string;
  date: string;
  duration: string;
  status: 'ready' | 'processing' | 'failed';
}

export interface MatchStats {
  avgHomePlayers: number;
  avgAwayPlayers: number;
  homeScore: number;
  awayScore: number;
  possession: { home: number; away: number };
  territorialAdvantage: { home: number; away: number };
  pressingActions: { home: number; away: number };
  highIntensityMoments: { home: number; away: number };
  compactness: { home: number; away: number };
  widthUsage: { home: number; away: number };
  possessionByZone: { home: number; away: number }[];
  momentumPeriods: { label: string; homeControl: number; awayControl: number; dominant: 'home' | 'away' }[];
  momentumShifts: { time: string; description: string; favor: 'home' | 'away' }[];
  periodStats: { label: string; homePossession: number; awayPossession: number; pressing: number; dominant: 'home' | 'away' }[];
  insights: { type: 'positive' | 'warning' | 'info'; title: string; description: string }[];
}

export interface SystemStatus {
  gpu: { available: boolean; name: string | null; memory_gb: number | null };
  cpu: { cores: number; threads: number; name: string };
  memory: { total_gb: number; available_gb: number };
  processing_estimates: Record<string, { description: string; cpu_minutes: number; gpu_minutes: number }>;
  recommended_mode: string;
  recommendation: string;
}

export type ProcessingModeType = 'quick_preview' | 'standard' | 'full';
