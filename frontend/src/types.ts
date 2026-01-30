// Football Match Analyzer - TypeScript Types

export type TeamSide = 'home' | 'away' | 'unknown';

export type EventType =
  | 'pass'
  | 'shot'
  | 'tackle'
  | 'interception'
  | 'foul'
  | 'goal'
  | 'corner'
  | 'throw_in'
  | 'free_kick'
  | 'offside';

export type AlertPriority = 'immediate' | 'tactical' | 'strategic';

export type ProcessingMode = 'live' | 'post_match';

export type AnalysisMode = 'full' | 'my_team' | 'opponent' | 'quick_overview';

// Position types
export interface Position {
  x: number;
  y: number;
}

export interface PixelPosition {
  x: number;
  y: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface Velocity {
  vx: number;
  vy: number;
  speed_kmh: number;
}

// Detection types
export interface DetectedPlayer {
  track_id: number;
  bbox: BoundingBox;
  pixel_position: PixelPosition;
  pitch_position?: Position;
  team: TeamSide;
  jersey_color?: number[];
  is_goalkeeper: boolean;
}

export interface DetectedBall {
  bbox: BoundingBox;
  pixel_position: PixelPosition;
  pitch_position?: Position;
  velocity?: Velocity;
  possessed_by?: number;
}

export interface FrameDetection {
  frame_number: number;
  timestamp_ms: number;
  players: DetectedPlayer[];
  ball?: DetectedBall;
  home_players: number;
  away_players: number;
}

// Event types
export interface MatchEvent {
  event_id: string;
  event_type: EventType;
  timestamp_ms: number;
  frame_number: number;
  position: Position;
  player_id?: number;
  team?: TeamSide;
  recipient_id?: number;
  success?: boolean;
  metadata: Record<string, unknown>;
}

// Analytics types
export interface PlayerMetrics {
  track_id: number;
  team: TeamSide;
  distance_covered_m: number;
  sprint_count: number;
  sprint_distance_m: number;
  high_intensity_distance_m: number;
  max_speed_kmh: number;
  avg_speed_kmh: number;
  passes_attempted: number;
  passes_completed: number;
  touches: number;
  tackles: number;
  interceptions: number;
}

export interface TeamMetrics {
  team: TeamSide;
  possession_pct: number;
  total_passes: number;
  pass_completion_pct: number;
  shots: number;
  shots_on_target: number;
  xg: number;
  total_distance_km: number;
  avg_formation?: string;
}

export interface HeatmapData {
  player_id?: number;
  team?: TeamSide;
  grid: number[][];
  grid_size: [number, number];
}

export interface PassNetworkNode {
  id: number;
  x: number;
  y: number;
  passes: number;
}

export interface PassNetworkEdge {
  source: number;
  target: number;
  weight: number;
}

export interface PassNetwork {
  team: TeamSide;
  nodes: PassNetworkNode[];
  edges: PassNetworkEdge[];
}

// Alert types
export interface TacticalAlert {
  alert_id: string;
  priority: AlertPriority;
  timestamp_ms: number;
  message: string;
  details?: string;
  suggested_action?: string;
  related_players: number[];
  position?: Position;
  expires_at_ms?: number;
}

// Match types
export interface MatchInfo {
  match_id: string;
  home_team: string;
  away_team: string;
  date: string;
  venue?: string;
  competition?: string;
}

export interface MatchState {
  match_id: string;
  current_time_ms: number;
  period: number;
  home_score: number;
  away_score: number;
  home_metrics: TeamMetrics;
  away_metrics: TeamMetrics;
  recent_events: MatchEvent[];
  active_alerts: TacticalAlert[];
}

// API types
export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  duration_ms: number;
  fps: number;
  resolution: [number, number];
  status: string;
}

export interface ProcessingStatus {
  video_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress_pct: number;
  current_frame: number;
  total_frames: number;
  estimated_remaining_s?: number;
  error_message?: string;
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'frame' | 'alert' | 'event' | 'status' | 'pong';
  data: unknown;
}

export interface FrameUpdateMessage {
  type: 'frame';
  data: FrameDetection;
}

export interface AlertMessage {
  type: 'alert';
  data: TacticalAlert;
}

// Dashboard state
export interface DashboardState {
  matchId: string | null;
  isLive: boolean;
  currentFrame: FrameDetection | null;
  alerts: TacticalAlert[];
  matchState: MatchState | null;
  videoUrl: string | null;
}

// ============== Focused Analysis Types ==============

export interface PlayerImprovementSuggestion {
  player_id: number;
  category: string;
  observation: string;
  suggestion: string;
  priority: 'high' | 'medium' | 'low';
  evidence_frames: number[];
}

export interface TeamImprovementArea {
  area: string;
  observation: string;
  drill_suggestion: string;
  priority: 'high' | 'medium' | 'low';
}

export interface TeamImprovementReport {
  team_name: string;
  match_id: string;
  analysis_summary: string;
  team_metrics: TeamMetrics;
  formation_analysis: Record<string, unknown>;
  strengths: string[];
  improvement_areas: TeamImprovementArea[];
  player_improvements: PlayerImprovementSuggestion[];
  training_focus: string[];
  recommended_drills: string[];
  passing_network_issues: string[];
  defensive_vulnerabilities: string[];
  attacking_patterns: string[];
}

export interface PlayerScoutReport {
  player_id: number;
  position: string;
  tendencies: string[];
  strengths: string[];
  weaknesses: string[];
  avg_position?: Position;
  touches_per_zone: Record<string, number>;
  duel_win_rate?: number;
  threat_level: number;
  tactical_advice: string;
}

export interface TeamWeakness {
  category: string;
  description: string;
  how_to_exploit: string;
  confidence: number;
  evidence: string[];
}

export interface TeamPattern {
  pattern_type: string;
  description: string;
  frequency: 'always' | 'often' | 'sometimes';
  counter_strategy: string;
}

export interface OpponentScoutReport {
  opponent_name: string;
  match_id: string;
  analysis_summary: string;
  team_metrics: TeamMetrics;
  formation: string;
  attacking_patterns: TeamPattern[];
  defensive_patterns: TeamPattern[];
  build_up_patterns: TeamPattern[];
  weaknesses: TeamWeakness[];
  strengths: string[];
  danger_players: number[];
  key_player_reports: PlayerScoutReport[];
  set_piece_vulnerabilities: string[];
  recommended_formation: string;
  tactical_approach: string;
  tactical_recommendations: string[];
  key_battles: string[];
}

// Analysis mode descriptions for UI
export const ANALYSIS_MODE_INFO: Record<AnalysisMode, { label: string; description: string; timeReduction: string }> = {
  full: {
    label: 'Full Analysis',
    description: 'Complete analysis of both teams with all metrics',
    timeReduction: 'Baseline'
  },
  my_team: {
    label: 'My Team Focus',
    description: 'Focus on your team\'s performance and improvements',
    timeReduction: '~40% faster'
  },
  opponent: {
    label: 'Scout Opponent',
    description: 'Analyze opponent weaknesses and tactics to beat them',
    timeReduction: '~40% faster'
  },
  quick_overview: {
    label: 'Quick Overview',
    description: 'Fast summary with key stats only',
    timeReduction: '~70% faster'
  }
};
