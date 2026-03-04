export interface PassStats {
  home: TeamPassStats;
  away: TeamPassStats;
}

export interface TeamPassStats {
  total: number;
  completed: number;
  accuracy: number;
  forward: number;
  backward: number;
  sideways: number;
  average_length: number;
  long_balls: number;
}

export interface FormationSnapshot {
  timestamp: number;
  formation: string;
  team: 'home' | 'away';
  confidence: number;
}

export interface FormationStats {
  home: { formations: Record<string, number>; primary: string; changes: FormationSnapshot[] };
  away: { formations: Record<string, number>; primary: string; changes: FormationSnapshot[] };
}

export interface XGShot {
  timestamp: number;
  frame_number: number;
  position: [number, number];
  team: 'home' | 'away';
  xg: number;
  on_target: boolean;
  is_goal: boolean;
}

export interface XGData {
  home_xg: number;
  away_xg: number;
  shots: XGShot[];
}

export interface TacticalEvent {
  event_type: string;
  timestamp: number;
  minute: number;
  team: string;
  description: string;
  details?: Record<string, unknown>;
}

export interface PlayerStats {
  player_name: string;
  total_clips: number;
  total_play_time_seconds: number;
  attacking: {
    ball_touches: number;
    passes_attempted: number;
    passes_completed: number;
    pass_accuracy: number;
    shots: number;
    shots_on_target: number;
    shot_accuracy: number;
  };
  defensive: {
    tackles_attempted: number;
    tackles_won: number;
    tackle_success_rate: number;
    headers: number;
    interceptions: number;
  };
  physical: {
    distance_covered_pixels: number;
    distance_covered_meters_estimate: number;
    sprints: number;
  };
}
