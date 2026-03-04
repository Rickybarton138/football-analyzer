export interface CoachingInsight {
  category: string;
  priority: string;
  title: string;
  message: string;
  recommendation: string;
  supporting_data: Record<string, unknown>;
}

export interface MatchSummary {
  overall_rating: string;
  key_strengths: string[];
  areas_to_improve: string[];
  tactical_summary: string;
  half_time_message: string;
  full_time_message: string;
}

export interface AICoachData {
  summary: MatchSummary | null;
  insights: CoachingInsight[];
  critical_insights: CoachingInsight[];
  total_insights: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  confidence?: string;
  relatedInsights?: string[];
}

export interface PriorityArea {
  area: string;
  team: string;
  severity: 'high' | 'medium';
  metric: string;
  drill: string;
  detail: string;
  duration_mins: number;
}

export interface SessionPlan {
  warm_up: { activity: string; duration_mins: number };
  main_focus: { area: string; drill: string; detail: string; duration_mins: number };
  secondary_focus: { area: string; drill: string; detail: string; duration_mins: number };
  game: { activity: string; conditions: string; duration_mins: number };
  cool_down: { activity: string; duration_mins: number };
}

export interface TrainingFocusData {
  priority_areas: PriorityArea[];
  session_plan: SessionPlan;
  generated_at?: string;
}

export interface TacticalAlert {
  alert_id: string;
  alert_type: string;
  timestamp: number;
  minute: number;
  team: string;
  description: string;
  severity: string;
  details?: Record<string, unknown>;
}
