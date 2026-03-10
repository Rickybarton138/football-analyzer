import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../../lib/api';
import { Card, CardTitle } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { EmptyState } from '../../components/ui/EmptyState';
import { cn } from '../../lib/utils';
import {
  ArrowLeft, Target, Shield, Zap, Brain,
  Play, ChevronRight, Star, TrendingUp, Activity,
} from 'lucide-react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer,
} from 'recharts';

/* ─── Types ──────────────────────────────────────────────────────────── */

interface ClipData {
  clip_id: string;
  jersey_number: number;
  event_type: string;
  timestamp_start_ms: number;
  timestamp_end_ms: number;
  duration_seconds: number;
  importance: number;
  clip_path: string;
}

interface FourCorner {
  technical: number;
  tactical: number;
  physical: number;
  psychological: number;
}

interface DevPriority {
  priority_level: string;
  area: string;
  detail: string;
  metric_current: string;
  metric_benchmark: string;
  drill: string;
  drill_description: string;
}

interface IDPData {
  jersey_number: number;
  player_name: string;
  position: string;
  overall_rating: number;
  four_corner: FourCorner;
  key_strengths: string[];
  development_priorities: DevPriority[];
  weekly_focus: string;
  session_plan: string;
  three_month_goals: string[];
  six_month_goals: string[];
  raw_text: string;
  generated_at: string;
}

interface ClipFeedback {
  clip_id: string;
  feedback: {
    overall_rating: number;
    strengths: string[];
    weaknesses: string[];
    coaching_points: string[];
    drill_recommendations: string[];
    raw_text: string;
  };
}

/* ─── Helpers ────────────────────────────────────────────────────────── */

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, '0')}`;
}

function priorityColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'HIGH': return 'text-red-600 bg-red-50 border-red-200';
    case 'MEDIUM': return 'text-amber-600 bg-amber-50 border-amber-200';
    case 'LOW': return 'text-sky bg-sky-light border-sky/20';
    default: return 'text-text-muted bg-surface-alt border-border';
  }
}

const POSITION_OPTIONS = ['GK', 'CB', 'FB', 'CM', 'CAM', 'WM', 'ST'];

/* ─── Component ──────────────────────────────────────────────────────── */

export default function PlayerDetailPage() {
  const { playerId } = useParams<{ playerId: string }>();
  const navigate = useNavigate();
  const jerseyNumber = Number(playerId) || 0;

  const [tab, setTab] = useState<'clips' | 'stats' | 'idp'>('clips');
  const [clips, setClips] = useState<ClipData[]>([]);
  const [idp, setIdp] = useState<IDPData | null>(null);
  const [loading, setLoading] = useState(true);
  const [idpLoading, setIdpLoading] = useState(false);
  const [position, setPosition] = useState('CM');
  const [selectedClip, setSelectedClip] = useState<ClipData | null>(null);
  const [clipFeedback, setClipFeedback] = useState<ClipFeedback | null>(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  // Load clips on mount
  useEffect(() => {
    if (!jerseyNumber) return;
    setLoading(true);
    api.get<{ clips: ClipData[] }>(`/api/player/${jerseyNumber}/clips`)
      .then(d => setClips(d.clips || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [jerseyNumber]);

  // Generate IDP
  const generateIDP = useCallback(async () => {
    setIdpLoading(true);
    try {
      const data = await api.post<IDPData>(`/api/player/${jerseyNumber}/generate-idp`, {
        position,
        player_name: '',
        team: 'home',
      });
      setIdp(data);
      setTab('idp');
    } catch {
      // silent
    }
    setIdpLoading(false);
  }, [jerseyNumber, position]);

  // Load clip feedback
  const loadClipFeedback = useCallback(async (clip: ClipData) => {
    setSelectedClip(clip);
    setFeedbackLoading(true);
    try {
      const data = await api.get<ClipFeedback>(
        `/api/player/${jerseyNumber}/clips/${clip.clip_id}/feedback?position=${position}`
      );
      setClipFeedback(data);
    } catch {
      setClipFeedback(null);
    }
    setFeedbackLoading(false);
  }, [jerseyNumber, position]);

  if (loading) return <Spinner label="Loading player data..." className="py-20" />;

  const radarData = idp ? [
    { corner: 'Technical', value: idp.four_corner.technical },
    { corner: 'Tactical', value: idp.four_corner.tactical },
    { corner: 'Physical', value: idp.four_corner.physical },
    { corner: 'Psychological', value: idp.four_corner.psychological },
  ] : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-pitch-light rounded-card p-6 border border-pitch/20">
        <div className="flex items-center gap-4">
          <button onClick={() => navigate(-1)} className="text-text-muted hover:text-text-primary transition">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="w-14 h-14 bg-gradient-to-br from-pitch-deep to-pitch rounded-full flex items-center justify-center text-white font-bold text-lg">
            {jerseyNumber > 0 ? `#${jerseyNumber}` : '?'}
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-text-primary">
              Player #{jerseyNumber}
            </h2>
            <div className="flex items-center gap-3 mt-1">
              <select
                value={position}
                onChange={e => setPosition(e.target.value)}
                className="text-sm border border-border rounded-btn px-2 py-1 bg-white text-text-primary"
              >
                {POSITION_OPTIONS.map(p => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              <span className="text-text-muted text-sm">{clips.length} clips</span>
              {idp && (
                <span className="text-pitch font-semibold text-sm">
                  Rating: {idp.overall_rating}/10
                </span>
              )}
            </div>
          </div>
          <button
            onClick={generateIDP}
            disabled={idpLoading}
            className="bg-gradient-to-r from-pitch-deep to-pitch text-white px-4 py-2 rounded-btn text-sm font-medium hover:opacity-90 transition disabled:opacity-50 flex items-center gap-2"
          >
            {idpLoading ? <Spinner size="sm" /> : <Brain className="w-4 h-4" />}
            Generate IDP
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-surface-alt rounded-card p-1">
        {(['clips', 'stats', 'idp'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              'flex-1 py-2 px-4 rounded-btn text-sm font-medium transition',
              tab === t
                ? 'bg-white text-pitch shadow-sm'
                : 'text-text-muted hover:text-text-primary'
            )}
          >
            {t === 'clips' && <Play className="w-3.5 h-3.5 inline mr-1.5" />}
            {t === 'stats' && <Activity className="w-3.5 h-3.5 inline mr-1.5" />}
            {t === 'idp' && <TrendingUp className="w-3.5 h-3.5 inline mr-1.5" />}
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      {/* ── Clips Tab ─────────────────────────────────────────────────── */}
      {tab === 'clips' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Clip List */}
          <Card>
            <CardTitle>Ball-Interaction Clips</CardTitle>
            {clips.length > 0 ? (
              <div className="space-y-2 mt-3 max-h-[500px] overflow-y-auto">
                {clips.map(clip => (
                  <button
                    key={clip.clip_id}
                    onClick={() => loadClipFeedback(clip)}
                    className={cn(
                      'w-full text-left p-3 rounded-btn border transition-all',
                      selectedClip?.clip_id === clip.clip_id
                        ? 'bg-pitch-light border-pitch/30'
                        : 'border-transparent hover:bg-surface-alt'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium text-text-primary capitalize">
                          {clip.event_type}
                        </span>
                        <span className="text-xs text-text-muted ml-2">
                          {formatTime(clip.timestamp_start_ms)}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-text-muted">
                          {clip.duration_seconds.toFixed(1)}s
                        </span>
                        <ChevronRight className="w-4 h-4 text-text-muted" />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<Play className="w-6 h-6 text-text-muted" />}
                title="No Clips Yet"
                description="Clips are generated during video analysis"
              />
            )}
          </Card>

          {/* Clip Feedback */}
          <Card>
            <CardTitle>AI Coaching Feedback</CardTitle>
            {feedbackLoading ? (
              <Spinner label="Generating feedback..." className="py-10" />
            ) : clipFeedback ? (
              <div className="mt-3 space-y-4">
                <div className="flex items-center gap-2">
                  <Star className="w-5 h-5 text-gold" />
                  <span className="text-lg font-bold text-text-primary">
                    {clipFeedback.feedback.overall_rating}/10
                  </span>
                </div>

                {clipFeedback.feedback.strengths.length > 0 && (
                  <div>
                    <h4 className="text-pitch text-sm font-semibold flex items-center gap-1 mb-1">
                      <Target className="w-3.5 h-3.5" /> Strengths
                    </h4>
                    <ul className="text-sm text-text-secondary space-y-1">
                      {clipFeedback.feedback.strengths.map((s, i) => (
                        <li key={i} className="pl-3 border-l-2 border-pitch/30">{s}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {clipFeedback.feedback.weaknesses.length > 0 && (
                  <div>
                    <h4 className="text-amber-600 text-sm font-semibold flex items-center gap-1 mb-1">
                      <Shield className="w-3.5 h-3.5" /> Areas to Improve
                    </h4>
                    <ul className="text-sm text-text-secondary space-y-1">
                      {clipFeedback.feedback.weaknesses.map((w, i) => (
                        <li key={i} className="pl-3 border-l-2 border-amber-300">{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {clipFeedback.feedback.coaching_points.length > 0 && (
                  <div>
                    <h4 className="text-sky text-sm font-semibold flex items-center gap-1 mb-1">
                      <Zap className="w-3.5 h-3.5" /> Coaching Points
                    </h4>
                    <ul className="text-sm text-text-secondary space-y-1">
                      {clipFeedback.feedback.coaching_points.map((c, i) => (
                        <li key={i} className="pl-3 border-l-2 border-sky/30">{c}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {clipFeedback.feedback.drill_recommendations.length > 0 && (
                  <div className="bg-pitch-light rounded-btn p-3 border border-pitch/10">
                    <h4 className="text-pitch-deep text-sm font-semibold mb-1">Recommended Drill</h4>
                    {clipFeedback.feedback.drill_recommendations.map((d, i) => (
                      <p key={i} className="text-sm text-text-secondary">{d}</p>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <EmptyState
                icon={<Brain className="w-6 h-6 text-text-muted" />}
                title="Select a Clip"
                description="Click a clip to get AI coaching feedback"
              />
            )}
          </Card>
        </div>
      )}

      {/* ── Stats Tab ─────────────────────────────────────────────────── */}
      {tab === 'stats' && (
        <div className="space-y-6">
          {idp ? (
            <>
              {/* Radar Chart */}
              <Card>
                <CardTitle>Four Corner Assessment</CardTitle>
                <div className="h-64 mt-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#e5e7eb" />
                      <PolarAngleAxis dataKey="corner" tick={{ fill: '#444', fontSize: 12 }} />
                      <PolarRadiusAxis domain={[0, 10]} tick={{ fill: '#888', fontSize: 10 }} />
                      <Radar
                        dataKey="value"
                        stroke="#43A047"
                        fill="#43A047"
                        fillOpacity={0.2}
                        strokeWidth={2}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* Four corner scores */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {([
                  { label: 'Technical', value: idp.four_corner.technical, icon: Target, color: 'text-pitch' },
                  { label: 'Tactical', value: idp.four_corner.tactical, icon: Brain, color: 'text-sky' },
                  { label: 'Physical', value: idp.four_corner.physical, icon: Zap, color: 'text-amber-600' },
                  { label: 'Psychological', value: idp.four_corner.psychological, icon: Shield, color: 'text-purple-600' },
                ] as const).map(({ label, value, icon: Icon, color }) => (
                  <Card key={label}>
                    <div className="text-center">
                      <Icon className={cn('w-5 h-5 mx-auto mb-1', color)} />
                      <div className="text-2xl font-bold text-text-primary">{value.toFixed(1)}</div>
                      <div className="text-xs text-text-muted">{label}</div>
                    </div>
                  </Card>
                ))}
              </div>
            </>
          ) : (
            <Card>
              <EmptyState
                icon={<Activity className="w-8 h-8 text-text-muted" />}
                title="No Stats Available"
                description="Generate an IDP to see the Four Corner assessment and stats"
              />
            </Card>
          )}
        </div>
      )}

      {/* ── IDP Tab ───────────────────────────────────────────────────── */}
      {tab === 'idp' && (
        <div className="space-y-6">
          {idp ? (
            <>
              {/* Overall Rating */}
              <Card className="border-pitch/20">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-bold text-text-primary">
                      Individual Development Plan
                    </h3>
                    <p className="text-text-muted text-sm">
                      {idp.position} | Generated {new Date(idp.generated_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-pitch">{idp.overall_rating}</div>
                    <div className="text-xs text-text-muted">/10</div>
                  </div>
                </div>
              </Card>

              {/* Key Strengths */}
              {idp.key_strengths.length > 0 && (
                <Card>
                  <CardTitle>Key Strengths</CardTitle>
                  <ul className="mt-3 space-y-2">
                    {idp.key_strengths.map((s, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                        <div className="w-5 h-5 bg-pitch-light rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-pitch text-xs font-bold">{i + 1}</span>
                        </div>
                        {s}
                      </li>
                    ))}
                  </ul>
                </Card>
              )}

              {/* Development Priorities */}
              {idp.development_priorities.length > 0 && (
                <Card>
                  <CardTitle>Development Priorities</CardTitle>
                  <div className="mt-3 space-y-3">
                    {idp.development_priorities.map((dp, i) => (
                      <div
                        key={i}
                        className={cn(
                          'p-3 rounded-btn border',
                          priorityColor(dp.priority_level)
                        )}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-bold uppercase">
                            [{dp.priority_level}]
                          </span>
                          <span className="text-sm font-medium">{dp.area}</span>
                        </div>
                        {dp.drill && (
                          <p className="text-xs mt-1 opacity-80">
                            Drill: {dp.drill}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {/* Weekly Focus & Session Plan */}
              {(idp.weekly_focus || idp.session_plan) && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {idp.weekly_focus && (
                    <Card>
                      <CardTitle>Weekly Focus</CardTitle>
                      <p className="mt-2 text-sm text-text-secondary">{idp.weekly_focus}</p>
                    </Card>
                  )}
                  {idp.session_plan && (
                    <Card>
                      <CardTitle>Session Plan</CardTitle>
                      <p className="mt-2 text-sm text-text-secondary">{idp.session_plan}</p>
                    </Card>
                  )}
                </div>
              )}

              {/* Goals */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {idp.three_month_goals.length > 0 && (
                  <Card>
                    <CardTitle>3-Month Goals</CardTitle>
                    <ul className="mt-2 space-y-1">
                      {idp.three_month_goals.map((g, i) => (
                        <li key={i} className="text-sm text-text-secondary flex items-start gap-2">
                          <TrendingUp className="w-3.5 h-3.5 text-pitch mt-0.5 flex-shrink-0" />
                          {g}
                        </li>
                      ))}
                    </ul>
                  </Card>
                )}
                {idp.six_month_goals.length > 0 && (
                  <Card>
                    <CardTitle>6-Month Goals</CardTitle>
                    <ul className="mt-2 space-y-1">
                      {idp.six_month_goals.map((g, i) => (
                        <li key={i} className="text-sm text-text-secondary flex items-start gap-2">
                          <Star className="w-3.5 h-3.5 text-gold mt-0.5 flex-shrink-0" />
                          {g}
                        </li>
                      ))}
                    </ul>
                  </Card>
                )}
              </div>
            </>
          ) : (
            <Card>
              <EmptyState
                icon={<Brain className="w-8 h-8 text-text-muted" />}
                title="No IDP Generated"
                description="Click 'Generate IDP' to create an Individual Development Plan using AI"
                action={
                  <button
                    onClick={generateIDP}
                    disabled={idpLoading}
                    className="mt-3 bg-gradient-to-r from-pitch-deep to-pitch text-white px-4 py-2 rounded-btn text-sm font-medium"
                  >
                    Generate IDP
                  </button>
                }
              />
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
