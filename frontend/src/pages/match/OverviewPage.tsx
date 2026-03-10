import { useMemo, useEffect, useState } from 'react';
import { useMatchStore } from '../../stores/matchStore';
import { useUIStore } from '../../stores/uiStore';
import { Card, CardTitle } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { api } from '../../lib/api';
import { formatDuration, cn } from '../../lib/utils';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Shield, Crosshair, Users } from 'lucide-react';

const VIDEO_WIDTH = 1920;

export default function OverviewPage() {
  const { analysis } = useMatchStore();
  const { selectedPeriod } = useUIStore();
  const [summary, setSummary] = useState<{ tactical_summary?: string; overall_rating?: string; key_strengths?: string[]; areas_to_improve?: string[] } | null>(null);

  useEffect(() => {
    api.get<{ summary?: typeof summary }>('/api/ai-coach/summary').then(d => setSummary(d.summary || d as any)).catch(() => {});
  }, []);

  const stats = useMemo(() => {
    if (!analysis) return null;
    return calculateMatchStats(analysis, selectedPeriod);
  }, [analysis, selectedPeriod]);

  if (!analysis || !stats) return <Spinner label="Loading overview..." className="py-20" />;

  const possData = [
    { name: 'Home', value: stats.possession.home },
    { name: 'Away', value: stats.possession.away },
  ];

  return (
    <div className="space-y-6">
      {/* Match Summary Header */}
      <div className="bg-pitch-light rounded-card p-6 border border-pitch/20">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-text-primary">Match Overview</h2>
            <p className="text-text-muted text-sm">{formatDuration(analysis.duration_seconds)} analyzed &middot; {analysis.analyzed_frames} frames</p>
          </div>
          {summary?.overall_rating && (
            <div className="text-right">
              <span className="text-text-muted text-xs block mb-1">AI Rating</span>
              <span className="text-2xl font-bold text-pitch">{summary.overall_rating}</span>
            </div>
          )}
        </div>
        {summary?.tactical_summary && (
          <p className="text-text-secondary text-sm">{summary.tactical_summary}</p>
        )}
      </div>

      {/* Key Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Avg Home Players" value={stats.avgHomePlayers.toFixed(1)} icon={<Users className="w-5 h-5" />} color="text-blue-400" />
        <StatCard label="Avg Away Players" value={stats.avgAwayPlayers.toFixed(1)} icon={<Users className="w-5 h-5" />} color="text-red-400" />
        <StatCard label="Territorial Adv." value={`${stats.territorialAdvantage.home}%`} icon={<Crosshair className="w-5 h-5" />} color="text-pitch" />
        <StatCard label="Pressing Actions" value={String(stats.pressingActions.home)} icon={<Shield className="w-5 h-5" />} color="text-sky" />
      </div>

      {/* Possession + Zones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardTitle>Possession</CardTitle>
          <div className="flex items-center justify-center gap-8">
            <div className="w-40 h-40">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={possData} dataKey="value" innerRadius={45} outerRadius={70} paddingAngle={2}>
                    <Cell fill="#3b82f6" />
                    <Cell fill="#ef4444" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-blue-500 rounded-full" />
                <span className="text-text-primary font-medium">{stats.possession.home}%</span>
                <span className="text-text-muted text-sm">Home</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-red-500 rounded-full" />
                <span className="text-text-primary font-medium">{stats.possession.away}%</span>
                <span className="text-text-muted text-sm">Away</span>
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <CardTitle>Territorial Control by Zone</CardTitle>
          <div className="space-y-3 mt-4">
            {['Defensive', 'Midfield', 'Attacking'].map((zone, i) => (
              <div key={zone}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-text-muted">{zone}</span>
                  <span className="text-text-secondary">{stats.possessionByZone[i]?.home || 0}% — {stats.possessionByZone[i]?.away || 0}%</span>
                </div>
                <div className="flex h-2 rounded-full overflow-hidden bg-surface-alt">
                  <div className="bg-blue-500" style={{ width: `${stats.possessionByZone[i]?.home || 0}%` }} />
                  <div className="flex-1" />
                  <div className="bg-red-500" style={{ width: `${stats.possessionByZone[i]?.away || 0}%` }} />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Strengths / Improvements */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {summary.key_strengths && summary.key_strengths.length > 0 && (
            <Card className="border-pitch/20">
              <CardTitle className="flex items-center gap-2 text-pitch">
                <TrendingUp className="w-5 h-5" /> Key Strengths
              </CardTitle>
              <ul className="space-y-2 mt-3">
                {summary.key_strengths.map((s, i) => (
                  <li key={i} className="text-text-secondary text-sm flex items-start gap-2">
                    <span className="text-pitch-deep mt-0.5">&#x2022;</span> {s}
                  </li>
                ))}
              </ul>
            </Card>
          )}
          {summary.areas_to_improve && summary.areas_to_improve.length > 0 && (
            <Card className="border-amber-500/20">
              <CardTitle className="flex items-center gap-2 text-amber-400">
                <TrendingDown className="w-5 h-5" /> Areas to Improve
              </CardTitle>
              <ul className="space-y-2 mt-3">
                {summary.areas_to_improve.map((a, i) => (
                  <li key={i} className="text-text-secondary text-sm flex items-start gap-2">
                    <span className="text-amber-500 mt-0.5">&#x2022;</span> {a}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      {/* AI Insights */}
      {stats.insights.length > 0 && (
        <Card>
          <CardTitle>Match Insights</CardTitle>
          <div className="grid gap-3 mt-3">
            {stats.insights.map((insight, i) => (
              <div key={i} className={cn('rounded-lg p-4 border',
                insight.type === 'positive' ? 'bg-pitch-light border-pitch/20' :
                insight.type === 'warning' ? 'bg-amber-500/10 border-amber-500/20' :
                'bg-surface-alt border-border'
              )}>
                <h4 className={cn('font-semibold text-sm mb-1',
                  insight.type === 'positive' ? 'text-pitch' :
                  insight.type === 'warning' ? 'text-amber-400' : 'text-text-secondary'
                )}>{insight.title}</h4>
                <p className="text-text-muted text-sm">{insight.description}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

function StatCard({ label, value, icon, color }: { label: string; value: string; icon: React.ReactNode; color: string }) {
  return (
    <Card className="text-center">
      <div className={cn('mx-auto mb-2', color)}>{icon}</div>
      <div className="text-2xl font-bold text-text-primary">{value}</div>
      <div className="text-text-muted text-xs mt-1">{label}</div>
    </Card>
  );
}

// Re-use the stats calculation from old App.tsx
function calculateMatchStats(analysis: { frames: Array<{ home_players: number; away_players: number; timestamp: number; detections: Array<{ team: string; bbox: [number, number, number, number] }> }> }, period: 'full' | '1st' | '2nd') {
  const frames = period === 'full' ? analysis.frames :
    period === '1st' ? analysis.frames.slice(0, Math.floor(analysis.frames.length / 2)) :
    analysis.frames.slice(Math.floor(analysis.frames.length / 2));

  const avgHomePlayers = frames.reduce((sum, f) => sum + f.home_players, 0) / (frames.length || 1);
  const avgAwayPlayers = frames.reduce((sum, f) => sum + f.away_players, 0) / (frames.length || 1);

  let homeTerritorySum = 0, awayTerritorySum = 0;
  let homeDefZone = 0, homeMidZone = 0, homeAttZone = 0;
  let awayDefZone = 0, awayMidZone = 0, awayAttZone = 0;
  let homeCompactFrames = 0, homePressing = 0, awayPressing = 0;
  let homeHighIntensity = 0, awayHighIntensity = 0;

  frames.forEach((frame, idx) => {
    const homeD = frame.detections.filter(d => d.team === 'home');
    const awayD = frame.detections.filter(d => d.team === 'away');

    homeD.forEach(d => {
      const x = (d.bbox[0] + d.bbox[2]) / 2 / VIDEO_WIDTH;
      homeTerritorySum += x;
      if (x < 0.33) homeDefZone++; else if (x < 0.67) homeMidZone++; else homeAttZone++;
    });
    awayD.forEach(d => {
      const x = (d.bbox[0] + d.bbox[2]) / 2 / VIDEO_WIDTH;
      awayTerritorySum += 1 - x;
      if (x > 0.67) awayDefZone++; else if (x > 0.33) awayMidZone++; else awayAttZone++;
    });

    if (homeD.length >= 3) {
      const xs = homeD.map(d => (d.bbox[0] + d.bbox[2]) / 2);
      if (Math.max(...xs) - Math.min(...xs) < VIDEO_WIDTH * 0.4) homeCompactFrames++;
    }

    if (homeD.filter(d => (d.bbox[0] + d.bbox[2]) / 2 > VIDEO_WIDTH * 0.5).length >= 4) homePressing++;
    if (awayD.filter(d => (d.bbox[0] + d.bbox[2]) / 2 < VIDEO_WIDTH * 0.5).length >= 4) awayPressing++;

    if (idx > 0) {
      const prev = frames[idx - 1];
      const prevHomeX = prev.detections.filter(d => d.team === 'home').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (prev.home_players || 1);
      const currHomeX = homeD.reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (homeD.length || 1);
      if (Math.abs(currHomeX - prevHomeX) > 100) homeHighIntensity++;
      const prevAwayX = prev.detections.filter(d => d.team === 'away').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (prev.away_players || 1);
      const currAwayX = awayD.reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (awayD.length || 1);
      if (Math.abs(currAwayX - prevAwayX) > 100) awayHighIntensity++;
    }
  });

  const totalHome = homeDefZone + homeMidZone + homeAttZone || 1;
  const totalAway = awayDefZone + awayMidZone + awayAttZone || 1;
  const homePoss = Math.round((homeTerritorySum / (homeTerritorySum + awayTerritorySum + 0.1)) * 100);
  const homeTerritory = Math.round((homeAttZone / totalHome) * 100);

  const insights: { type: 'positive' | 'warning' | 'info'; title: string; description: string }[] = [];
  if (homePressing > frames.length * 0.3)
    insights.push({ type: 'positive', title: 'Strong Pressing', description: 'Excellent pressing intensity with 4+ players regularly in the opposition half.' });
  else
    insights.push({ type: 'warning', title: 'Pressing Opportunities', description: 'Consider pressing higher. Look for triggers like poor first touches.' });
  if (homeCompactFrames > frames.length * 0.5)
    insights.push({ type: 'positive', title: 'Solid Defensive Shape', description: 'Good compactness maintained, making it hard for opposition to find space.' });
  else
    insights.push({ type: 'warning', title: 'Shape Work Needed', description: 'Players spreading too wide at times. Work on staying compact.' });
  if (homeTerritory > 40)
    insights.push({ type: 'positive', title: 'Attacking Presence', description: `${homeTerritory}% of positions in the attacking third. Good forward commitment.` });
  else
    insights.push({ type: 'info', title: 'Push Forward', description: 'Consider committing more players forward when in possession.' });

  return {
    avgHomePlayers, avgAwayPlayers, homeScore: 0, awayScore: 0,
    possession: { home: homePoss, away: 100 - homePoss },
    territorialAdvantage: { home: homeTerritory, away: Math.round((awayAttZone / totalAway) * 100) },
    pressingActions: { home: homePressing, away: awayPressing },
    highIntensityMoments: { home: homeHighIntensity, away: awayHighIntensity },
    compactness: { home: Math.round((homeCompactFrames / (frames.length || 1)) * 100), away: 0 },
    widthUsage: { home: 0, away: 0 },
    possessionByZone: [
      { home: Math.round((homeDefZone / totalHome) * 100), away: Math.round((awayAttZone / totalAway) * 100) },
      { home: Math.round((homeMidZone / totalHome) * 100), away: Math.round((awayMidZone / totalAway) * 100) },
      { home: Math.round((homeAttZone / totalHome) * 100), away: Math.round((awayDefZone / totalAway) * 100) },
    ],
    momentumPeriods: [], momentumShifts: [], periodStats: [], insights,
  };
}
