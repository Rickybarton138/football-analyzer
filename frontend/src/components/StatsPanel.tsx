import { useMemo } from 'react';
import type { MatchState, TeamMetrics } from '../types';

interface StatsPanelProps {
  matchState: MatchState | null;
  homeMetrics?: TeamMetrics;
  awayMetrics?: TeamMetrics;
}

export function StatsPanel({ matchState, homeMetrics, awayMetrics }: StatsPanelProps) {
  if (!matchState && !homeMetrics && !awayMetrics) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400">
        <p>No match data available</p>
      </div>
    );
  }

  const matchTime = useMemo(() => {
    if (!matchState) return '00:00';
    const minutes = Math.floor(matchState.current_time_ms / 60000);
    const seconds = Math.floor((matchState.current_time_ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, [matchState]);

  return (
    <div className="h-full flex flex-col gap-4 text-sm">
      {/* Score and Time */}
      {matchState && (
        <div className="text-center py-2 bg-slate-700/50 rounded">
          <div className="text-2xl font-bold">
            <span className="text-blue-400">{matchState.home_score}</span>
            <span className="text-slate-400 mx-2">-</span>
            <span className="text-red-400">{matchState.away_score}</span>
          </div>
          <div className="text-slate-400 text-xs mt-1">
            {matchState.period === 1 ? '1st Half' : '2nd Half'} • {matchTime}
          </div>
        </div>
      )}

      {/* Key Stats Comparison */}
      {homeMetrics && awayMetrics && (
        <div className="flex-1 space-y-3">
          <StatBar
            label="Possession"
            homeValue={homeMetrics.possession_pct}
            awayValue={awayMetrics.possession_pct}
            format={(v) => `${v.toFixed(0)}%`}
          />

          <StatBar
            label="Pass Accuracy"
            homeValue={homeMetrics.pass_completion_pct}
            awayValue={awayMetrics.pass_completion_pct}
            format={(v) => `${v.toFixed(0)}%`}
          />

          <StatBar
            label="Shots"
            homeValue={homeMetrics.shots}
            awayValue={awayMetrics.shots}
          />

          <StatBar
            label="xG"
            homeValue={homeMetrics.xg}
            awayValue={awayMetrics.xg}
            format={(v) => v.toFixed(2)}
          />

          <StatBar
            label="Distance (km)"
            homeValue={homeMetrics.total_distance_km}
            awayValue={awayMetrics.total_distance_km}
            format={(v) => v.toFixed(1)}
          />

          {homeMetrics.avg_formation && (
            <div className="flex justify-between items-center text-xs">
              <span className="text-blue-400">{homeMetrics.avg_formation}</span>
              <span className="text-slate-400">Formation</span>
              <span className="text-red-400">{awayMetrics.avg_formation || '-'}</span>
            </div>
          )}
        </div>
      )}

      {/* Recent Events */}
      {matchState && matchState.recent_events.length > 0 && (
        <div className="border-t border-slate-700 pt-3">
          <h4 className="text-xs text-slate-400 mb-2">Recent Events</h4>
          <div className="space-y-1 max-h-24 overflow-y-auto">
            {matchState.recent_events.slice(0, 5).map((event) => (
              <div
                key={event.event_id}
                className="flex justify-between items-center text-xs py-1"
              >
                <span className={event.team === 'home' ? 'text-blue-400' : 'text-red-400'}>
                  {formatEventType(event.event_type)}
                </span>
                <span className="text-slate-500">
                  {formatTime(event.timestamp_ms)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface StatBarProps {
  label: string;
  homeValue: number;
  awayValue: number;
  format?: (value: number) => string;
}

function StatBar({ label, homeValue, awayValue, format = String }: StatBarProps) {
  const total = homeValue + awayValue || 1;
  const homePercent = (homeValue / total) * 100;
  const awayPercent = (awayValue / total) * 100;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-blue-400 font-medium">{format(homeValue)}</span>
        <span className="text-slate-400">{label}</span>
        <span className="text-red-400 font-medium">{format(awayValue)}</span>
      </div>
      <div className="flex h-2 rounded overflow-hidden bg-slate-700">
        <div
          className="bg-blue-500 transition-all duration-500"
          style={{ width: `${homePercent}%` }}
        />
        <div
          className="bg-red-500 transition-all duration-500"
          style={{ width: `${awayPercent}%` }}
        />
      </div>
    </div>
  );
}

function formatEventType(type: string): string {
  const typeMap: Record<string, string> = {
    pass: 'Pass',
    shot: 'Shot',
    tackle: 'Tackle',
    interception: 'Interception',
    foul: 'Foul',
    goal: 'GOAL',
    corner: 'Corner',
    throw_in: 'Throw-in',
    free_kick: 'Free kick',
    offside: 'Offside',
  };
  return typeMap[type] || type;
}

function formatTime(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  return `${minutes}'${seconds.toString().padStart(2, '0')}`;
}
