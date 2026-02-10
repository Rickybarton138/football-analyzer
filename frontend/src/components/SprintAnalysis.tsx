import React, { useMemo } from 'react';

interface Sprint {
  player_id: number;
  team: 'home' | 'away';
  jersey_number?: number;
  max_speed_kmh: number;
  distance_m: number;
  duration_seconds: number;
  start_time_ms: number;
}

interface PlayerMovementStats {
  player_id: number;
  team: string;
  jersey_number?: number;
  total_distance_m: number;
  total_distance_km: number;
  max_speed_kmh: number;
  avg_speed_kmh: number;
  sprint_count: number;
  total_sprint_distance_m: number;
  high_intensity_distance_m: number;
  distance_per_minute_m: number;
  time_in_zones_seconds?: Record<string, number>;
}

interface TeamStats {
  team: string;
  player_count: number;
  total_distance_km: number;
  avg_distance_per_player_m: number;
  total_sprints: number;
  avg_sprints_per_player: number;
  total_high_intensity_m: number;
  team_max_speed_kmh: number;
  players: PlayerMovementStats[];
}

interface SprintAnalysisProps {
  homeStats?: TeamStats;
  awayStats?: TeamStats;
  recentSprints?: Sprint[];
  homeTeam?: string;
  awayTeam?: string;
}

export function SprintAnalysis({
  homeStats,
  awayStats,
  recentSprints = [],
  homeTeam = 'Home',
  awayTeam = 'Away'
}: SprintAnalysisProps) {

  // Speed zone colors
  const zoneColors = {
    standing: '#6b7280',
    walking: '#10b981',
    jogging: '#3b82f6',
    running: '#f59e0b',
    high_speed: '#ef4444',
    sprinting: '#dc2626'
  };

  // Speed zone labels
  const zoneLabels: Record<string, string> = {
    standing: 'Standing (0-2 km/h)',
    walking: 'Walking (2-7 km/h)',
    jogging: 'Jogging (7-14 km/h)',
    running: 'Running (14-20 km/h)',
    high_speed: 'High Speed (20-25 km/h)',
    sprinting: 'Sprinting (25+ km/h)'
  };

  const formatTime = (ms: number) => {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}'${seconds.toString().padStart(2, '0')}`;
  };

  const getSpeedColor = (speed: number) => {
    if (speed >= 25) return '#dc2626';
    if (speed >= 20) return '#ef4444';
    if (speed >= 14) return '#f59e0b';
    if (speed >= 7) return '#3b82f6';
    return '#10b981';
  };

  return (
    <div className="bg-slate-800 rounded-lg p-4 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h3 className="text-lg font-semibold text-white">Physical Performance Analysis</h3>
        <p className="text-sm text-slate-400">Sprint detection and distance tracking</p>
      </div>

      {/* Team comparison cards */}
      {homeStats && awayStats && (
        <div className="grid grid-cols-2 gap-4">
          {/* Home team */}
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-blue-400 font-medium">{homeTeam}</span>
              <span className="text-xs text-slate-400">{homeStats.player_count} players</span>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Total Distance</span>
                  <span className="text-white font-semibold">{homeStats.total_distance_km.toFixed(1)} km</span>
                </div>
                <div className="h-2 bg-slate-600 rounded mt-1">
                  <div
                    className="h-full bg-blue-500 rounded"
                    style={{ width: `${Math.min(100, (homeStats.total_distance_km / Math.max(homeStats.total_distance_km, awayStats.total_distance_km || 1)) * 100)}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Total Sprints</span>
                <span className="text-white">{homeStats.total_sprints}</span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-slate-400">High Intensity</span>
                <span className="text-white">{homeStats.total_high_intensity_m.toFixed(0)}m</span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Top Speed</span>
                <span className="text-white font-semibold" style={{ color: getSpeedColor(homeStats.team_max_speed_kmh) }}>
                  {homeStats.team_max_speed_kmh.toFixed(1)} km/h
                </span>
              </div>
            </div>
          </div>

          {/* Away team */}
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-red-400 font-medium">{awayTeam}</span>
              <span className="text-xs text-slate-400">{awayStats.player_count} players</span>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Total Distance</span>
                  <span className="text-white font-semibold">{awayStats.total_distance_km.toFixed(1)} km</span>
                </div>
                <div className="h-2 bg-slate-600 rounded mt-1">
                  <div
                    className="h-full bg-red-500 rounded"
                    style={{ width: `${Math.min(100, (awayStats.total_distance_km / Math.max(homeStats.total_distance_km, awayStats.total_distance_km || 1)) * 100)}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Total Sprints</span>
                <span className="text-white">{awayStats.total_sprints}</span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-slate-400">High Intensity</span>
                <span className="text-white">{awayStats.total_high_intensity_m.toFixed(0)}m</span>
              </div>

              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Top Speed</span>
                <span className="text-white font-semibold" style={{ color: getSpeedColor(awayStats.team_max_speed_kmh) }}>
                  {awayStats.team_max_speed_kmh.toFixed(1)} km/h
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Top performers */}
      {homeStats && awayStats && (
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-sm font-medium text-slate-300 mb-3">Top Performers</h4>

          <div className="grid grid-cols-3 gap-4">
            {/* Distance leaders */}
            <div>
              <div className="text-xs text-slate-400 mb-2">Distance</div>
              <div className="space-y-1">
                {[...homeStats.players, ...awayStats.players]
                  .sort((a, b) => b.total_distance_m - a.total_distance_m)
                  .slice(0, 3)
                  .map((player, idx) => (
                    <div key={player.player_id} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1">
                        <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] ${player.team === 'home' ? 'bg-blue-600' : 'bg-red-600'
                          }`}>
                          {player.jersey_number || '?'}
                        </span>
                        <span className="text-slate-300">#{idx + 1}</span>
                      </div>
                      <span className="text-white">{player.total_distance_km.toFixed(1)}km</span>
                    </div>
                  ))}
              </div>
            </div>

            {/* Sprint leaders */}
            <div>
              <div className="text-xs text-slate-400 mb-2">Sprints</div>
              <div className="space-y-1">
                {[...homeStats.players, ...awayStats.players]
                  .sort((a, b) => b.sprint_count - a.sprint_count)
                  .slice(0, 3)
                  .map((player, idx) => (
                    <div key={player.player_id} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1">
                        <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] ${player.team === 'home' ? 'bg-blue-600' : 'bg-red-600'
                          }`}>
                          {player.jersey_number || '?'}
                        </span>
                        <span className="text-slate-300">#{idx + 1}</span>
                      </div>
                      <span className="text-white">{player.sprint_count}</span>
                    </div>
                  ))}
              </div>
            </div>

            {/* Speed leaders */}
            <div>
              <div className="text-xs text-slate-400 mb-2">Top Speed</div>
              <div className="space-y-1">
                {[...homeStats.players, ...awayStats.players]
                  .sort((a, b) => b.max_speed_kmh - a.max_speed_kmh)
                  .slice(0, 3)
                  .map((player, idx) => (
                    <div key={player.player_id} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1">
                        <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] ${player.team === 'home' ? 'bg-blue-600' : 'bg-red-600'
                          }`}>
                          {player.jersey_number || '?'}
                        </span>
                        <span className="text-slate-300">#{idx + 1}</span>
                      </div>
                      <span className="text-white" style={{ color: getSpeedColor(player.max_speed_kmh) }}>
                        {player.max_speed_kmh.toFixed(1)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent sprints feed */}
      {recentSprints.length > 0 && (
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h4 className="text-sm font-medium text-slate-300 mb-3">Recent Sprints</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {recentSprints.map((sprint, idx) => (
              <div
                key={idx}
                className={`flex items-center justify-between p-2 rounded ${sprint.team === 'home' ? 'bg-blue-900/30' : 'bg-red-900/30'
                  }`}
              >
                <div className="flex items-center gap-2">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${sprint.team === 'home' ? 'bg-blue-600' : 'bg-red-600'
                    }`}>
                    {sprint.jersey_number || '?'}
                  </div>
                  <div>
                    <div className="text-xs text-white">{formatTime(sprint.start_time_ms)}</div>
                    <div className="text-[10px] text-slate-400">
                      {sprint.duration_seconds.toFixed(1)}s • {sprint.distance_m.toFixed(0)}m
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className="text-sm font-semibold"
                    style={{ color: getSpeedColor(sprint.max_speed_kmh) }}
                  >
                    {sprint.max_speed_kmh.toFixed(1)} km/h
                  </div>
                  <div className="text-[10px] text-slate-400">max speed</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Speed zone legend */}
      <div className="bg-slate-700/30 rounded-lg p-3">
        <div className="text-xs text-slate-400 mb-2">Speed Zones</div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(zoneLabels).map(([zone, label]) => (
            <div key={zone} className="flex items-center gap-1 text-xs">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: zoneColors[zone as keyof typeof zoneColors] }}
              />
              <span className="text-slate-300">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* No data state */}
      {!homeStats && !awayStats && recentSprints.length === 0 && (
        <div className="text-center py-8 text-slate-400">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <p>No sprint data available yet</p>
          <p className="text-sm text-slate-500 mt-1">Sprint detection will begin during video analysis</p>
        </div>
      )}
    </div>
  );
}

export default SprintAnalysis;
