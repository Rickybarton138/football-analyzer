import React, { useMemo } from 'react';

interface Shot {
  x: number;        // 0-100 pitch coordinates
  y: number;        // 0-100 pitch coordinates
  xg: number;       // Expected goals value
  is_goal: boolean;
  on_target: boolean;
  team: 'home' | 'away';
  player?: number;  // Jersey number
  timestamp_ms?: number;
  shot_type?: string;
}

interface XGShotMapProps {
  shots: Shot[];
  homeTeam?: string;
  awayTeam?: string;
  showXGTimeline?: boolean;
}

export function XGShotMap({ shots, homeTeam = 'Home', awayTeam = 'Away', showXGTimeline = true }: XGShotMapProps) {
  const homeShots = useMemo(() => shots.filter(s => s.team === 'home'), [shots]);
  const awayShots = useMemo(() => shots.filter(s => s.team === 'away'), [shots]);

  const homeXG = useMemo(() => homeShots.reduce((sum, s) => sum + s.xg, 0), [homeShots]);
  const awayXG = useMemo(() => awayShots.reduce((sum, s) => sum + s.xg, 0), [awayShots]);

  const homeGoals = useMemo(() => homeShots.filter(s => s.is_goal).length, [homeShots]);
  const awayGoals = useMemo(() => awayShots.filter(s => s.is_goal).length, [awayShots]);

  // Calculate xG timeline
  const xgTimeline = useMemo(() => {
    const sorted = [...shots].sort((a, b) => (a.timestamp_ms || 0) - (b.timestamp_ms || 0));
    let homeAcc = 0;
    let awayAcc = 0;

    return sorted.map(shot => {
      if (shot.team === 'home') {
        homeAcc += shot.xg;
      } else {
        awayAcc += shot.xg;
      }
      return {
        timestamp_ms: shot.timestamp_ms || 0,
        home_xg: homeAcc,
        away_xg: awayAcc,
        event_team: shot.team,
        is_goal: shot.is_goal
      };
    });
  }, [shots]);

  const getShotColor = (shot: Shot) => {
    if (shot.is_goal) return shot.team === 'home' ? '#22c55e' : '#ef4444'; // Green/Red for goals
    if (shot.on_target) return shot.team === 'home' ? '#3b82f6' : '#f97316'; // Blue/Orange for on target
    return shot.team === 'home' ? '#60a5fa' : '#fdba74'; // Lighter for off target
  };

  const getShotSize = (xg: number) => {
    // Size based on xG value (min 6px, max 24px)
    return Math.max(6, Math.min(24, xg * 50 + 6));
  };

  const getQualityLabel = (xg: number) => {
    if (xg >= 0.4) return 'Big chance';
    if (xg >= 0.2) return 'Good chance';
    if (xg >= 0.08) return 'Average';
    return 'Low quality';
  };

  return (
    <div className="bg-slate-800 rounded-lg p-4">
      {/* Header with xG comparison */}
      <div className="flex justify-between items-center mb-4">
        <div className="text-center">
          <div className="text-sm text-slate-400">{homeTeam}</div>
          <div className="text-3xl font-bold text-blue-400">{homeXG.toFixed(2)}</div>
          <div className="text-xs text-slate-500">{homeShots.length} shots • {homeGoals} goals</div>
        </div>

        <div className="text-center">
          <div className="text-xl font-semibold text-slate-300">xG</div>
          <div className="text-sm text-slate-500">Expected Goals</div>
        </div>

        <div className="text-center">
          <div className="text-sm text-slate-400">{awayTeam}</div>
          <div className="text-3xl font-bold text-red-400">{awayXG.toFixed(2)}</div>
          <div className="text-xs text-slate-500">{awayShots.length} shots • {awayGoals} goals</div>
        </div>
      </div>

      {/* Shot Map - Half pitch view */}
      <div className="relative bg-green-800 rounded-lg overflow-hidden" style={{ aspectRatio: '1.5/1' }}>
        {/* Pitch markings */}
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 66.67" preserveAspectRatio="none">
          {/* Pitch outline */}
          <rect x="0" y="0" width="100" height="66.67" fill="none" stroke="white" strokeWidth="0.5" opacity="0.4" />

          {/* Goal */}
          <rect x="100" y="25" width="0" height="16.67" stroke="white" strokeWidth="1" opacity="0.6" />
          <line x1="100" y1="25" x2="100" y2="41.67" stroke="white" strokeWidth="1" opacity="0.8" />

          {/* 6-yard box */}
          <rect x="94.5" y="22" width="5.5" height="22.67" fill="none" stroke="white" strokeWidth="0.3" opacity="0.4" />

          {/* Penalty area */}
          <rect x="83.5" y="13.33" width="16.5" height="40" fill="none" stroke="white" strokeWidth="0.3" opacity="0.4" />

          {/* Penalty spot */}
          <circle cx="89" cy="33.33" r="0.5" fill="white" opacity="0.5" />

          {/* Penalty arc */}
          <path d="M 83.5 25 A 10 10 0 0 1 83.5 41.67" fill="none" stroke="white" strokeWidth="0.3" opacity="0.4" />

          {/* Center line */}
          <line x1="0" y1="33.33" x2="0" y2="33.33" stroke="white" strokeWidth="0.3" opacity="0.4" />

          {/* Grid lines for reference */}
          {[20, 40, 60, 80].map(x => (
            <line key={x} x1={x} y1="0" x2={x} y2="66.67" stroke="white" strokeWidth="0.1" opacity="0.2" strokeDasharray="2,2" />
          ))}
        </svg>

        {/* Shot markers */}
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 66.67" preserveAspectRatio="none">
          {shots.map((shot, idx) => {
            // Transform coordinates: x stays same, y needs to be scaled
            const plotX = shot.x;
            const plotY = shot.y * 0.6667; // Scale y from 0-100 to 0-66.67
            const size = getShotSize(shot.xg);

            return (
              <g key={idx}>
                {/* Shot marker */}
                <circle
                  cx={plotX}
                  cy={plotY}
                  r={size / 4}
                  fill={getShotColor(shot)}
                  opacity={0.85}
                  stroke={shot.is_goal ? 'white' : 'none'}
                  strokeWidth={shot.is_goal ? 0.5 : 0}
                  className="cursor-pointer hover:opacity-100 transition-opacity"
                >
                  <title>
                    {`${shot.team === 'home' ? homeTeam : awayTeam} #${shot.player || '?'}\n` +
                      `xG: ${shot.xg.toFixed(3)}\n` +
                      `${getQualityLabel(shot.xg)}\n` +
                      `${shot.is_goal ? 'GOAL!' : shot.on_target ? 'On target' : 'Off target'}`}
                  </title>
                </circle>

                {/* Goal indicator */}
                {shot.is_goal && (
                  <text
                    x={plotX}
                    y={plotY + 0.3}
                    textAnchor="middle"
                    fontSize="3"
                    fill="white"
                    fontWeight="bold"
                    pointerEvents="none"
                  >
                    G
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="absolute bottom-2 left-2 bg-black/60 rounded px-2 py-1 text-xs">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-green-500 border border-white"></div>
              <span className="text-white/80">Goal</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="text-white/80">On target</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full bg-blue-300"></div>
              <span className="text-white/80">Off target</span>
            </div>
          </div>
          <div className="text-white/60 mt-1">Size = xG value</div>
        </div>
      </div>

      {/* xG Timeline */}
      {showXGTimeline && xgTimeline.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm text-slate-400 mb-2">xG Timeline</h4>
          <div className="relative h-24 bg-slate-900 rounded">
            <svg className="w-full h-full" preserveAspectRatio="none">
              {/* Grid lines */}
              <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#475569" strokeWidth="1" strokeDasharray="4,4" />

              {/* Home xG line (top half) */}
              {xgTimeline.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points={xgTimeline.map((point, i) => {
                    const x = (i / (xgTimeline.length - 1)) * 100;
                    const y = 50 - (point.home_xg / Math.max(homeXG, awayXG, 1)) * 45;
                    return `${x},${y}`;
                  }).join(' ')}
                />
              )}

              {/* Away xG line (bottom half) */}
              {xgTimeline.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="2"
                  points={xgTimeline.map((point, i) => {
                    const x = (i / (xgTimeline.length - 1)) * 100;
                    const y = 50 + (point.away_xg / Math.max(homeXG, awayXG, 1)) * 45;
                    return `${x},${y}`;
                  }).join(' ')}
                />
              )}

              {/* Goal markers */}
              {xgTimeline.map((point, i) => point.is_goal && (
                <circle
                  key={i}
                  cx={`${(i / Math.max(xgTimeline.length - 1, 1)) * 100}%`}
                  cy={point.event_team === 'home' ? '25%' : '75%'}
                  r="4"
                  fill={point.event_team === 'home' ? '#22c55e' : '#ef4444'}
                  stroke="white"
                  strokeWidth="1"
                />
              ))}
            </svg>

            {/* Labels */}
            <div className="absolute top-1 left-2 text-xs text-blue-400">{homeTeam}</div>
            <div className="absolute bottom-1 left-2 text-xs text-red-400">{awayTeam}</div>
          </div>
        </div>
      )}

      {/* Shot breakdown table */}
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="bg-slate-700/50 rounded p-3">
          <h4 className="text-blue-400 font-medium mb-2">{homeTeam}</h4>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">Big chances (xG &gt; 0.4)</span>
              <span className="text-white">{homeShots.filter(s => s.xg >= 0.4).length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">On target</span>
              <span className="text-white">{homeShots.filter(s => s.on_target).length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">xG per shot</span>
              <span className="text-white">{(homeXG / Math.max(homeShots.length, 1)).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Conversion rate</span>
              <span className="text-white">{((homeGoals / Math.max(homeShots.length, 1)) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-700/50 rounded p-3">
          <h4 className="text-red-400 font-medium mb-2">{awayTeam}</h4>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">Big chances (xG &gt; 0.4)</span>
              <span className="text-white">{awayShots.filter(s => s.xg >= 0.4).length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">On target</span>
              <span className="text-white">{awayShots.filter(s => s.on_target).length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">xG per shot</span>
              <span className="text-white">{(awayXG / Math.max(awayShots.length, 1)).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Conversion rate</span>
              <span className="text-white">{((awayGoals / Math.max(awayShots.length, 1)) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default XGShotMap;
