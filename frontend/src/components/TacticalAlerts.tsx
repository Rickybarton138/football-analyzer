import React, { useState, useEffect } from 'react';

interface TacticalAlert {
  type: string;
  priority: number;
  priority_name: string;
  team: 'home' | 'away';
  message: string;
  recommendation: string;
  frame: number;
  timestamp_ms: number;
  time_str: string;
  position?: [number, number];
  involved_players?: number[];
  expires_at_frame: number;
}

interface TacticalAlertsProps {
  alerts: TacticalAlert[];
  currentFrame: number;
  onDismiss?: (frame: number) => void;
  homeTeam?: string;
  awayTeam?: string;
  maxVisible?: number;
}

export function TacticalAlerts({
  alerts,
  currentFrame,
  onDismiss,
  homeTeam = 'Home',
  awayTeam = 'Away',
  maxVisible = 5
}: TacticalAlertsProps) {

  const [dismissed, setDismissed] = useState<Set<number>>(new Set());

  // Filter active alerts
  const activeAlerts = alerts
    .filter(alert => alert.expires_at_frame > currentFrame && !dismissed.has(alert.frame))
    .sort((a, b) => b.priority - a.priority) // Highest priority first
    .slice(0, maxVisible);

  const handleDismiss = (frame: number) => {
    setDismissed(prev => new Set(prev).add(frame));
    onDismiss?.(frame);
  };

  const getPriorityStyles = (priority: number, priorityName: string) => {
    switch (priorityName) {
      case 'CRITICAL':
        return {
          bg: 'bg-red-900/90',
          border: 'border-red-500',
          icon: '🚨',
          pulse: true
        };
      case 'HIGH':
        return {
          bg: 'bg-orange-900/80',
          border: 'border-orange-500',
          icon: '⚠️',
          pulse: false
        };
      case 'MEDIUM':
        return {
          bg: 'bg-yellow-900/70',
          border: 'border-yellow-500',
          icon: '💡',
          pulse: false
        };
      default:
        return {
          bg: 'bg-slate-700/70',
          border: 'border-slate-500',
          icon: 'ℹ️',
          pulse: false
        };
    }
  };

  const getAlertTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      'space_behind_defense': '🏃',
      'overload_opportunity': '👥',
      'switch_play': '↔️',
      'through_ball_lane': '⬆️',
      'third_man_run': '🔄',
      'cutback_zone': '↩️',
      'defensive_gap': '🕳️',
      'unmarked_runner': '❗',
      'high_line_vulnerable': '📈',
      'transition_danger': '⚡',
      'wide_area_exposed': '📐',
      'pressing_trap': '🪤',
      'press_trigger': '🎯',
      'press_release': '🔓',
      'counter_press': '🔄',
      'formation_shift': '📊',
      'momentum_shift': '📈'
    };
    return icons[type] || '📋';
  };

  const getTeamColor = (team: string) => {
    return team === 'home' ? 'text-blue-400' : 'text-red-400';
  };

  // Auto-remove expired alerts from dismissed set (cleanup)
  useEffect(() => {
    const expiredFrames = alerts
      .filter(a => a.expires_at_frame <= currentFrame)
      .map(a => a.frame);

    if (expiredFrames.length > 0) {
      setDismissed(prev => {
        const next = new Set(prev);
        expiredFrames.forEach(f => next.delete(f));
        return next;
      });
    }
  }, [currentFrame, alerts]);

  if (activeAlerts.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      {activeAlerts.map((alert) => {
        const styles = getPriorityStyles(alert.priority, alert.priority_name);

        return (
          <div
            key={alert.frame}
            className={`
              ${styles.bg} ${styles.border} border-l-4 rounded-r-lg p-3
              ${styles.pulse ? 'animate-pulse' : ''}
              transition-all duration-300
            `}
          >
            {/* Header */}
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <span className="text-lg">{getAlertTypeIcon(alert.type)}</span>
                <div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-semibold ${getTeamColor(alert.team)}`}>
                      {alert.team === 'home' ? homeTeam : awayTeam}
                    </span>
                    <span className="text-xs text-slate-400">{alert.time_str}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      alert.priority_name === 'CRITICAL' ? 'bg-red-600 text-white' :
                      alert.priority_name === 'HIGH' ? 'bg-orange-600 text-white' :
                      alert.priority_name === 'MEDIUM' ? 'bg-yellow-600 text-black' :
                      'bg-slate-600 text-white'
                    }`}>
                      {alert.priority_name}
                    </span>
                  </div>
                </div>
              </div>

              {/* Dismiss button */}
              <button
                onClick={() => handleDismiss(alert.frame)}
                className="text-slate-400 hover:text-white transition-colors p-1"
                title="Dismiss"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Message */}
            <div className="mt-2">
              <p className="text-white text-sm font-medium">{alert.message}</p>
            </div>

            {/* Recommendation */}
            <div className="mt-2 bg-black/20 rounded p-2">
              <p className="text-xs text-slate-300">
                <span className="text-yellow-400 font-medium">Action: </span>
                {alert.recommendation}
              </p>
            </div>

            {/* Players involved */}
            {alert.involved_players && alert.involved_players.length > 0 && (
              <div className="mt-2 flex items-center gap-1">
                <span className="text-xs text-slate-400">Players:</span>
                {alert.involved_players.map((jersey, idx) => (
                  <span
                    key={idx}
                    className={`
                      w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-medium
                      ${alert.team === 'home' ? 'bg-blue-600' : 'bg-red-600'}
                    `}
                  >
                    {jersey}
                  </span>
                ))}
              </div>
            )}

            {/* Expiry progress bar */}
            <div className="mt-2 h-1 bg-black/30 rounded-full overflow-hidden">
              <div
                className="h-full bg-white/30 transition-all duration-100"
                style={{
                  width: `${Math.max(0, ((alert.expires_at_frame - currentFrame) / (alert.expires_at_frame - alert.frame)) * 100)}%`
                }}
              />
            </div>
          </div>
        );
      })}

      {/* Alert count indicator */}
      {alerts.length > maxVisible && (
        <div className="text-center text-xs text-slate-400">
          +{alerts.length - maxVisible} more alerts
        </div>
      )}
    </div>
  );
}

// Compact version for side panel
export function TacticalAlertsFeed({
  alerts,
  currentFrame,
  maxVisible = 10
}: {
  alerts: TacticalAlert[];
  currentFrame: number;
  maxVisible?: number;
}) {
  const recentAlerts = alerts
    .sort((a, b) => b.timestamp_ms - a.timestamp_ms)
    .slice(0, maxVisible);

  if (recentAlerts.length === 0) {
    return (
      <div className="text-center py-4 text-slate-400 text-sm">
        No tactical alerts yet
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {recentAlerts.map((alert, idx) => (
        <div
          key={alert.frame}
          className={`
            px-2 py-1.5 rounded text-xs
            ${alert.priority_name === 'CRITICAL' ? 'bg-red-900/50 border-l-2 border-red-500' :
              alert.priority_name === 'HIGH' ? 'bg-orange-900/40 border-l-2 border-orange-500' :
              'bg-slate-700/50'}
          `}
        >
          <div className="flex items-center justify-between">
            <span className={alert.team === 'home' ? 'text-blue-400' : 'text-red-400'}>
              {alert.time_str}
            </span>
            <span className="text-slate-400 text-[10px]">{alert.type.replace(/_/g, ' ')}</span>
          </div>
          <p className="text-slate-200 mt-0.5 line-clamp-2">{alert.message}</p>
        </div>
      ))}
    </div>
  );
}

export default TacticalAlerts;
