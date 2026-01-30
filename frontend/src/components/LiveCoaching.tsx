import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Play, Pause, Radio, Volume2, VolumeX, Settings,
  AlertTriangle, Zap, Lightbulb, X, Clock, Target,
  TrendingUp, Users, Activity, RefreshCw
} from 'lucide-react';
import clsx from 'clsx';
import type { TacticalAlert, AlertPriority, DetectedPlayer, Position } from '../types';

// Types for live coaching
interface LiveStats {
  possession: { home: number; away: number };
  score: { home: number; away: number };
  match_time_ms: number;
  period: number;
  total_frames_analyzed: number;
}

interface LiveStreamStatus {
  is_active: boolean;
  session_id: string | null;
  stream: {
    status: string;
    stream_url: string | null;
    stream_type: string | null;
    is_running: boolean;
    metrics: {
      actual_fps: number;
      latency_ms: number;
      frames_processed: number;
      frames_dropped: number;
      reconnect_count: number;
      uptime_seconds: number;
    };
  };
  stats: LiveStats;
}

interface LiveAlert {
  alert_id: string;
  priority: AlertPriority;
  category: string;
  timestamp_ms: number;
  title: string;
  message: string;
  action?: string;
  highlight_players: number[];
  highlight_zone?: string;
  duration_seconds: number;
  expires_at_ms: number;
  play_sound: boolean;
  sound_type: string;
}

interface TeamStatistics {
  possession_pct: number;
  possession_defensive_pct: number;
  possession_middle_pct: number;
  possession_attacking_pct: number;
  passes_total: number;
  passes_successful: number;
  passes_failed: number;
  pass_accuracy_pct: number;
  passes_forward: number;
  passes_sideways: number;
  passes_backward: number;
  passes_defensive_third: number;
  passes_middle_third: number;
  passes_attacking_third: number;
  longest_pass_sequence: number;
  shots_total: number;
  shots_on_target: number;
  shots_off_target: number;
  shot_accuracy_pct: number;
  goals: number;
  tackles: number;
  interceptions: number;
  headers: number;
  corners: number;
  free_kicks: number;
  throw_ins: number;
  goal_kicks: number;
}

interface FullTeamStats {
  home: {
    full_match: TeamStatistics;
    first_half: TeamStatistics;
    second_half: TeamStatistics;
  };
  away: {
    full_match: TeamStatistics;
    first_half: TeamStatistics;
    second_half: TeamStatistics;
  };
  events_count: number;
  current_period: number;
  team_colors_detected: boolean;
  home_team_color: number[] | null;
  away_team_color: number[] | null;
}

interface LiveCoachingProps {
  onClose?: () => void;
}

// Audio alert sounds
const ALERT_SOUNDS = {
  immediate: '/sounds/alert-urgent.mp3',
  tactical: '/sounds/alert-tactical.mp3',
  strategic: '/sounds/alert-info.mp3',
};

export function LiveCoaching({ onClose }: LiveCoachingProps) {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [streamUrl, setStreamUrl] = useState('');
  const [streamType, setStreamType] = useState<'rtsp' | 'hls'>('hls');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStreamStatus | null>(null);
  const [alerts, setAlerts] = useState<LiveAlert[]>([]);
  const [teamStats, setTeamStats] = useState<FullTeamStats | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [players, setPlayers] = useState<DetectedPlayer[]>([]);
  const [ballPosition, setBallPosition] = useState<Position | null>(null);
  const [homeScore, setHomeScore] = useState(0);
  const [awayScore, setAwayScore] = useState(0);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  // Play alert sound
  const playAlertSound = useCallback((priority: AlertPriority) => {
    if (!soundEnabled) return;

    // Create audio element if not exists
    if (!audioRef.current) {
      audioRef.current = new Audio();
    }

    // Use Web Audio API for alert beep since we may not have audio files
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // Different frequencies for different priorities
      const frequencies: Record<AlertPriority, number> = {
        immediate: 880,  // High A
        tactical: 659,   // E
        strategic: 440,  // A
      };

      oscillator.frequency.value = frequencies[priority];
      oscillator.type = priority === 'immediate' ? 'square' : 'sine';

      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.3);

      // For immediate alerts, play a second beep
      if (priority === 'immediate') {
        setTimeout(() => {
          const osc2 = audioContext.createOscillator();
          const gain2 = audioContext.createGain();
          osc2.connect(gain2);
          gain2.connect(audioContext.destination);
          osc2.frequency.value = 880;
          osc2.type = 'square';
          gain2.gain.setValueAtTime(0.3, audioContext.currentTime);
          gain2.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
          osc2.start(audioContext.currentTime);
          osc2.stop(audioContext.currentTime + 0.2);
        }, 150);
      }
    } catch (e) {
      console.log('Audio not available:', e);
    }
  }, [soundEnabled]);

  // Fetch live status
  const fetchLiveStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/live/status');
      if (response.ok) {
        const data = await response.json();
        setLiveStatus(data);
        setIsLive(data.is_active);
        if (data.session_id) {
          setSessionId(data.session_id);
        }
      }
    } catch (e) {
      console.error('Failed to fetch live status:', e);
    }
  }, []);

  // Fetch team stats
  const fetchTeamStats = useCallback(async () => {
    try {
      const response = await fetch('/api/match-events/team-stats');
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          setTeamStats(data.statistics);
        }
      }
    } catch (e) {
      console.error('Failed to fetch team stats:', e);
    }
  }, []);

  // Start live stream
  const startLiveStream = async () => {
    if (!streamUrl) {
      setError('Please enter a stream URL');
      return;
    }

    setError(null);

    try {
      const response = await fetch('/api/live/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stream_url: streamUrl,
          stream_type: streamType,
          target_fps: 2.0,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        setIsLive(true);
        connectWebSocket(data.session_id);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to start stream');
      }
    } catch (e) {
      setError('Failed to connect to server');
      console.error('Start stream error:', e);
    }
  };

  // Stop live stream
  const stopLiveStream = async () => {
    try {
      await fetch('/api/live/stop', { method: 'POST' });
      setIsLive(false);
      setSessionId(null);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    } catch (e) {
      console.error('Stop stream error:', e);
    }
  };

  // Connect to WebSocket for live updates
  const connectWebSocket = (sid: string) => {
    const wsUrl = `ws://${window.location.hostname}:8000/ws/live-coaching/${sid}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected for live coaching');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case 'frame_update':
              setPlayers(data.players || []);
              if (data.ball) {
                setBallPosition(data.ball);
              }
              break;

            case 'alert':
              const alert = data.alert as LiveAlert;
              setAlerts(prev => [alert, ...prev].slice(0, 10));
              playAlertSound(alert.priority);
              break;

            case 'stats_update':
              if (data.stats) {
                // Update quick stats
                setHomeScore(data.stats.home_score || 0);
                setAwayScore(data.stats.away_score || 0);
              }
              break;

            case 'event':
              // Handle match events
              console.log('Match event:', data.event);
              break;
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // Attempt reconnect if still live
        if (isLive && sessionId) {
          setTimeout(() => connectWebSocket(sessionId), 3000);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
    }
  };

  // Update score manually
  const updateScore = async (team: 'home' | 'away', increment: number) => {
    const newHome = team === 'home' ? homeScore + increment : homeScore;
    const newAway = team === 'away' ? awayScore + increment : awayScore;

    try {
      await fetch(`/api/live/score?home=${newHome}&away=${newAway}`, {
        method: 'POST',
      });
      setHomeScore(newHome);
      setAwayScore(newAway);
    } catch (e) {
      console.error('Failed to update score:', e);
    }
  };

  // Dismiss alert
  const dismissAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(a => a.alert_id !== alertId));

    // Send dismiss to server if connected
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'dismiss_alert',
        alert_id: alertId,
      }));
    }
  };

  // Poll for status updates
  useEffect(() => {
    fetchLiveStatus();
    fetchTeamStats();

    pollIntervalRef.current = window.setInterval(() => {
      fetchLiveStatus();
      fetchTeamStats();
    }, 5000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [fetchLiveStatus, fetchTeamStats]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Format time
  const formatTime = (ms: number): string => {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="h-full flex flex-col bg-[#0a0f1a]">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <Radio className={clsx(
            'w-5 h-5',
            isLive ? 'text-red-500 animate-pulse' : 'text-slate-400'
          )} />
          <h2 className="text-lg font-semibold">Live Coaching</h2>
          {isLive && (
            <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">
              LIVE
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="p-2 rounded hover:bg-slate-700"
            title={soundEnabled ? 'Mute alerts' : 'Unmute alerts'}
          >
            {soundEnabled ? (
              <Volume2 className="w-5 h-5 text-slate-400" />
            ) : (
              <VolumeX className="w-5 h-5 text-slate-400" />
            )}
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded hover:bg-slate-700"
          >
            <Settings className="w-5 h-5 text-slate-400" />
          </button>

          {onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded hover:bg-slate-700"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden grid grid-cols-12 gap-4 p-4">
        {/* Left Panel - Stream Controls & Stats */}
        <div className="col-span-3 space-y-4 overflow-y-auto">
          {/* Stream Controls */}
          <div className="bg-[#111827] rounded-lg p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Stream Setup</h3>

            {!isLive ? (
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Stream URL</label>
                  <input
                    type="text"
                    value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    placeholder="rtsp:// or https://.m3u8"
                    className="w-full px-3 py-2 bg-slate-800 rounded border border-slate-600 text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">Stream Type</label>
                  <select
                    value={streamType}
                    onChange={(e) => setStreamType(e.target.value as 'rtsp' | 'hls')}
                    className="w-full px-3 py-2 bg-slate-800 rounded border border-slate-600 text-sm focus:outline-none focus:border-blue-500"
                  >
                    <option value="hls">HLS (VEO Live)</option>
                    <option value="rtsp">RTSP (Direct)</option>
                  </select>
                </div>

                {error && (
                  <p className="text-xs text-red-400">{error}</p>
                )}

                <button
                  onClick={startLiveStream}
                  className="w-full py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Start Live Analysis
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Status</span>
                  <span className={clsx(
                    'flex items-center gap-1',
                    isConnected ? 'text-green-400' : 'text-amber-400'
                  )}>
                    <span className={clsx(
                      'w-2 h-2 rounded-full',
                      isConnected ? 'bg-green-400' : 'bg-amber-400'
                    )} />
                    {isConnected ? 'Connected' : 'Connecting...'}
                  </span>
                </div>

                {liveStatus?.stream.metrics && (
                  <>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-400">FPS</span>
                      <span>{liveStatus.stream.metrics.actual_fps.toFixed(1)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-400">Latency</span>
                      <span>{liveStatus.stream.metrics.latency_ms.toFixed(0)}ms</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-400">Frames</span>
                      <span>{liveStatus.stream.metrics.frames_processed}</span>
                    </div>
                  </>
                )}

                <button
                  onClick={stopLiveStream}
                  className="w-full py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium flex items-center justify-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  Stop Analysis
                </button>
              </div>
            )}
          </div>

          {/* Score Control */}
          <div className="bg-[#111827] rounded-lg p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Score</h3>
            <div className="flex items-center justify-between">
              <div className="text-center">
                <button
                  onClick={() => updateScore('home', 1)}
                  className="w-8 h-8 bg-blue-600/20 text-blue-400 rounded hover:bg-blue-600/30 text-lg font-bold"
                >
                  +
                </button>
                <div className="text-3xl font-bold text-blue-400 my-2">{homeScore}</div>
                <button
                  onClick={() => updateScore('home', -1)}
                  disabled={homeScore === 0}
                  className="w-8 h-8 bg-blue-600/20 text-blue-400 rounded hover:bg-blue-600/30 text-lg font-bold disabled:opacity-50"
                >
                  -
                </button>
                <div className="text-xs text-slate-400 mt-1">HOME</div>
              </div>

              <div className="text-2xl text-slate-500">-</div>

              <div className="text-center">
                <button
                  onClick={() => updateScore('away', 1)}
                  className="w-8 h-8 bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 text-lg font-bold"
                >
                  +
                </button>
                <div className="text-3xl font-bold text-red-400 my-2">{awayScore}</div>
                <button
                  onClick={() => updateScore('away', -1)}
                  disabled={awayScore === 0}
                  className="w-8 h-8 bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 text-lg font-bold disabled:opacity-50"
                >
                  -
                </button>
                <div className="text-xs text-slate-400 mt-1">AWAY</div>
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          {teamStats && (
            <div className="bg-[#111827] rounded-lg p-4">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Match Stats</h3>
              <div className="space-y-2">
                <StatRow
                  label="Possession"
                  homeValue={teamStats.home.full_match.possession_pct}
                  awayValue={teamStats.away.full_match.possession_pct}
                  format={(v) => `${v.toFixed(0)}%`}
                />
                <StatRow
                  label="Pass Accuracy"
                  homeValue={teamStats.home.full_match.pass_accuracy_pct}
                  awayValue={teamStats.away.full_match.pass_accuracy_pct}
                  format={(v) => `${v.toFixed(0)}%`}
                />
                <StatRow
                  label="Passes"
                  homeValue={teamStats.home.full_match.passes_total}
                  awayValue={teamStats.away.full_match.passes_total}
                />
                <StatRow
                  label="Shots"
                  homeValue={teamStats.home.full_match.shots_total}
                  awayValue={teamStats.away.full_match.shots_total}
                />
                <StatRow
                  label="On Target"
                  homeValue={teamStats.home.full_match.shots_on_target}
                  awayValue={teamStats.away.full_match.shots_on_target}
                />
              </div>
            </div>
          )}
        </div>

        {/* Center Panel - Tactical Board & Video */}
        <div className="col-span-6 space-y-4">
          {/* Mini Tactical Board */}
          <div className="bg-[#111827] rounded-lg p-4 h-[400px]">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-slate-300">Live Tactical View</h3>
              <button
                onClick={fetchTeamStats}
                className="p-1 rounded hover:bg-slate-700"
                title="Refresh"
              >
                <RefreshCw className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            <MiniTacticalBoard
              players={players}
              ballPosition={ballPosition}
              teamColors={teamStats ? {
                home: teamStats.home_team_color,
                away: teamStats.away_team_color,
              } : undefined}
            />
          </div>

          {/* Pass Direction Stats */}
          {teamStats && (
            <div className="bg-[#111827] rounded-lg p-4">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Pass Analysis</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-xs text-blue-400 mb-2">Home Team</h4>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Forward</span>
                      <span>{teamStats.home.full_match.passes_forward}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sideways</span>
                      <span>{teamStats.home.full_match.passes_sideways}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Backward</span>
                      <span>{teamStats.home.full_match.passes_backward}</span>
                    </div>
                    <div className="flex justify-between text-green-400">
                      <span>Longest Sequence</span>
                      <span>{teamStats.home.full_match.longest_pass_sequence}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="text-xs text-red-400 mb-2">Away Team</h4>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Forward</span>
                      <span>{teamStats.away.full_match.passes_forward}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Sideways</span>
                      <span>{teamStats.away.full_match.passes_sideways}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Backward</span>
                      <span>{teamStats.away.full_match.passes_backward}</span>
                    </div>
                    <div className="flex justify-between text-green-400">
                      <span>Longest Sequence</span>
                      <span>{teamStats.away.full_match.longest_pass_sequence}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Alerts */}
        <div className="col-span-3 space-y-4 overflow-y-auto">
          {/* Alert Feed */}
          <div className="bg-[#111827] rounded-lg p-4 h-full">
            <h3 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              Coaching Alerts
            </h3>

            {alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-48 text-slate-400">
                <Activity className="w-8 h-8 mb-2 opacity-50" />
                <p className="text-sm">No active alerts</p>
                <p className="text-xs opacity-75">Alerts will appear here during live analysis</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {alerts.map((alert) => (
                  <AlertCard
                    key={alert.alert_id}
                    alert={alert}
                    onDismiss={() => dismissAlert(alert.alert_id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Stat row component
interface StatRowProps {
  label: string;
  homeValue: number;
  awayValue: number;
  format?: (v: number) => string;
}

function StatRow({ label, homeValue, awayValue, format = String }: StatRowProps) {
  const total = homeValue + awayValue || 1;
  const homePercent = (homeValue / total) * 100;

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-blue-400">{format(homeValue)}</span>
        <span className="text-slate-400">{label}</span>
        <span className="text-red-400">{format(awayValue)}</span>
      </div>
      <div className="flex h-1.5 rounded overflow-hidden bg-slate-700">
        <div
          className="bg-blue-500 transition-all duration-500"
          style={{ width: `${homePercent}%` }}
        />
        <div
          className="bg-red-500 transition-all duration-500"
          style={{ width: `${100 - homePercent}%` }}
        />
      </div>
    </div>
  );
}

// Alert card component
interface AlertCardProps {
  alert: LiveAlert;
  onDismiss: () => void;
}

function AlertCard({ alert, onDismiss }: AlertCardProps) {
  const priorityStyles: Record<AlertPriority, { bg: string; border: string; icon: string }> = {
    immediate: {
      bg: 'bg-red-900/50',
      border: 'border-red-500',
      icon: 'text-red-400',
    },
    tactical: {
      bg: 'bg-amber-900/50',
      border: 'border-amber-500',
      icon: 'text-amber-400',
    },
    strategic: {
      bg: 'bg-blue-900/50',
      border: 'border-blue-500',
      icon: 'text-blue-400',
    },
  };

  const style = priorityStyles[alert.priority];

  return (
    <div
      className={clsx(
        'relative rounded-lg p-3 border',
        style.bg,
        style.border,
        alert.priority === 'immediate' && 'animate-pulse'
      )}
    >
      <button
        onClick={onDismiss}
        className="absolute top-2 right-2 text-slate-400 hover:text-white"
      >
        <X className="w-4 h-4" />
      </button>

      <div className="flex items-start gap-2 pr-6">
        <div className={clsx('mt-0.5', style.icon)}>
          {alert.priority === 'immediate' && <Zap className="w-4 h-4" />}
          {alert.priority === 'tactical' && <AlertTriangle className="w-4 h-4" />}
          {alert.priority === 'strategic' && <Lightbulb className="w-4 h-4" />}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm">{alert.title}</p>
          <p className="text-xs text-slate-300 mt-1">{alert.message}</p>
          {alert.action && (
            <p className="text-xs mt-2 text-green-400">
              {alert.action}
            </p>
          )}
        </div>
      </div>

      {alert.highlight_players.length > 0 && (
        <div className="mt-2 flex items-center gap-1">
          <span className="text-xs text-slate-400">Players:</span>
          {alert.highlight_players.map((id) => (
            <span
              key={id}
              className="text-xs bg-slate-700 px-1.5 py-0.5 rounded"
            >
              #{id}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// Mini tactical board component
interface MiniTacticalBoardProps {
  players: DetectedPlayer[];
  ballPosition: Position | null;
  teamColors?: {
    home: number[] | null;
    away: number[] | null;
  };
}

function MiniTacticalBoard({ players, ballPosition, teamColors }: MiniTacticalBoardProps) {
  // Pitch dimensions in meters (standard 105x68)
  const pitchWidth = 105;
  const pitchHeight = 68;

  // SVG viewBox with padding
  const padding = 5;
  const viewBox = `${-padding} ${-padding} ${pitchWidth + padding * 2} ${pitchHeight + padding * 2}`;

  return (
    <svg viewBox={viewBox} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
      {/* Pitch background */}
      <rect
        x={0}
        y={0}
        width={pitchWidth}
        height={pitchHeight}
        fill="#1a472a"
        stroke="#2d5a3d"
        strokeWidth={0.5}
      />

      {/* Center line */}
      <line
        x1={pitchWidth / 2}
        y1={0}
        x2={pitchWidth / 2}
        y2={pitchHeight}
        stroke="#2d5a3d"
        strokeWidth={0.3}
      />

      {/* Center circle */}
      <circle
        cx={pitchWidth / 2}
        cy={pitchHeight / 2}
        r={9.15}
        fill="none"
        stroke="#2d5a3d"
        strokeWidth={0.3}
      />

      {/* Penalty areas */}
      <rect
        x={0}
        y={(pitchHeight - 40.3) / 2}
        width={16.5}
        height={40.3}
        fill="none"
        stroke="#2d5a3d"
        strokeWidth={0.3}
      />
      <rect
        x={pitchWidth - 16.5}
        y={(pitchHeight - 40.3) / 2}
        width={16.5}
        height={40.3}
        fill="none"
        stroke="#2d5a3d"
        strokeWidth={0.3}
      />

      {/* Goal areas */}
      <rect
        x={0}
        y={(pitchHeight - 18.3) / 2}
        width={5.5}
        height={18.3}
        fill="none"
        stroke="#2d5a3d"
        strokeWidth={0.3}
      />
      <rect
        x={pitchWidth - 5.5}
        y={(pitchHeight - 18.3) / 2}
        width={5.5}
        height={18.3}
        fill="none"
        stroke="#2d5a3d"
        strokeWidth={0.3}
      />

      {/* Goals */}
      <rect
        x={-2}
        y={(pitchHeight - 7.32) / 2}
        width={2}
        height={7.32}
        fill="none"
        stroke="#ffffff"
        strokeWidth={0.3}
      />
      <rect
        x={pitchWidth}
        y={(pitchHeight - 7.32) / 2}
        width={2}
        height={7.32}
        fill="none"
        stroke="#ffffff"
        strokeWidth={0.3}
      />

      {/* Thirds dividers */}
      <line
        x1={pitchWidth / 3}
        y1={0}
        x2={pitchWidth / 3}
        y2={pitchHeight}
        stroke="#2d5a3d"
        strokeWidth={0.2}
        strokeDasharray="2,2"
      />
      <line
        x1={(pitchWidth * 2) / 3}
        y1={0}
        x2={(pitchWidth * 2) / 3}
        y2={pitchHeight}
        stroke="#2d5a3d"
        strokeWidth={0.2}
        strokeDasharray="2,2"
      />

      {/* Players */}
      {players.map((player) => {
        if (!player.pitch_position) return null;

        // Scale position to pitch dimensions
        const x = (player.pitch_position.x / 105) * pitchWidth;
        const y = (player.pitch_position.y / 68) * pitchHeight;

        // Determine color based on team
        let color = '#9ca3af'; // gray for unknown
        if (player.team === 'home') {
          color = teamColors?.home
            ? `rgb(${teamColors.home.join(',')})`
            : '#3b82f6'; // blue
        } else if (player.team === 'away') {
          color = teamColors?.away
            ? `rgb(${teamColors.away.join(',')})`
            : '#ef4444'; // red
        }

        // Goalkeeper gets different treatment
        if (player.is_goalkeeper) {
          color = player.team === 'home' ? '#22c55e' : '#f97316';
        }

        return (
          <g key={player.track_id}>
            <circle
              cx={x}
              cy={y}
              r={1.5}
              fill={color}
              stroke="#ffffff"
              strokeWidth={0.3}
            />
            <text
              x={x}
              y={y + 0.5}
              textAnchor="middle"
              fontSize={1.8}
              fill="#ffffff"
              fontWeight="bold"
            >
              {player.track_id}
            </text>
          </g>
        );
      })}

      {/* Ball */}
      {ballPosition && (
        <circle
          cx={(ballPosition.x / 105) * pitchWidth}
          cy={(ballPosition.y / 68) * pitchHeight}
          r={1}
          fill="#ffffff"
          stroke="#000000"
          strokeWidth={0.2}
        />
      )}
    </svg>
  );
}

export default LiveCoaching;
