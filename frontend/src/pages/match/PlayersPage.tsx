import { useState, useEffect } from 'react';
import { api } from '../../lib/api';
import { Card, CardTitle } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { EmptyState } from '../../components/ui/EmptyState';
import { cn } from '../../lib/utils';
import { Users, User, Target, Shield, Zap } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from 'recharts';
import type { PlayerStats } from '../../types/analytics';

export default function PlayersPage() {
  const [players, setPlayers] = useState<{ name: string; clips_analyzed: number; play_time_seconds: number; ball_touches: number }[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    api.get<{ players: typeof players }>('/api/players/list')
      .then(d => setPlayers(d.players || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const loadPlayer = async (name: string) => {
    setDetailLoading(true);
    try {
      const data = await api.get<PlayerStats>(`/api/players/${encodeURIComponent(name)}/stats`);
      setSelectedPlayer(data);
    } catch {
      setSelectedPlayer(null);
    }
    setDetailLoading(false);
  };

  if (loading) return <Spinner label="Loading players..." className="py-20" />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-2xl p-6 border border-emerald-500/20">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <Users className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Player Analysis</h2>
            <p className="text-slate-400 text-sm">{players.length} players detected</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Player List */}
        <div className="md:col-span-1">
          <Card>
            <CardTitle>Players</CardTitle>
            {players.length > 0 ? (
              <div className="space-y-2 mt-3 max-h-[500px] overflow-y-auto">
                {players.map((p, i) => (
                  <button key={i} onClick={() => loadPlayer(p.name)}
                    className={cn('w-full text-left p-3 rounded-lg transition-all flex items-center gap-3',
                      selectedPlayer?.player_name === p.name
                        ? 'bg-emerald-500/10 border border-emerald-500/30'
                        : 'hover:bg-slate-800 border border-transparent'
                    )}>
                    <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center">
                      <User className="w-4 h-4 text-slate-300" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-white text-sm font-medium truncate">{p.name}</div>
                      <div className="text-slate-500 text-xs">{p.clips_analyzed} clips &middot; {p.ball_touches} touches</div>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <EmptyState title="No Players" description="Player data will be available after analysis" />
            )}
          </Card>
        </div>

        {/* Player Detail */}
        <div className="md:col-span-2">
          {detailLoading ? (
            <Spinner label="Loading player stats..." className="py-20" />
          ) : selectedPlayer ? (
            <div className="space-y-4">
              <Card>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-full flex items-center justify-center">
                    <User className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">{selectedPlayer.player_name}</h3>
                    <p className="text-slate-400 text-sm">{selectedPlayer.total_clips} clips analyzed</p>
                  </div>
                </div>

                {/* Radar Chart */}
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={[
                      { stat: 'Passing', value: selectedPlayer.attacking.pass_accuracy },
                      { stat: 'Shooting', value: selectedPlayer.attacking.shot_accuracy },
                      { stat: 'Tackles', value: selectedPlayer.defensive.tackle_success_rate },
                      { stat: 'Touches', value: Math.min(100, selectedPlayer.attacking.ball_touches * 5) },
                      { stat: 'Distance', value: Math.min(100, selectedPlayer.physical.distance_covered_meters_estimate / 10) },
                      { stat: 'Sprints', value: Math.min(100, selectedPlayer.physical.sprints * 10) },
                    ]}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="stat" stroke="#94a3b8" fontSize={12} />
                      <Radar dataKey="value" stroke="#10B981" fill="#10B981" fillOpacity={0.2} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <h4 className="text-emerald-400 text-sm font-semibold flex items-center gap-1 mb-3"><Target className="w-4 h-4" /> Attacking</h4>
                  <div className="space-y-2 text-sm">
                    <StatRow label="Touches" value={selectedPlayer.attacking.ball_touches} />
                    <StatRow label="Passes" value={`${selectedPlayer.attacking.passes_completed}/${selectedPlayer.attacking.passes_attempted}`} />
                    <StatRow label="Accuracy" value={`${selectedPlayer.attacking.pass_accuracy}%`} />
                    <StatRow label="Shots" value={`${selectedPlayer.attacking.shots_on_target}/${selectedPlayer.attacking.shots}`} />
                  </div>
                </Card>
                <Card>
                  <h4 className="text-blue-400 text-sm font-semibold flex items-center gap-1 mb-3"><Shield className="w-4 h-4" /> Defensive</h4>
                  <div className="space-y-2 text-sm">
                    <StatRow label="Tackles Won" value={`${selectedPlayer.defensive.tackles_won}/${selectedPlayer.defensive.tackles_attempted}`} />
                    <StatRow label="Success Rate" value={`${selectedPlayer.defensive.tackle_success_rate}%`} />
                    <StatRow label="Interceptions" value={selectedPlayer.defensive.interceptions} />
                    <StatRow label="Headers" value={selectedPlayer.defensive.headers} />
                  </div>
                </Card>
                <Card>
                  <h4 className="text-amber-400 text-sm font-semibold flex items-center gap-1 mb-3"><Zap className="w-4 h-4" /> Physical</h4>
                  <div className="space-y-2 text-sm">
                    <StatRow label="Distance" value={`${selectedPlayer.physical.distance_covered_meters_estimate.toFixed(0)}m`} />
                    <StatRow label="Sprints" value={selectedPlayer.physical.sprints} />
                  </div>
                </Card>
              </div>
            </div>
          ) : (
            <EmptyState
              icon={<User className="w-8 h-8 text-slate-500" />}
              title="Select a Player"
              description="Click on a player to view their detailed statistics"
            />
          )}
        </div>
      </div>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="text-white font-medium">{value}</span>
    </div>
  );
}
