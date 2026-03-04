import { useEffect, useState } from 'react';
import { useMatchStore } from '../../stores/matchStore';
import { api } from '../../lib/api';
import { Card, CardTitle } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { cn } from '../../lib/utils';
import { BarChart3, GitBranch, Crosshair } from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import type { PassStats, FormationStats, XGData } from '../../types/analytics';

export default function AnalyticsPage() {
  const { analysis } = useMatchStore();
  const [passStats, setPassStats] = useState<PassStats | null>(null);
  const [formationStats, setFormationStats] = useState<FormationStats | null>(null);
  const [xgData, setXgData] = useState<XGData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.get<PassStats>('/api/analytics/passes').catch(() => null),
      api.get<FormationStats>('/api/analytics/formations').catch(() => null),
    ]).then(([passes, formations]) => {
      setPassStats(passes);
      setFormationStats(formations);
      // xG from main analysis data
      if (analysis?.xg_data) setXgData(analysis.xg_data as unknown as XGData);
      setLoading(false);
    });
  }, []);

  if (loading) return <Spinner label="Loading analytics..." className="py-20" />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 rounded-2xl p-6 border border-emerald-500/20">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Analytics</h2>
            <p className="text-slate-400 text-sm">Passes, formations, and expected goals</p>
          </div>
        </div>
      </div>

      {/* Pass Accuracy Comparison */}
      {passStats && (
        <Card>
          <CardTitle>Pass Accuracy</CardTitle>
          <div className="grid grid-cols-2 gap-6 mt-4">
            {(['home', 'away'] as const).map(side => {
              const s = passStats[side];
              if (!s) return null;
              return (
                <div key={side}>
                  <h4 className={cn('font-semibold mb-3', side === 'home' ? 'text-blue-400' : 'text-red-400')}>
                    {side === 'home' ? 'Home' : 'Away'}
                  </h4>
                  <div className="space-y-2">
                    <StatBar label="Total Passes" value={s.total} max={Math.max(passStats.home.total, passStats.away.total)} color={side === 'home' ? 'bg-blue-500' : 'bg-red-500'} />
                    <StatBar label="Completed" value={s.completed} max={Math.max(passStats.home.total, passStats.away.total)} color={side === 'home' ? 'bg-blue-500' : 'bg-red-500'} />
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Accuracy</span>
                      <span className="text-white font-medium">{s.accuracy}%</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Pass Direction */}
      {passStats && (
        <Card>
          <CardTitle>Pass Direction</CardTitle>
          <div className="grid grid-cols-2 gap-6 mt-4">
            {(['home', 'away'] as const).map(side => {
              const s = passStats[side];
              if (!s) return null;
              const dirData = [
                { name: 'Forward', value: s.forward, fill: '#10B981' },
                { name: 'Sideways', value: s.sideways, fill: '#06B6D4' },
                { name: 'Backward', value: s.backward, fill: '#64748b' },
              ];
              return (
                <div key={side}>
                  <h4 className={cn('font-semibold mb-3 text-center', side === 'home' ? 'text-blue-400' : 'text-red-400')}>
                    {side === 'home' ? 'Home' : 'Away'}
                  </h4>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={dirData} dataKey="value" nameKey="name" innerRadius={40} outerRadius={70} paddingAngle={2}>
                          {dirData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                        </Pie>
                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Formations */}
      {formationStats && (
        <Card>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-emerald-400" /> Formations
          </CardTitle>
          <div className="grid grid-cols-2 gap-6 mt-4">
            {(['home', 'away'] as const).map(side => {
              const fs = formationStats[side];
              if (!fs) return null;
              return (
                <div key={side}>
                  <h4 className={cn('font-semibold mb-2', side === 'home' ? 'text-blue-400' : 'text-red-400')}>
                    {side === 'home' ? 'Home' : 'Away'}
                  </h4>
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-2xl font-bold text-white mb-2">{fs.primary}</div>
                    <div className="space-y-1">
                      {Object.entries(fs.formations).map(([formation, count]) => (
                        <div key={formation} className="flex justify-between text-sm">
                          <span className="text-slate-400">{formation}</span>
                          <span className="text-slate-300">{count as number} frames</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* xG Shot Map */}
      {xgData && (
        <Card>
          <CardTitle className="flex items-center gap-2">
            <Crosshair className="w-5 h-5 text-emerald-400" /> Expected Goals (xG)
          </CardTitle>
          <div className="grid grid-cols-2 gap-6 mt-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400">{xgData.home_xg?.toFixed(2) || '0.00'}</div>
              <div className="text-slate-400 text-sm">Home xG</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-red-400">{xgData.away_xg?.toFixed(2) || '0.00'}</div>
              <div className="text-slate-400 text-sm">Away xG</div>
            </div>
          </div>
          {xgData.shots && xgData.shots.length > 0 && (
            <div className="mt-4">
              {/* Simple pitch representation */}
              <div className="relative w-full aspect-[2/1] bg-emerald-900/30 rounded-lg border border-emerald-700/30 overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center text-slate-600 text-sm">Pitch</div>
                {xgData.shots.map((shot, i) => (
                  <div key={i}
                    className={cn('absolute w-4 h-4 rounded-full border-2 transform -translate-x-1/2 -translate-y-1/2',
                      shot.team === 'home' ? 'border-blue-400' : 'border-red-400',
                      shot.is_goal ? 'bg-emerald-500' : shot.on_target ? 'bg-amber-500/50' : 'bg-slate-600/50'
                    )}
                    style={{
                      left: `${(shot.position[0] / 1920) * 100}%`,
                      top: `${(shot.position[1] / 1080) * 100}%`,
                      width: `${Math.max(12, shot.xg * 40)}px`,
                      height: `${Math.max(12, shot.xg * 40)}px`,
                    }}
                    title={`xG: ${shot.xg.toFixed(2)} | ${shot.team} | ${shot.is_goal ? 'GOAL' : shot.on_target ? 'On target' : 'Off target'}`}
                  />
                ))}
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function StatBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-white">{value}</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
