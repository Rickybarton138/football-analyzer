import { useEffect, useState } from 'react';
import { useCoachingStore } from '../../stores/coachingStore';
import { api } from '../../lib/api';
import { Card, CardTitle } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Spinner } from '../../components/ui/Spinner';
import { EmptyState } from '../../components/ui/EmptyState';
import { cn } from '../../lib/utils';
import { Shield, AlertTriangle, Clock, BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { TacticalEvent } from '../../types/analytics';

export default function TacticalPage() {
  const { tacticalAlerts, alertsLoading, loadTacticalAlerts } = useCoachingStore();
  const [events, setEvents] = useState<TacticalEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(true);

  useEffect(() => {
    loadTacticalAlerts();
    api.get<{ events: TacticalEvent[] }>('/api/analytics/tactical-events')
      .then(d => setEvents(d.events || []))
      .catch(() => {})
      .finally(() => setEventsLoading(false));
  }, []);

  if (alertsLoading && eventsLoading) return <Spinner label="Loading tactical data..." className="py-20" />;

  // Alert type breakdown for chart
  const alertTypes: Record<string, number> = {};
  tacticalAlerts.forEach(a => {
    alertTypes[a.alert_type] = (alertTypes[a.alert_type] || 0) + 1;
  });
  const typeChartData = Object.entries(alertTypes).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count);

  // Severity breakdown
  const sevCounts = { high: 0, medium: 0, low: 0 };
  tacticalAlerts.forEach(a => {
    if (a.severity in sevCounts) sevCounts[a.severity as keyof typeof sevCounts]++;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-pitch-light rounded-card p-6 border border-pitch/20">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-pitch-deep to-pitch rounded-card flex items-center justify-center">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-text-primary">Tactical Analysis</h2>
            <p className="text-text-muted text-sm">{tacticalAlerts.length} alerts &middot; {events.length} events</p>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="text-center">
          <div className="text-2xl font-bold text-text-primary">{tacticalAlerts.length}</div>
          <div className="text-text-muted text-xs">Total Alerts</div>
        </Card>
        <Card className="text-center border-red-200">
          <div className="text-2xl font-bold text-red-600">{sevCounts.high}</div>
          <div className="text-text-muted text-xs">High Severity</div>
        </Card>
        <Card className="text-center border-amber-200">
          <div className="text-2xl font-bold text-amber-600">{sevCounts.medium}</div>
          <div className="text-text-muted text-xs">Medium</div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl font-bold text-text-muted">{sevCounts.low}</div>
          <div className="text-text-muted text-xs">Low</div>
        </Card>
      </div>

      {/* Alert Type Breakdown Chart */}
      {typeChartData.length > 0 && (
        <Card>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-pitch" /> Alert Type Breakdown
          </CardTitle>
          <div className="h-64 mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={typeChartData} layout="vertical">
                <XAxis type="number" stroke="#888888" fontSize={12} />
                <YAxis type="category" dataKey="name" stroke="#888888" fontSize={11} width={120} />
                <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #E0E0E0', borderRadius: 8 }} />
                <Bar dataKey="count" fill="#43A047" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Alert Feed */}
      <div>
        <h3 className="text-lg font-semibold text-text-primary mb-4">Alert Feed</h3>
        {tacticalAlerts.length > 0 ? (
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {tacticalAlerts.map((alert, i) => (
              <Card key={i}>
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className={cn('w-8 h-8 rounded-lg flex items-center justify-center mt-0.5',
                      alert.severity === 'high' ? 'bg-red-50' : alert.severity === 'medium' ? 'bg-amber-50' : 'bg-surface-alt'
                    )}>
                      <AlertTriangle className={cn('w-4 h-4',
                        alert.severity === 'high' ? 'text-red-600' : alert.severity === 'medium' ? 'text-amber-600' : 'text-text-muted'
                      )} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={alert.severity === 'high' ? 'critical' : alert.severity === 'medium' ? 'medium' : 'low'}>
                          {alert.severity}
                        </Badge>
                        <Badge variant="default">{alert.alert_type}</Badge>
                        <span className="text-text-muted text-xs capitalize">{alert.team}</span>
                      </div>
                      <p className="text-text-secondary text-sm">{alert.description}</p>
                    </div>
                  </div>
                  <span className="text-text-muted text-xs flex items-center gap-1 flex-shrink-0">
                    <Clock className="w-3 h-3" /> {alert.minute}'
                  </span>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <EmptyState title="No Tactical Alerts" description="Alerts will appear when match analysis detects tactical patterns" />
        )}
      </div>

      {/* Events Timeline */}
      {events.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-4">Events Timeline</h3>
          <div className="space-y-2">
            {events.map((ev, i) => (
              <div key={i} className="flex items-center gap-4 p-3 bg-surface-alt rounded-btn border border-border">
                <span className="text-text-muted text-xs font-mono w-12">{ev.minute}'</span>
                <Badge variant="default">{ev.event_type}</Badge>
                <span className="text-text-secondary text-sm flex-1">{ev.description}</span>
                <span className="text-text-muted text-xs capitalize">{ev.team}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
