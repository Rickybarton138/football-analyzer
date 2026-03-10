import { useEffect } from 'react';
import { useCoachingStore } from '../../stores/coachingStore';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { Spinner } from '../../components/ui/Spinner';
import { EmptyState } from '../../components/ui/EmptyState';
import { cn } from '../../lib/utils';
import { Target, Clock, Printer } from 'lucide-react';
import type { PriorityArea, SessionPlan } from '../../types/coaching';

export default function TrainingPage() {
  const { trainingFocus, trainingLoading, trainingError, loadTrainingFocus } = useCoachingStore();

  useEffect(() => { loadTrainingFocus(); }, []);

  if (trainingLoading) return <Spinner label="Generating training recommendations..." className="py-20" />;

  if (trainingError || !trainingFocus) {
    return (
      <EmptyState
        icon={<Target className="w-8 h-8 text-text-muted" />}
        title="Training Focus Unavailable"
        description={trainingError || 'No training data available for this match.'}
        action={<Button variant="secondary" size="sm" onClick={loadTrainingFocus}>Retry</Button>}
      />
    );
  }

  const { priority_areas, session_plan } = trainingFocus;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-pitch-light rounded-card p-6 border border-pitch/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-pitch-deep to-pitch rounded-card flex items-center justify-center">
              <Target className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-text-primary">Training Focus</h2>
              <p className="text-text-muted text-sm">AI-generated drills based on match weaknesses</p>
            </div>
          </div>
          <Button variant="secondary" size="sm" onClick={() => window.print()}>
            <Printer className="w-4 h-4" /> Print Plan
          </Button>
        </div>
      </div>

      {/* Priority Areas */}
      {priority_areas && priority_areas.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-4">Priority Areas ({priority_areas.length})</h3>
          <div className="grid gap-4">
            {priority_areas.map((area, i) => (
              <PriorityAreaCard key={i} area={area} index={i + 1} />
            ))}
          </div>
        </div>
      )}

      {/* Session Plan */}
      {session_plan && <SessionPlanView plan={session_plan} />}
    </div>
  );
}

function PriorityAreaCard({ area, index }: { area: PriorityArea; index: number }) {
  return (
    <Card className={cn(
      'border-l-4',
      area.severity === 'high' ? 'border-l-red-500' : 'border-l-amber-500'
    )}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={cn(
            'w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold',
            area.severity === 'high' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'
          )}>
            {index}
          </div>
          <div>
            <h4 className="text-text-primary font-semibold">{area.area}</h4>
            <p className="text-text-muted text-sm capitalize">{area.team} team</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={area.severity === 'high' ? 'critical' : 'medium'}>
            {area.severity === 'high' ? 'High Priority' : 'Medium'}
          </Badge>
          {area.metric && <span className="text-text-secondary text-sm font-mono">{area.metric}</span>}
        </div>
      </div>

      <div className="bg-surface-alt rounded-card p-4 mt-3">
        <div className="flex items-center gap-2 text-pitch text-sm font-semibold mb-2">
          <Target className="w-4 h-4" />
          Recommended Drill
        </div>
        <h5 className="text-text-primary font-medium mb-1">{area.drill}</h5>
        <p className="text-text-muted text-sm">{area.detail}</p>
        {area.duration_mins > 0 && (
          <div className="flex items-center gap-1 text-text-muted text-xs mt-2">
            <Clock className="w-3 h-3" /> {area.duration_mins} minutes
          </div>
        )}
      </div>
    </Card>
  );
}

function SessionPlanView({ plan }: { plan: SessionPlan }) {
  const blocks = [
    { label: 'Warm Up', data: plan.warm_up, color: 'from-blue-500 to-blue-600', icon: '1' },
    { label: 'Main Focus', data: plan.main_focus, color: 'from-emerald-500 to-emerald-600', icon: '2' },
    { label: 'Secondary Focus', data: plan.secondary_focus, color: 'from-cyan-500 to-cyan-600', icon: '3' },
    { label: 'Game', data: plan.game, color: 'from-purple-500 to-purple-600', icon: '4' },
    { label: 'Cool Down', data: plan.cool_down, color: 'from-slate-500 to-slate-600', icon: '5' },
  ];

  const totalMins = blocks.reduce((sum, b) => sum + ((b.data as any)?.duration_mins || 0), 0);

  return (
    <div className="print:break-before-page">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text-primary">Session Plan</h3>
        <span className="text-text-muted text-sm flex items-center gap-1">
          <Clock className="w-4 h-4" /> {totalMins} minutes total
        </span>
      </div>

      <div className="space-y-3">
        {blocks.map((block, i) => {
          const d = block.data as any;
          if (!d) return null;
          return (
            <div key={i} className="flex items-stretch gap-3">
              {/* Timeline connector */}
              <div className="flex flex-col items-center">
                <div className={cn('w-10 h-10 rounded-full bg-gradient-to-br flex items-center justify-center text-white font-bold text-sm', block.color)}>
                  {block.icon}
                </div>
                {i < blocks.length - 1 && <div className="w-0.5 flex-1 bg-border my-1" />}
              </div>

              {/* Content */}
              <Card className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-text-primary font-semibold">{block.label}</h4>
                  {d.duration_mins && (
                    <span className="text-text-muted text-xs flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {d.duration_mins} min
                    </span>
                  )}
                </div>
                {d.activity && <p className="text-text-secondary text-sm">{d.activity}</p>}
                {d.area && <p className="text-pitch text-sm font-medium">{d.area}</p>}
                {d.drill && <p className="text-text-secondary text-sm mt-1">{d.drill}</p>}
                {d.detail && <p className="text-text-muted text-sm mt-1">{d.detail}</p>}
                {d.conditions && <p className="text-text-muted text-sm mt-1">Conditions: {d.conditions}</p>}
              </Card>
            </div>
          );
        })}
      </div>
    </div>
  );
}
