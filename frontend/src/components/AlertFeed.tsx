import { X, AlertTriangle, Zap, Lightbulb } from 'lucide-react';
import clsx from 'clsx';
import type { TacticalAlert, AlertPriority } from '../types';

interface AlertFeedProps {
  alerts: TacticalAlert[];
  onDismiss: (alertId: string) => void;
}

export function AlertFeed({ alerts, onDismiss }: AlertFeedProps) {
  if (alerts.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400">
        <p className="text-sm">No active alerts</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto space-y-2">
      {alerts.map((alert) => (
        <AlertCard
          key={alert.alert_id}
          alert={alert}
          onDismiss={() => onDismiss(alert.alert_id)}
        />
      ))}
    </div>
  );
}

interface AlertCardProps {
  alert: TacticalAlert;
  onDismiss: () => void;
}

function AlertCard({ alert, onDismiss }: AlertCardProps) {
  const priorityStyles = getPriorityStyles(alert.priority);

  return (
    <div
      className={clsx(
        'relative rounded-lg p-3 border',
        priorityStyles.bg,
        priorityStyles.border,
        alert.priority === 'immediate' && 'alert-immediate'
      )}
    >
      {/* Dismiss button */}
      <button
        onClick={onDismiss}
        className="absolute top-2 right-2 text-slate-400 hover:text-white"
      >
        <X className="w-4 h-4" />
      </button>

      {/* Priority icon and message */}
      <div className="flex items-start gap-2 pr-6">
        <div className={clsx('mt-0.5', priorityStyles.icon)}>
          {getPriorityIcon(alert.priority)}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm">{alert.message}</p>
          {alert.details && (
            <p className="text-xs text-slate-300 mt-1">{alert.details}</p>
          )}
          {alert.suggested_action && (
            <p className="text-xs mt-2 text-green-400">
              → {alert.suggested_action}
            </p>
          )}
        </div>
      </div>

      {/* Related players */}
      {alert.related_players.length > 0 && (
        <div className="mt-2 flex items-center gap-1">
          <span className="text-xs text-slate-400">Players:</span>
          {alert.related_players.map((id) => (
            <span
              key={id}
              className="text-xs bg-slate-700 px-1.5 py-0.5 rounded"
            >
              #{id}
            </span>
          ))}
        </div>
      )}

      {/* Timestamp */}
      <div className="mt-2 text-xs text-slate-400">
        {formatAlertTime(alert.timestamp_ms)}
      </div>
    </div>
  );
}

function getPriorityStyles(priority: AlertPriority) {
  switch (priority) {
    case 'immediate':
      return {
        bg: 'bg-red-900/50',
        border: 'border-red-500',
        icon: 'text-red-400',
      };
    case 'tactical':
      return {
        bg: 'bg-amber-900/50',
        border: 'border-amber-500',
        icon: 'text-amber-400',
      };
    case 'strategic':
      return {
        bg: 'bg-blue-900/50',
        border: 'border-blue-500',
        icon: 'text-blue-400',
      };
  }
}

function getPriorityIcon(priority: AlertPriority) {
  switch (priority) {
    case 'immediate':
      return <Zap className="w-4 h-4" />;
    case 'tactical':
      return <AlertTriangle className="w-4 h-4" />;
    case 'strategic':
      return <Lightbulb className="w-4 h-4" />;
  }
}

function formatAlertTime(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  return `${minutes}'${seconds.toString().padStart(2, '0')}`;
}
