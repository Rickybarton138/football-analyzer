import { cn } from '../../lib/utils';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('text-center py-12 bg-surface-alt rounded-card border border-border', className)}>
      {icon && (
        <div className="w-16 h-16 bg-pitch-light rounded-full flex items-center justify-center mx-auto mb-4">
          {icon}
        </div>
      )}
      <h4 className="text-text-primary font-medium mb-2">{title}</h4>
      {description && <p className="text-text-muted text-sm mb-4">{description}</p>}
      {action}
    </div>
  );
}
