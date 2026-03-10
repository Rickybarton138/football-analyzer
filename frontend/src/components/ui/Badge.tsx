import { cn } from '../../lib/utils';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'critical' | 'high' | 'medium' | 'low' | 'info' | 'success';
  className?: string;
}

const VARIANT_CLASSES: Record<string, string> = {
  default: 'bg-gray-100 text-gray-600 border-gray-200',
  critical: 'bg-red-50 text-red-700 border-red-200',
  high: 'bg-orange-50 text-orange-700 border-orange-200',
  medium: 'bg-amber-50 text-amber-700 border-amber-200',
  low: 'bg-sky-50 text-sky-700 border-sky-200',
  info: 'bg-gray-50 text-gray-500 border-gray-200',
  success: 'bg-pitch-light text-pitch-deep border-pitch/30',
};

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span className={cn(
      'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
      VARIANT_CLASSES[variant] || VARIANT_CLASSES.default,
      className
    )}>
      {children}
    </span>
  );
}
