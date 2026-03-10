import { cn } from '../../lib/utils';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

const VARIANT_CLASSES = {
  primary: 'bg-gradient-to-r from-pitch-deep to-pitch text-white hover:from-pitch-forest hover:to-pitch-deep shadow-btn',
  secondary: 'bg-pitch-light text-pitch-deep hover:bg-pitch-100 border border-pitch/30',
  ghost: 'text-text-secondary hover:text-text-primary hover:bg-gray-100',
  danger: 'bg-red-50 text-red-700 hover:bg-red-100 border border-red-200',
};

const SIZE_CLASSES = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2.5 text-sm',
  lg: 'px-6 py-3 text-base',
};

export function Button({ variant = 'primary', size = 'md', className, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 font-semibold rounded-btn transition-all disabled:opacity-50 disabled:cursor-not-allowed',
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
