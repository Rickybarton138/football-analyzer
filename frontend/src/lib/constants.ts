export const APP_NAME = 'DugoutIQ';
export const APP_TAGLINE = 'AI Coaching Intelligence';

export const BRAND = {
  primary: '#10B981',    // Emerald — the pitch
  accent: '#06B6D4',     // Cyan — data/AI
  background: '#0f172a', // Slate-900
  surface: '#111827',    // Cards/panels
  emerald: {
    400: '#34d399',
    500: '#10B981',
    600: '#059669',
  },
  cyan: {
    400: '#22d3ee',
    500: '#06B6D4',
    600: '#0891b2',
  },
} as const;

export const SEVERITY_COLORS = {
  critical: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
  high: { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30' },
  medium: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
  low: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  info: { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30' },
} as const;

export const RATING_COLORS: Record<string, string> = {
  Excellent: 'text-green-400',
  Good: 'text-emerald-400',
  Average: 'text-amber-400',
  Poor: 'text-red-400',
};

export const PRIORITY_LABELS: Record<string, string> = {
  critical: 'Critical',
  high: 'High',
  medium: 'Medium',
  low: 'Low',
  info: 'Info',
};

export const FORMATIONS = ['4-4-2', '4-3-3', '4-2-3-1', '3-5-2', '3-4-3', '5-3-2', '5-4-1', '4-5-1', '4-1-4-1'];

export const JERSEY_COLORS = [
  { name: 'Red', value: '#dc2626' },
  { name: 'Blue', value: '#3b82f6' },
  { name: 'Green', value: '#16a34a' },
  { name: 'Yellow', value: '#eab308' },
  { name: 'White', value: '#f8fafc' },
  { name: 'Black', value: '#1e293b' },
  { name: 'Orange', value: '#ea580c' },
  { name: 'Purple', value: '#9333ea' },
  { name: 'Pink', value: '#ec4899' },
  { name: 'Sky Blue', value: '#0ea5e9' },
];
