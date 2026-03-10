export const APP_NAME = 'CoachMentor';
export const APP_MODULE = 'Match Analysis';
export const APP_TAGLINE = 'AI Coaching Intelligence';

export const BRAND = {
  primary: '#43A047',     // Pitch Green
  deep: '#2E7D32',        // Deep Green
  forest: '#1B5E20',      // Forest Green
  gold: '#E9C46A',        // Gold highlights
  sky: '#42A5F5',         // Data/AI blue
  coral: '#E76F51',       // Alerts/warnings
  background: '#FAFAFA',  // Off-white
  surface: '#FFFFFF',     // Cards
  text: '#1A1A1A',        // Charcoal
  textSecondary: '#444444',
  textMuted: '#888888',
  border: '#E0E0E0',
} as const;

export const SEVERITY_COLORS = {
  critical: { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-300' },
  high: { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-300' },
  medium: { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-300' },
  low: { bg: 'bg-sky-100', text: 'text-sky-700', border: 'border-sky-300' },
  info: { bg: 'bg-gray-100', text: 'text-gray-600', border: 'border-gray-300' },
} as const;

export const RATING_COLORS: Record<string, string> = {
  Excellent: 'text-pitch-500',
  Good: 'text-pitch-600',
  Average: 'text-amber-600',
  Poor: 'text-red-600',
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
