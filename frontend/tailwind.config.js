/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f172a',
        surface: '#111827',
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
        pitch: {
          grass: '#2d5a27',
          lines: '#ffffff',
        },
        team: {
          home: '#3b82f6',
          away: '#ef4444',
        },
        alert: {
          immediate: '#ef4444',
          tactical: '#f59e0b',
          strategic: '#3b82f6',
        },
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
