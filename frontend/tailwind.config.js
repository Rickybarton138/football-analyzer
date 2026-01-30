/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
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
        }
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
