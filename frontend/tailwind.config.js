/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Coach Mentor brand
        background: '#FAFAFA',
        surface: '#FFFFFF',
        'surface-alt': '#F1F5F1',
        border: '#E0E0E0',
        'text-primary': '#1A1A1A',
        'text-secondary': '#444444',
        'text-muted': '#888888',
        pitch: {
          DEFAULT: '#43A047',
          deep: '#2E7D32',
          forest: '#1B5E20',
          light: '#E8F5E9',
          50: '#E8F5E9',
          100: '#C8E6C9',
          500: '#43A047',
          600: '#2E7D32',
          700: '#1B5E20',
        },
        gold: {
          DEFAULT: '#E9C46A',
          light: '#FFF8E7',
          dark: '#C4A34A',
        },
        sky: {
          DEFAULT: '#42A5F5',
          light: '#E3F2FD',
          dark: '#1E88E5',
        },
        coral: {
          DEFAULT: '#E76F51',
          light: '#FBE9E7',
          dark: '#D84315',
        },
        // Video analysis dark mode
        'video-bg': '#1A1A1A',
        'video-surface': '#2A2A2A',
        'video-border': '#3A3A3A',
        // Semantic
        alert: {
          immediate: '#E63946',
          tactical: '#FF9800',
          strategic: '#42A5F5',
        },
        team: {
          home: '#3b82f6',
          away: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        display: ['Playfair Display', 'Georgia', 'serif'],
      },
      borderRadius: {
        'card': '12px',
        'btn': '10px',
      },
      boxShadow: {
        'card': '0 2px 12px rgba(0,0,0,0.06)',
        'card-hover': '0 8px 24px rgba(0,0,0,0.10)',
        'btn': '0 4px 15px rgba(67,160,71,0.3)',
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
