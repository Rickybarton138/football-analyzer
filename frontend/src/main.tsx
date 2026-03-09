import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('React Error Boundary caught:', error, info);
  }
  render() {
    if (this.state.error) {
      return React.createElement('div', { style: { color: 'white', padding: 40, fontFamily: 'monospace' } },
        React.createElement('h1', null, 'Something went wrong'),
        React.createElement('pre', { style: { color: '#ef4444', whiteSpace: 'pre-wrap' } }, this.state.error.message),
        React.createElement('pre', { style: { color: '#94a3b8', fontSize: 12 } }, this.state.error.stack),
      );
    }
    return this.props.children;
  }
}

// Register service worker for PWA share target
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>,
)
