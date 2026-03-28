const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...options?.headers as Record<string, string> };
  if (!headers['Content-Type'] && options?.body && typeof options.body === 'string') {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.message || 'Request failed');
  }
  return res.json();
}

export const api = {
  // Matches
  listMatches: () => request<any[]>('/matches'),
  getMatch: (matchId: string) => request(`/matches/${matchId}`),
  getMatchStatus: (matchId: string) => request(`/matches/${matchId}/status`),

  // Analysis
  runAnalysis: (data: { match_id: string; analysis_type?: string; prompt?: string }) =>
    request('/analysis/run', { method: 'POST', body: JSON.stringify(data) }),

  getAnalysis: (analysisId: string) => request(`/analysis/${analysisId}`),
  getMatchAnalyses: (matchId: string) => request<any[]>(`/analysis/match/${matchId}`),

  askQuestion: (matchId: string, question: string) =>
    request(`/analysis/ask?match_id=${matchId}&question=${encodeURIComponent(question)}`, { method: 'POST' }),

  generateSessionPlan: (matchId: string, minutes?: number) =>
    request(`/analysis/session-plan?match_id=${matchId}&available_minutes=${minutes || 90}`, { method: 'POST' }),

  // Search
  searchFootage: (query: string, matchId?: string) =>
    request('/search', { method: 'POST', body: JSON.stringify({ query, match_id: matchId }) }),

  // Players
  listPlayers: () => request<any[]>('/players'),
  createPlayer: (data: { name: string; squad_number?: number; position?: string }) =>
    request('/players', { method: 'POST', body: JSON.stringify(data) }),
};
