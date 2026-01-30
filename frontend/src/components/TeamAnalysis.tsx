import { useState, useEffect } from 'react';

interface TeamProfile {
  matches_analyzed: number;
  record: string;
  goals: string;
  form: string[];
  avg_possession: number | null;
  avg_shots: number | null;
  avg_xg: number | null;
  home_record?: Record<string, number>;
  away_record?: Record<string, number>;
}

interface Strength {
  category: string;
  description: string;
  confidence: number;
  evidence: string[];
}

interface Recommendation {
  priority: string;
  area: string;
  recommendation: string;
  reasoning: string;
}

interface MyTeamReport {
  team_name: string;
  profile: TeamProfile;
  strengths: Strength[];
  weaknesses: Strength[];
  trends: string[];
  improvement_areas: Recommendation[];
  training_focus: string[];
}

interface OpponentReport {
  opponent: string;
  profile: TeamProfile;
  their_strengths: Strength[];
  their_weaknesses: Strength[];
  key_patterns: string[];
  danger_areas: string[];
  game_plan: {
    recommended_formation: string;
    tactical_approach: string;
    key_battles: string[];
  };
  tactical_recommendations: Recommendation[];
}

export function TeamAnalysis() {
  const [teams, setTeams] = useState<string[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<string>('');
  const [opponentTeam, setOpponentTeam] = useState<string>('');
  const [myTeamReport, setMyTeamReport] = useState<MyTeamReport | null>(null);
  const [opponentReport, setOpponentReport] = useState<OpponentReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'my-team' | 'opponent'>('my-team');
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Ready');
  const [teamsLoading, setTeamsLoading] = useState(true);

  // Fetch available teams
  useEffect(() => {
    setStatus('Loading teams from database...');
    setTeamsLoading(true);
    fetch('http://127.0.0.1:8000/api/teams')
      .then(res => res.json())
      .then(data => {
        setTeams(data.teams || []);
        setStatus(data.count > 0 ? `Ready - ${data.count} teams available` : 'No teams found - scrape some data first');
        setTeamsLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch teams:', err);
        setStatus('Error: Backend not connected');
        setTeamsLoading(false);
      });
  }, []);

  const analyzeMyTeam = async () => {
    if (!selectedTeam) return;
    setLoading(true);
    setError(null);
    setStatus(`Analyzing ${selectedTeam}...`);

    try {
      setStatus(`Fetching match history for ${selectedTeam}...`);
      const res = await fetch(
        `http://127.0.0.1:8000/api/analysis/my-team/${encodeURIComponent(selectedTeam)}`
      );
      if (!res.ok) throw new Error('Failed to analyze team');
      setStatus(`Processing statistics...`);
      const data = await res.json();
      setMyTeamReport(data);
      setStatus(`Analysis complete - ${data.profile.matches_analyzed} matches analyzed`);
    } catch (err) {
      setError('Failed to analyze team. Make sure the team exists in the database.');
      setStatus('Error during analysis');
    } finally {
      setLoading(false);
    }
  };

  const scoutOpponent = async () => {
    if (!opponentTeam) return;
    setLoading(true);
    setError(null);
    setStatus(`Scouting ${opponentTeam}...`);

    try {
      setStatus(`Gathering intelligence on ${opponentTeam}...`);
      const res = await fetch(
        `http://127.0.0.1:8000/api/analysis/opponent/${encodeURIComponent(opponentTeam)}`
      );
      if (!res.ok) throw new Error('Failed to scout opponent');
      setStatus(`Analyzing patterns and weaknesses...`);
      const data = await res.json();
      setOpponentReport(data);
      setStatus(`Scouting complete - ${data.profile.matches_analyzed} matches analyzed`);
    } catch (err) {
      setError('Failed to scout opponent. Make sure the team exists in the database.');
      setStatus('Error during scouting');
    } finally {
      setLoading(false);
    }
  };

  const FormBadge = ({ result }: { result: string }) => {
    const colors: Record<string, string> = {
      W: 'bg-green-500',
      D: 'bg-yellow-500',
      L: 'bg-red-500',
    };
    return (
      <span
        className={`${colors[result] || 'bg-gray-400'} text-white text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center`}
      >
        {result}
      </span>
    );
  };

  const PriorityBadge = ({ priority }: { priority: string }) => {
    const colors: Record<string, string> = {
      high: 'bg-red-100 text-red-800',
      medium: 'bg-yellow-100 text-yellow-800',
      low: 'bg-blue-100 text-blue-800',
    };
    return (
      <span
        className={`${colors[priority] || 'bg-gray-100'} text-xs px-2 py-1 rounded-full font-medium`}
      >
        {priority.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* Status Bar */}
      <div className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm ${
        loading
          ? 'bg-blue-50 border border-blue-200 text-blue-700'
          : status.includes('Error')
            ? 'bg-red-50 border border-red-200 text-red-700'
            : status.includes('complete') || status.includes('Ready')
              ? 'bg-green-50 border border-green-200 text-green-700'
              : 'bg-slate-50 border border-slate-200 text-slate-600'
      }`}>
        {loading && (
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        )}
        {!loading && status.includes('Ready') && (
          <span className="w-2 h-2 bg-green-500 rounded-full" />
        )}
        {!loading && status.includes('complete') && (
          <svg className="h-4 w-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )}
        <span className="font-medium">Status:</span>
        <span>{status}</span>
      </div>

      {/* Tab Selection */}
      <div className="flex border-b border-slate-200">
        <button
          onClick={() => setActiveTab('my-team')}
          className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'my-team'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          My Team Analysis
        </button>
        <button
          onClick={() => setActiveTab('opponent')}
          className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'opponent'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          Scout Opponent
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* My Team Analysis Tab */}
      {activeTab === 'my-team' && (
        <div className="space-y-4">
          <div className="flex gap-3">
            <select
              value={selectedTeam}
              onChange={(e) => setSelectedTeam(e.target.value)}
              className="flex-1 px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select your team...</option>
              {teams.map((team) => (
                <option key={team} value={team}>
                  {team}
                </option>
              ))}
            </select>
            <button
              onClick={analyzeMyTeam}
              disabled={!selectedTeam || loading}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze Team'}
            </button>
          </div>

          {myTeamReport && (
            <div className="space-y-6">
              {/* Team Overview */}
              <div className="bg-slate-50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-slate-800 mb-3">
                  {myTeamReport.team_name} Overview
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-slate-500 text-sm">Matches</p>
                    <p className="font-semibold">{myTeamReport.profile.matches_analyzed}</p>
                  </div>
                  <div>
                    <p className="text-slate-500 text-sm">Record</p>
                    <p className="font-semibold">{myTeamReport.profile.record}</p>
                  </div>
                  <div>
                    <p className="text-slate-500 text-sm">Goals</p>
                    <p className="font-semibold">{myTeamReport.profile.goals}</p>
                  </div>
                  <div>
                    <p className="text-slate-500 text-sm">Form</p>
                    <div className="flex gap-1">
                      {myTeamReport.profile.form.map((r, i) => (
                        <FormBadge key={i} result={r} />
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-600">
                    {myTeamReport.profile.avg_possession ?? '-'}%
                  </p>
                  <p className="text-slate-500 text-sm">Avg Possession</p>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-600">
                    {myTeamReport.profile.avg_shots ?? '-'}
                  </p>
                  <p className="text-slate-500 text-sm">Avg Shots</p>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-600">
                    {myTeamReport.profile.avg_xg ?? '-'}
                  </p>
                  <p className="text-slate-500 text-sm">Avg xG</p>
                </div>
              </div>

              {/* Strengths */}
              {myTeamReport.strengths.length > 0 && (
                <div>
                  <h4 className="font-semibold text-green-700 mb-2">Strengths</h4>
                  <div className="space-y-2">
                    {myTeamReport.strengths.map((s, i) => (
                      <div key={i} className="bg-green-50 border border-green-200 rounded p-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">
                              {s.category}
                            </span>
                            <p className="mt-1 font-medium text-green-800">{s.description}</p>
                          </div>
                        </div>
                        {s.evidence.length > 0 && (
                          <p className="text-sm text-green-600 mt-1">{s.evidence[0]}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Weaknesses */}
              {myTeamReport.weaknesses.length > 0 && (
                <div>
                  <h4 className="font-semibold text-red-700 mb-2">Areas to Improve</h4>
                  <div className="space-y-2">
                    {myTeamReport.weaknesses.map((w, i) => (
                      <div key={i} className="bg-red-50 border border-red-200 rounded p-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <span className="text-xs bg-red-100 text-red-800 px-2 py-0.5 rounded">
                              {w.category}
                            </span>
                            <p className="mt-1 font-medium text-red-800">{w.description}</p>
                          </div>
                        </div>
                        {w.evidence.length > 0 && (
                          <p className="text-sm text-red-600 mt-1">{w.evidence[0]}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Trends */}
              {myTeamReport.trends.length > 0 && (
                <div>
                  <h4 className="font-semibold text-slate-700 mb-2">Current Trends</h4>
                  <ul className="list-disc list-inside space-y-1 text-slate-600">
                    {myTeamReport.trends.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Improvement Recommendations */}
              {myTeamReport.improvement_areas.length > 0 && (
                <div>
                  <h4 className="font-semibold text-slate-700 mb-2">Recommended Improvements</h4>
                  <div className="space-y-2">
                    {myTeamReport.improvement_areas.map((r, i) => (
                      <div key={i} className="bg-white border border-slate-200 rounded p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <PriorityBadge priority={r.priority} />
                          <span className="text-xs text-slate-500 uppercase">{r.area}</span>
                        </div>
                        <p className="font-medium text-slate-800">{r.recommendation}</p>
                        <p className="text-sm text-slate-500 mt-1">{r.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Training Focus */}
              {myTeamReport.training_focus.length > 0 && (
                <div>
                  <h4 className="font-semibold text-slate-700 mb-2">Training Focus</h4>
                  <div className="flex flex-wrap gap-2">
                    {myTeamReport.training_focus.map((focus, i) => (
                      <span
                        key={i}
                        className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm"
                      >
                        {focus}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Opponent Scouting Tab */}
      {activeTab === 'opponent' && (
        <div className="space-y-4">
          <div className="flex gap-3">
            <select
              value={opponentTeam}
              onChange={(e) => setOpponentTeam(e.target.value)}
              className="flex-1 px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select opponent...</option>
              {teams.map((team) => (
                <option key={team} value={team}>
                  {team}
                </option>
              ))}
            </select>
            <button
              onClick={scoutOpponent}
              disabled={!opponentTeam || loading}
              className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Scouting...' : 'Scout Opponent'}
            </button>
          </div>

          {opponentReport && (
            <div className="space-y-6">
              {/* Opponent Overview */}
              <div className="bg-orange-50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-slate-800 mb-3">
                  Scouting Report: {opponentReport.opponent}
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-slate-500 text-sm">Matches</p>
                    <p className="font-semibold">{opponentReport.profile.matches_analyzed}</p>
                  </div>
                  <div>
                    <p className="text-slate-500 text-sm">Record</p>
                    <p className="font-semibold">{opponentReport.profile.record}</p>
                  </div>
                  <div>
                    <p className="text-slate-500 text-sm">Goals</p>
                    <p className="font-semibold">{opponentReport.profile.goals}</p>
                  </div>
                  <div>
                    <p className="text-slate-500 text-sm">Form</p>
                    <div className="flex gap-1">
                      {opponentReport.profile.form.map((r, i) => (
                        <FormBadge key={i} result={r} />
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Patterns */}
              {opponentReport.key_patterns.length > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-800 mb-2">Key Patterns</h4>
                  <ul className="list-disc list-inside space-y-1 text-blue-700">
                    {opponentReport.key_patterns.map((p, i) => (
                      <li key={i}>{p}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Danger Areas */}
              {opponentReport.danger_areas.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h4 className="font-semibold text-red-800 mb-2">Danger Areas - Watch Out!</h4>
                  <ul className="list-disc list-inside space-y-1 text-red-700">
                    {opponentReport.danger_areas.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Their Strengths (to nullify) */}
              {opponentReport.their_strengths.length > 0 && (
                <div>
                  <h4 className="font-semibold text-orange-700 mb-2">
                    Their Strengths - Must Nullify
                  </h4>
                  <div className="space-y-2">
                    {opponentReport.their_strengths.map((s, i) => (
                      <div key={i} className="bg-orange-50 border border-orange-200 rounded p-3">
                        <span className="text-xs bg-orange-100 text-orange-800 px-2 py-0.5 rounded">
                          {s.category}
                        </span>
                        <p className="mt-1 font-medium text-orange-800">{s.description}</p>
                        {s.evidence.length > 0 && (
                          <p className="text-sm text-orange-600 mt-1">{s.evidence[0]}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Their Weaknesses (to exploit) */}
              {opponentReport.their_weaknesses.length > 0 && (
                <div>
                  <h4 className="font-semibold text-green-700 mb-2">
                    Their Weaknesses - Exploit These!
                  </h4>
                  <div className="space-y-2">
                    {opponentReport.their_weaknesses.map((w, i) => (
                      <div key={i} className="bg-green-50 border border-green-200 rounded p-3">
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">
                          {w.category}
                        </span>
                        <p className="mt-1 font-medium text-green-800">{w.description}</p>
                        {w.evidence.length > 0 && (
                          <p className="text-sm text-green-600 mt-1">{w.evidence[0]}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Game Plan */}
              <div className="bg-slate-800 text-white rounded-lg p-4">
                <h4 className="font-semibold text-lg mb-3">Recommended Game Plan</h4>
                <div className="space-y-3">
                  <div>
                    <p className="text-slate-400 text-sm">Formation</p>
                    <p className="text-xl font-bold">
                      {opponentReport.game_plan.recommended_formation || 'Flexible'}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-400 text-sm">Tactical Approach</p>
                    <p className="font-medium">
                      {opponentReport.game_plan.tactical_approach || 'Balanced'}
                    </p>
                  </div>
                  {opponentReport.game_plan.key_battles.length > 0 && (
                    <div>
                      <p className="text-slate-400 text-sm mb-1">Key Battles to Win</p>
                      <ul className="space-y-1">
                        {opponentReport.game_plan.key_battles.map((b, i) => (
                          <li key={i} className="flex items-center gap-2">
                            <span className="w-2 h-2 bg-yellow-400 rounded-full" />
                            {b}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>

              {/* Tactical Recommendations */}
              {opponentReport.tactical_recommendations.length > 0 && (
                <div>
                  <h4 className="font-semibold text-slate-700 mb-2">Tactical Recommendations</h4>
                  <div className="space-y-2">
                    {opponentReport.tactical_recommendations.map((r, i) => (
                      <div key={i} className="bg-white border border-slate-200 rounded p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <PriorityBadge priority={r.priority} />
                          <span className="text-xs text-slate-500 uppercase">{r.area}</span>
                        </div>
                        <p className="font-medium text-slate-800">{r.recommendation}</p>
                        <p className="text-sm text-slate-500 mt-1">{r.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
