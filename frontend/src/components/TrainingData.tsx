import { useState, useCallback } from 'react';
import { Upload, Database, FileJson, FileSpreadsheet, Plus, Download, BarChart3 } from 'lucide-react';

interface TrainingStats {
  total_matches: number;
  total_players: number;
  total_events: number;
  total_frames: number;
  event_breakdown: {
    goals: number;
    shots: number;
    passes: number;
  };
  matches_by_competition: Record<string, number>;
}

interface MatchForm {
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  date: string;
  competition: string;
  venue: string;
  home_possession: number | null;
  away_possession: number | null;
  home_shots: number | null;
  away_shots: number | null;
  home_shots_on_target: number | null;
  away_shots_on_target: number | null;
  home_passes: number | null;
  away_passes: number | null;
  home_pass_accuracy: number | null;
  away_pass_accuracy: number | null;
  home_corners: number | null;
  away_corners: number | null;
  home_fouls: number | null;
  away_fouls: number | null;
  home_xg: number | null;
  away_xg: number | null;
}

const initialMatchForm: MatchForm = {
  home_team: '',
  away_team: '',
  home_score: 0,
  away_score: 0,
  date: new Date().toISOString().split('T')[0],
  competition: '',
  venue: '',
  home_possession: null,
  away_possession: null,
  home_shots: null,
  away_shots: null,
  home_shots_on_target: null,
  away_shots_on_target: null,
  home_passes: null,
  away_passes: null,
  home_pass_accuracy: null,
  away_pass_accuracy: null,
  home_corners: null,
  away_corners: null,
  home_fouls: null,
  away_fouls: null,
  home_xg: null,
  away_xg: null,
};

export function TrainingData() {
  const [activeTab, setActiveTab] = useState<'upload' | 'manual' | 'export'>('upload');
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [matchForm, setMatchForm] = useState<MatchForm>(initialMatchForm);

  // Fetch stats on mount
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/training/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  }, []);

  // Upload CSV file
  const handleCsvUpload = async (file: File, dataType: string) => {
    setIsLoading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`/api/training/upload/csv?data_type=${dataType}`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setMessage({ type: 'success', text: `Uploaded ${result.uploaded} ${dataType}` });
        fetchStats();
      } else {
        setMessage({ type: 'error', text: 'Upload failed' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Upload failed: ' + String(error) });
    } finally {
      setIsLoading(false);
    }
  };

  // Upload JSON file
  const handleJsonUpload = async (file: File) => {
    setIsLoading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/training/upload/json', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setMessage({
          type: 'success',
          text: `Uploaded ${result.uploaded.matches} matches, ${result.uploaded.players} players, ${result.uploaded.events} events`
        });
        fetchStats();
      } else {
        setMessage({ type: 'error', text: 'Upload failed' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Upload failed: ' + String(error) });
    } finally {
      setIsLoading(false);
    }
  };

  // Add match manually
  const handleAddMatch = async () => {
    setIsLoading(true);
    setMessage(null);

    const params = new URLSearchParams();
    params.append('home_team', matchForm.home_team);
    params.append('away_team', matchForm.away_team);
    params.append('home_score', String(matchForm.home_score));
    params.append('away_score', String(matchForm.away_score));
    params.append('date', matchForm.date);
    if (matchForm.competition) params.append('competition', matchForm.competition);
    if (matchForm.venue) params.append('venue', matchForm.venue);
    if (matchForm.home_possession !== null) params.append('home_possession', String(matchForm.home_possession));
    if (matchForm.away_possession !== null) params.append('away_possession', String(matchForm.away_possession));
    if (matchForm.home_shots !== null) params.append('home_shots', String(matchForm.home_shots));
    if (matchForm.away_shots !== null) params.append('away_shots', String(matchForm.away_shots));
    if (matchForm.home_shots_on_target !== null) params.append('home_shots_on_target', String(matchForm.home_shots_on_target));
    if (matchForm.away_shots_on_target !== null) params.append('away_shots_on_target', String(matchForm.away_shots_on_target));
    if (matchForm.home_passes !== null) params.append('home_passes', String(matchForm.home_passes));
    if (matchForm.away_passes !== null) params.append('away_passes', String(matchForm.away_passes));
    if (matchForm.home_xg !== null) params.append('home_xg', String(matchForm.home_xg));
    if (matchForm.away_xg !== null) params.append('away_xg', String(matchForm.away_xg));

    try {
      const response = await fetch(`/api/training/matches?${params.toString()}`, {
        method: 'POST',
      });

      if (response.ok) {
        setMessage({ type: 'success', text: 'Match added successfully' });
        setMatchForm(initialMatchForm);
        fetchStats();
      } else {
        setMessage({ type: 'error', text: 'Failed to add match' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to add match: ' + String(error) });
    } finally {
      setIsLoading(false);
    }
  };

  // Export data
  const handleExport = (format: string, dataType?: string) => {
    if (format === 'json') {
      window.open('/api/training/export/json', '_blank');
    } else if (format === 'csv' && dataType) {
      window.open(`/api/training/export/csv/${dataType}`, '_blank');
    }
  };

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Matches"
          value={stats?.total_matches || 0}
          icon={<Database className="w-5 h-5" />}
          onClick={fetchStats}
        />
        <StatCard
          label="Players"
          value={stats?.total_players || 0}
          icon={<BarChart3 className="w-5 h-5" />}
        />
        <StatCard
          label="Events"
          value={stats?.total_events || 0}
          icon={<BarChart3 className="w-5 h-5" />}
        />
        <StatCard
          label="Frames"
          value={stats?.total_frames || 0}
          icon={<BarChart3 className="w-5 h-5" />}
        />
      </div>

      {/* Message */}
      {message && (
        <div className={`p-3 rounded-lg ${message.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {message.text}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-200">
        <TabButton active={activeTab === 'upload'} onClick={() => setActiveTab('upload')}>
          <Upload className="w-4 h-4" />
          Upload Data
        </TabButton>
        <TabButton active={activeTab === 'manual'} onClick={() => setActiveTab('manual')}>
          <Plus className="w-4 h-4" />
          Add Match
        </TabButton>
        <TabButton active={activeTab === 'export'} onClick={() => setActiveTab('export')}>
          <Download className="w-4 h-4" />
          Export
        </TabButton>
      </div>

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="space-y-4">
          <p className="text-slate-600 text-sm">
            Upload match data, player stats, and events to build your training dataset.
            Supported formats: CSV, JSON
          </p>

          <div className="grid grid-cols-2 gap-4">
            {/* CSV Upload */}
            <UploadCard
              title="Matches CSV"
              description="Upload match results and team stats"
              icon={<FileSpreadsheet className="w-8 h-8 text-green-500" />}
              accept=".csv"
              onUpload={(file) => handleCsvUpload(file, 'matches')}
              isLoading={isLoading}
            />
            <UploadCard
              title="Players CSV"
              description="Upload player-level statistics"
              icon={<FileSpreadsheet className="w-8 h-8 text-blue-500" />}
              accept=".csv"
              onUpload={(file) => handleCsvUpload(file, 'players')}
              isLoading={isLoading}
            />
            <UploadCard
              title="Events CSV"
              description="Upload match events (goals, shots, passes)"
              icon={<FileSpreadsheet className="w-8 h-8 text-purple-500" />}
              accept=".csv"
              onUpload={(file) => handleCsvUpload(file, 'events')}
              isLoading={isLoading}
            />
            <UploadCard
              title="Full JSON Dataset"
              description="Upload complete dataset with all data types"
              icon={<FileJson className="w-8 h-8 text-orange-500" />}
              accept=".json"
              onUpload={handleJsonUpload}
              isLoading={isLoading}
            />
          </div>

          {/* CSV Format Guide */}
          <div className="bg-slate-50 rounded-lg p-4">
            <h4 className="font-medium text-slate-800 mb-2">CSV Format Guide</h4>
            <div className="text-sm text-slate-600 space-y-2">
              <p><strong>Matches CSV columns:</strong> home_team, away_team, home_score, away_score, date, competition, venue, home_possession, away_possession, home_shots, away_shots, home_xg, away_xg</p>
              <p><strong>Players CSV columns:</strong> match_id, name, team, position, minutes_played, goals, assists, passes, pass_accuracy, tackles, distance_covered_km</p>
              <p><strong>Events CSV columns:</strong> match_id, timestamp_seconds, event_type, team, player_name, x_position, y_position, outcome</p>
            </div>
          </div>
        </div>
      )}

      {/* Manual Add Tab */}
      {activeTab === 'manual' && (
        <div className="space-y-4">
          <p className="text-slate-600 text-sm">
            Manually add match data and statistics to your training dataset.
          </p>

          <div className="grid grid-cols-2 gap-6">
            {/* Basic Info */}
            <div className="space-y-4">
              <h4 className="font-medium text-slate-800">Match Info</h4>

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Home Team"
                  value={matchForm.home_team}
                  onChange={(v) => setMatchForm({ ...matchForm, home_team: v })}
                  required
                />
                <Input
                  label="Away Team"
                  value={matchForm.away_team}
                  onChange={(v) => setMatchForm({ ...matchForm, away_team: v })}
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Home Score"
                  type="number"
                  value={String(matchForm.home_score)}
                  onChange={(v) => setMatchForm({ ...matchForm, home_score: parseInt(v) || 0 })}
                />
                <Input
                  label="Away Score"
                  type="number"
                  value={String(matchForm.away_score)}
                  onChange={(v) => setMatchForm({ ...matchForm, away_score: parseInt(v) || 0 })}
                />
              </div>

              <Input
                label="Date"
                type="date"
                value={matchForm.date}
                onChange={(v) => setMatchForm({ ...matchForm, date: v })}
              />

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Competition"
                  value={matchForm.competition}
                  onChange={(v) => setMatchForm({ ...matchForm, competition: v })}
                  placeholder="e.g., Premier League"
                />
                <Input
                  label="Venue"
                  value={matchForm.venue}
                  onChange={(v) => setMatchForm({ ...matchForm, venue: v })}
                  placeholder="Stadium name"
                />
              </div>
            </div>

            {/* Stats */}
            <div className="space-y-4">
              <h4 className="font-medium text-slate-800">Match Stats (Optional)</h4>

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Home Possession %"
                  type="number"
                  value={matchForm.home_possession !== null ? String(matchForm.home_possession) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, home_possession: v ? parseFloat(v) : null })}
                  placeholder="e.g., 55"
                />
                <Input
                  label="Away Possession %"
                  type="number"
                  value={matchForm.away_possession !== null ? String(matchForm.away_possession) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, away_possession: v ? parseFloat(v) : null })}
                  placeholder="e.g., 45"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Home Shots"
                  type="number"
                  value={matchForm.home_shots !== null ? String(matchForm.home_shots) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, home_shots: v ? parseInt(v) : null })}
                />
                <Input
                  label="Away Shots"
                  type="number"
                  value={matchForm.away_shots !== null ? String(matchForm.away_shots) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, away_shots: v ? parseInt(v) : null })}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Home xG"
                  type="number"
                  step="0.01"
                  value={matchForm.home_xg !== null ? String(matchForm.home_xg) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, home_xg: v ? parseFloat(v) : null })}
                  placeholder="e.g., 1.85"
                />
                <Input
                  label="Away xG"
                  type="number"
                  step="0.01"
                  value={matchForm.away_xg !== null ? String(matchForm.away_xg) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, away_xg: v ? parseFloat(v) : null })}
                  placeholder="e.g., 0.92"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Home Passes"
                  type="number"
                  value={matchForm.home_passes !== null ? String(matchForm.home_passes) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, home_passes: v ? parseInt(v) : null })}
                />
                <Input
                  label="Away Passes"
                  type="number"
                  value={matchForm.away_passes !== null ? String(matchForm.away_passes) : ''}
                  onChange={(v) => setMatchForm({ ...matchForm, away_passes: v ? parseInt(v) : null })}
                />
              </div>
            </div>
          </div>

          <button
            onClick={handleAddMatch}
            disabled={isLoading || !matchForm.home_team || !matchForm.away_team}
            className={`
              w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg
              font-medium transition-colors
              ${isLoading || !matchForm.home_team || !matchForm.away_team
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-blue-500 text-white hover:bg-blue-600'}
            `}
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Adding...
              </>
            ) : (
              <>
                <Plus className="w-5 h-5" />
                Add Match to Training Data
              </>
            )}
          </button>
        </div>
      )}

      {/* Export Tab */}
      {activeTab === 'export' && (
        <div className="space-y-4">
          <p className="text-slate-600 text-sm">
            Export your training dataset for use with ML frameworks.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <ExportCard
              title="Full JSON Export"
              description="Complete dataset with matches, players, and events"
              onClick={() => handleExport('json')}
            />
            <ExportCard
              title="Matches CSV"
              description="Match results and team statistics"
              onClick={() => handleExport('csv', 'matches')}
            />
            <ExportCard
              title="Players CSV"
              description="Player-level statistics"
              onClick={() => handleExport('csv', 'players')}
            />
            <ExportCard
              title="Events CSV"
              description="Match events and coordinates"
              onClick={() => handleExport('csv', 'events')}
            />
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-medium text-blue-800 mb-2">ML Training Tips</h4>
            <ul className="text-sm text-blue-700 list-disc list-inside space-y-1">
              <li>Aim for at least 100+ matches for basic models</li>
              <li>Include diverse competitions and team levels</li>
              <li>Event coordinates enable spatial models (xG, pass prediction)</li>
              <li>Player stats allow individual performance modeling</li>
              <li>Export to YOLO format for custom detection training</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper Components

function StatCard({ label, value, icon, onClick }: { label: string; value: number; icon: React.ReactNode; onClick?: () => void }) {
  return (
    <div
      className={`bg-white rounded-lg p-4 border border-slate-200 ${onClick ? 'cursor-pointer hover:border-blue-300' : ''}`}
      onClick={onClick}
    >
      <div className="flex items-center gap-3">
        <div className="text-slate-400">{icon}</div>
        <div>
          <div className="text-2xl font-bold text-slate-800">{value}</div>
          <div className="text-sm text-slate-500">{label}</div>
        </div>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors
        ${active
          ? 'border-blue-500 text-blue-600'
          : 'border-transparent text-slate-500 hover:text-slate-700'}
      `}
    >
      {children}
    </button>
  );
}

function UploadCard({
  title,
  description,
  icon,
  accept,
  onUpload,
  isLoading
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
  accept: string;
  onUpload: (file: File) => void;
  isLoading: boolean;
}) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <label className={`
      flex flex-col items-center gap-3 p-6 border-2 border-dashed border-slate-300 rounded-lg
      cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors
      ${isLoading ? 'opacity-50 pointer-events-none' : ''}
    `}>
      {icon}
      <div className="text-center">
        <div className="font-medium text-slate-800">{title}</div>
        <div className="text-sm text-slate-500">{description}</div>
      </div>
      <input type="file" accept={accept} onChange={handleChange} className="hidden" />
    </label>
  );
}

function ExportCard({ title, description, onClick }: { title: string; description: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-4 p-4 border border-slate-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors text-left"
    >
      <Download className="w-6 h-6 text-blue-500" />
      <div>
        <div className="font-medium text-slate-800">{title}</div>
        <div className="text-sm text-slate-500">{description}</div>
      </div>
    </button>
  );
}

function Input({
  label,
  value,
  onChange,
  type = 'text',
  placeholder,
  required,
  step
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  placeholder?: string;
  required?: boolean;
  step?: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        step={step}
        className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      />
    </div>
  );
}
