import { useState, useCallback } from 'react';
import { Users, Palette, Play, Settings } from 'lucide-react';

interface MatchSetupProps {
  onMatchCreated: (matchId: string) => void;
  videoId?: string;
}

interface TeamConfig {
  name: string;
  color: string;
}

export function MatchSetup({ onMatchCreated, videoId }: MatchSetupProps) {
  const [homeTeam, setHomeTeam] = useState<TeamConfig>({ name: '', color: '#3b82f6' });
  const [awayTeam, setAwayTeam] = useState<TeamConfig>({ name: '', color: '#ef4444' });
  const [venue, setVenue] = useState('');
  const [competition, setCompetition] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [processingMode, setProcessingMode] = useState<'live' | 'post_match'>('post_match');

  const handleCreate = useCallback(async () => {
    if (!homeTeam.name || !awayTeam.name) {
      alert('Please enter both team names');
      return;
    }

    setIsCreating(true);

    try {
      // Create match
      const matchResponse = await fetch(
        `/api/match/create?home_team=${encodeURIComponent(homeTeam.name)}&away_team=${encodeURIComponent(awayTeam.name)}&venue=${encodeURIComponent(venue)}&competition=${encodeURIComponent(competition)}`,
        { method: 'POST' }
      );

      if (!matchResponse.ok) {
        throw new Error('Failed to create match');
      }

      const matchData = await matchResponse.json();
      const matchId = matchData.match_id;

      // Set team colors
      await fetch(`/api/match/${matchId}/teams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          home_team_color: hexToRgb(homeTeam.color),
          away_team_color: hexToRgb(awayTeam.color),
          home_team_name: homeTeam.name,
          away_team_name: awayTeam.name,
        }),
      });

      // Start processing if video uploaded
      if (videoId) {
        await fetch(`/api/video/${videoId}/process?mode=${processingMode}`, {
          method: 'POST',
        });
      }

      onMatchCreated(matchId);
    } catch (error) {
      console.error('Error creating match:', error);
      alert('Failed to create match. Please try again.');
    } finally {
      setIsCreating(false);
    }
  }, [homeTeam, awayTeam, venue, competition, videoId, processingMode, onMatchCreated]);

  return (
    <div className="space-y-6">
      {/* Teams */}
      <div className="grid grid-cols-2 gap-4">
        {/* Home Team */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
            <Users className="w-4 h-4" />
            Home Team
          </label>
          <input
            type="text"
            value={homeTeam.name}
            onChange={(e) => setHomeTeam({ ...homeTeam, name: e.target.value })}
            placeholder="Team name"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <div className="flex items-center gap-2">
            <Palette className="w-4 h-4 text-slate-400" />
            <input
              type="color"
              value={homeTeam.color}
              onChange={(e) => setHomeTeam({ ...homeTeam, color: e.target.value })}
              className="w-8 h-8 rounded cursor-pointer"
            />
            <span className="text-xs text-slate-500">Jersey color</span>
          </div>
        </div>

        {/* Away Team */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
            <Users className="w-4 h-4" />
            Away Team
          </label>
          <input
            type="text"
            value={awayTeam.name}
            onChange={(e) => setAwayTeam({ ...awayTeam, name: e.target.value })}
            placeholder="Team name"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <div className="flex items-center gap-2">
            <Palette className="w-4 h-4 text-slate-400" />
            <input
              type="color"
              value={awayTeam.color}
              onChange={(e) => setAwayTeam({ ...awayTeam, color: e.target.value })}
              className="w-8 h-8 rounded cursor-pointer"
            />
            <span className="text-xs text-slate-500">Jersey color</span>
          </div>
        </div>
      </div>

      {/* Optional Details */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Venue (optional)
          </label>
          <input
            type="text"
            value={venue}
            onChange={(e) => setVenue(e.target.value)}
            placeholder="Stadium name"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Competition (optional)
          </label>
          <input
            type="text"
            value={competition}
            onChange={(e) => setCompetition(e.target.value)}
            placeholder="League / Cup"
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      {/* Processing Mode */}
      {videoId && (
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
            <Settings className="w-4 h-4" />
            Processing Mode
          </label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="mode"
                value="post_match"
                checked={processingMode === 'post_match'}
                onChange={() => setProcessingMode('post_match')}
                className="text-blue-500"
              />
              <span className="text-sm text-slate-600">
                Post-match (full analysis, CPU overnight)
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="mode"
                value="live"
                checked={processingMode === 'live'}
                onChange={() => setProcessingMode('live')}
                className="text-blue-500"
              />
              <span className="text-sm text-slate-600">
                Live (cloud GPU, real-time)
              </span>
            </label>
          </div>
          {processingMode === 'live' && (
            <p className="text-xs text-amber-600">
              Note: Live mode requires cloud GPU setup and incurs costs (~$0.50-1.50/hour)
            </p>
          )}
        </div>
      )}

      {/* Create Button */}
      <button
        onClick={handleCreate}
        disabled={isCreating || !homeTeam.name || !awayTeam.name}
        className={`
          w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg
          font-medium transition-colors
          ${isCreating || !homeTeam.name || !awayTeam.name
            ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
            : 'bg-blue-500 text-white hover:bg-blue-600'}
        `}
      >
        {isCreating ? (
          <>
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Creating...
          </>
        ) : (
          <>
            <Play className="w-5 h-5" />
            Start Analysis
          </>
        )}
      </button>
    </div>
  );
}

function hexToRgb(hex: string): number[] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16),
      ]
    : [128, 128, 128];
}
