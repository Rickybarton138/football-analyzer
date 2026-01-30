import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import LiveCoaching from './components/LiveCoaching';

// Types
interface MatchAnalysis {
  video_path: string;
  duration_seconds: number;
  total_frames: number;
  analyzed_frames: number;
  fps_analyzed: number;
  avg_home_players: number;
  avg_away_players: number;
  frames: FrameData[];
}

interface FrameData {
  frame_number: number;
  timestamp: number;
  player_count: number;
  home_players: number;
  away_players: number;
  ball_position: [number, number] | null;
  ball_pitch_x?: number;  // Homography-transformed pitch X (0-100, along length)
  ball_pitch_y?: number;  // Homography-transformed pitch Y (0-100, along width)
  detections: Detection[];
}

interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
  team: 'home' | 'away' | 'referee' | 'unknown';
  pitch_x?: number;  // Homography-transformed pitch X (0-100, along length)
  pitch_y?: number;  // Homography-transformed pitch Y (0-100, along width)
}

// Simplified tab structure - consolidated from 13 to 5 main views + training + live
type Tab = 'video' | 'analysis' | 'aicoach' | 'players' | 'tactical' | 'training' | 'live' | 'jersey' | 'radar' | 'spotlight';

// Shot detection types
interface DetectedShot {
  timestamp: number;
  frame_number: number;
  position: [number, number];
  team: 'home' | 'away';
  on_target: boolean;
  distance_estimate: number; // pixels from goal
}

// Player stats types
interface PlayerStats {
  player_name: string;
  total_clips: number;
  total_play_time_seconds: number;
  attacking: {
    ball_touches: number;
    passes_attempted: number;
    passes_completed: number;
    pass_accuracy: number;
    shots: number;
    shots_on_target: number;
    shot_accuracy: number;
  };
  defensive: {
    tackles_attempted: number;
    tackles_won: number;
    tackle_success_rate: number;
    headers: number;
    interceptions: number;
  };
  physical: {
    distance_covered_pixels: number;
    distance_covered_meters_estimate: number;
    sprints: number;
  };
}

interface PlayerListItem {
  name: string;
  clips_analyzed: number;
  play_time_seconds: number;
  ball_touches: number;
}

const VIDEO_WIDTH = 1920;
const VIDEO_HEIGHT = 1080;

// Match metadata for upload
interface MatchMetadata {
  homeTeam: string;
  awayTeam: string;
  isHomeTeam: boolean; // Are we filming our team at home?
  homeJerseyColor: string;
  awayJerseyColor: string;
  homeFormation: string;
  awayFormation: string;
  matchDate: string;
  competition: string;
  venue: string;
}

// System status type
interface SystemStatus {
  gpu: { available: boolean; name: string | null; memory_gb: number | null };
  cpu: { cores: number; threads: number; name: string };
  memory: { total_gb: number; available_gb: number };
  processing_estimates: {
    quick_preview: { description: string; cpu_minutes: number; gpu_minutes: number };
    standard: { description: string; cpu_minutes: number; gpu_minutes: number };
    full: { description: string; cpu_minutes: number; gpu_minutes: number };
  };
  recommended_mode: string;
  recommendation: string;
}

// Processing mode type
type ProcessingMode = 'quick_preview' | 'standard' | 'full';

// Video Upload View Component
function VideoUploadView({ onUploadComplete, onSkip, error, hasExistingAnalysis }: {
  onUploadComplete: () => void;
  onSkip: () => void;
  error: string | null;
  hasExistingAnalysis: boolean;
}) {
  const [step, setStep] = useState<'metadata' | 'upload' | 'processing'>('metadata');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  const [videoId, setVideoId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('quick_preview');

  // Fetch system status on mount
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status');
        if (response.ok) {
          const data = await response.json();
          setSystemStatus(data);
          // Set recommended mode
          if (data.recommended_mode === 'full' && data.gpu?.available) {
            setProcessingMode('full');
          } else {
            setProcessingMode('quick_preview');
          }
        }
      } catch (err) {
        console.error('Failed to fetch system status:', err);
      }
    };
    fetchStatus();
  }, []);

  const [metadata, setMetadata] = useState<MatchMetadata>({
    homeTeam: '',
    awayTeam: '',
    isHomeTeam: true,
    homeJerseyColor: '#dc2626', // Red
    awayJerseyColor: '#3b82f6', // Blue
    homeFormation: '4-4-2',
    awayFormation: '4-3-3',
    matchDate: new Date().toISOString().split('T')[0],
    competition: '',
    venue: ''
  });

  const formations = ['4-4-2', '4-3-3', '4-2-3-1', '3-5-2', '3-4-3', '5-3-2', '5-4-1', '4-5-1', '4-1-4-1'];

  const jerseyColors = [
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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleMetadataSubmit = () => {
    if (!metadata.homeTeam || !metadata.awayTeam) {
      alert('Please enter both team names');
      return;
    }
    setStep('upload');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a video file');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      // Create form data with file and metadata
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('metadata', JSON.stringify(metadata));

      // Upload the video
      const xhr = new XMLHttpRequest();
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          setUploadProgress(Math.round((e.loaded / e.total) * 100));
        }
      });

      const uploadPromise = new Promise<{ video_id: string }>((resolve, reject) => {
        xhr.onload = () => {
          if (xhr.status === 200) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error('Upload failed'));
          }
        };
        xhr.onerror = () => reject(new Error('Upload failed'));
      });

      xhr.open('POST', 'http://localhost:8000/api/video/upload');
      xhr.send(formData);

      const result = await uploadPromise;
      setVideoId(result.video_id);

      // Start processing
      setStep('processing');
      setUploading(false);

      // Trigger video processing with selected mode
      await fetch(`http://localhost:8000/api/video/${result.video_id}/process?analysis_mode=${processingMode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metadata })
      });

      // Poll for processing status
      pollProcessingStatus(result.video_id);

    } catch (err) {
      console.error('Upload error:', err);
      alert('Upload failed. Please try again.');
      setUploading(false);
    }
  };

  const pollProcessingStatus = async (id: string) => {
    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/video/${id}/status`);
        const status = await response.json();

        // Backend uses progress_pct, not progress
        setProcessingProgress(status.progress_pct || status.progress || 0);
        setProcessingStatus(status.status || 'Processing...');

        // Backend uses 'completed' not 'complete'
        if (status.status === 'completed' || status.status === 'complete') {
          // Processing complete, reload analysis
          setTimeout(() => onUploadComplete(), 1000);
        } else if (status.status === 'failed' || status.status === 'error') {
          alert('Processing failed: ' + (status.error_message || status.error || 'Unknown error'));
          setStep('upload');
        } else {
          // Continue polling
          setTimeout(poll, 2000);
        }
      } catch (err) {
        console.error('Status poll error:', err);
        setTimeout(poll, 3000);
      }
    };
    poll();
  };

  // Metadata Form
  if (step === 'metadata') {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center p-4">
        <div className="bg-[#111827] rounded-2xl p-8 max-w-2xl w-full border border-slate-700/50">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-2xl font-bold">V</span>
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">Match Setup</h1>
            <p className="text-slate-400">Enter match details for accurate analysis</p>
          </div>

          <div className="space-y-6">
            {/* Team Names */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Home Team</label>
                <input
                  type="text"
                  value={metadata.homeTeam}
                  onChange={(e) => setMetadata({ ...metadata, homeTeam: e.target.value })}
                  placeholder="e.g. Manchester United"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Away Team</label>
                <input
                  type="text"
                  value={metadata.awayTeam}
                  onChange={(e) => setMetadata({ ...metadata, awayTeam: e.target.value })}
                  placeholder="e.g. Liverpool"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                />
              </div>
            </div>

            {/* Which team are we? */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Which team are you filming?</label>
              <div className="flex gap-4">
                <button
                  onClick={() => setMetadata({ ...metadata, isHomeTeam: true })}
                  className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
                    metadata.isHomeTeam
                      ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                      : 'border-slate-600 text-slate-400 hover:border-slate-500'
                  }`}
                >
                  Home Team (We're at home)
                </button>
                <button
                  onClick={() => setMetadata({ ...metadata, isHomeTeam: false })}
                  className={`flex-1 py-3 px-4 rounded-lg border-2 transition-all ${
                    !metadata.isHomeTeam
                      ? 'border-cyan-500 bg-cyan-500/10 text-cyan-400'
                      : 'border-slate-600 text-slate-400 hover:border-slate-500'
                  }`}
                >
                  Away Team (We're away)
                </button>
              </div>
            </div>

            {/* Jersey Colors */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Home Jersey Color</label>
                <div className="flex flex-wrap gap-2">
                  {jerseyColors.map(color => (
                    <button
                      key={color.value}
                      onClick={() => setMetadata({ ...metadata, homeJerseyColor: color.value })}
                      className={`w-8 h-8 rounded-full border-2 transition-all ${
                        metadata.homeJerseyColor === color.value ? 'border-white scale-110' : 'border-transparent'
                      }`}
                      style={{ backgroundColor: color.value }}
                      title={color.name}
                    />
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Away Jersey Color</label>
                <div className="flex flex-wrap gap-2">
                  {jerseyColors.map(color => (
                    <button
                      key={color.value}
                      onClick={() => setMetadata({ ...metadata, awayJerseyColor: color.value })}
                      className={`w-8 h-8 rounded-full border-2 transition-all ${
                        metadata.awayJerseyColor === color.value ? 'border-white scale-110' : 'border-transparent'
                      }`}
                      style={{ backgroundColor: color.value }}
                      title={color.name}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Formations */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Home Formation</label>
                <select
                  value={metadata.homeFormation}
                  onChange={(e) => setMetadata({ ...metadata, homeFormation: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                >
                  {formations.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Away Formation</label>
                <select
                  value={metadata.awayFormation}
                  onChange={(e) => setMetadata({ ...metadata, awayFormation: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                >
                  {formations.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
              </div>
            </div>

            {/* Optional Details */}
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Match Date</label>
                <input
                  type="date"
                  value={metadata.matchDate}
                  onChange={(e) => setMetadata({ ...metadata, matchDate: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Competition</label>
                <input
                  type="text"
                  value={metadata.competition}
                  onChange={(e) => setMetadata({ ...metadata, competition: e.target.value })}
                  placeholder="e.g. League Cup"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Venue</label>
                <input
                  type="text"
                  value={metadata.venue}
                  onChange={(e) => setMetadata({ ...metadata, venue: e.target.value })}
                  placeholder="e.g. Old Trafford"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
                />
              </div>
            </div>

            <button
              onClick={handleMetadataSubmit}
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all"
            >
              Continue to Upload →
            </button>

            {/* Skip option to view existing analysis */}
            <button
              onClick={onSkip}
              className="w-full mt-3 py-3 text-slate-400 hover:text-white transition-colors text-sm"
            >
              {hasExistingAnalysis ? 'Skip - View Previous Analysis →' : 'Skip - Go to Dashboard →'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Upload View
  if (step === 'upload') {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center p-4">
        <div className="bg-[#111827] rounded-2xl p-8 max-w-xl w-full border border-slate-700/50">
          <button
            onClick={() => setStep('metadata')}
            className="text-slate-400 hover:text-white mb-4 flex items-center gap-2"
          >
            ← Back to Match Setup
          </button>

          <div className="text-center mb-8">
            <h2 className="text-xl font-bold text-white mb-2">Upload Match Video</h2>
            <p className="text-slate-400 text-sm">
              {metadata.homeTeam} vs {metadata.awayTeam}
            </p>
          </div>

          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
              selectedFile
                ? 'border-cyan-500 bg-cyan-500/5'
                : 'border-slate-600 hover:border-slate-500'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              className="hidden"
            />
            {selectedFile ? (
              <div>
                <div className="w-16 h-16 bg-cyan-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-white font-medium">{selectedFile.name}</p>
                <p className="text-slate-400 text-sm mt-1">
                  {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                </p>
                <p className="text-cyan-400 text-sm mt-2">Click to change file</p>
              </div>
            ) : (
              <div>
                <div className="w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <p className="text-white font-medium">Drop video file here</p>
                <p className="text-slate-400 text-sm mt-1">or click to browse</p>
                <p className="text-slate-500 text-xs mt-4">Supports MP4, MOV, AVI up to 10GB</p>
              </div>
            )}
          </div>

          {/* Processing Mode Selection */}
          {selectedFile && (
            <div className="mt-6">
              <label className="block text-sm font-medium text-slate-300 mb-3">Processing Mode</label>

              {/* System Status Banner */}
              {systemStatus && (
                <div className={`mb-4 p-3 rounded-lg ${systemStatus.gpu.available ? 'bg-green-500/10 border border-green-500/30' : 'bg-amber-500/10 border border-amber-500/30'}`}>
                  <div className="flex items-center gap-2">
                    {systemStatus.gpu.available ? (
                      <>
                        <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="text-green-400 text-sm font-medium">GPU Detected: {systemStatus.gpu.name}</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <span className="text-amber-400 text-sm font-medium">CPU Only - {systemStatus.cpu.name}</span>
                      </>
                    )}
                  </div>
                  <p className="text-slate-400 text-xs mt-1">{systemStatus.recommendation}</p>
                </div>
              )}

              {/* Mode Cards */}
              <div className="space-y-3">
                {/* Quick Preview */}
                <button
                  onClick={() => setProcessingMode('quick_preview')}
                  className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
                    processingMode === 'quick_preview'
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-slate-600 hover:border-slate-500'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">Quick Preview</span>
                        {systemStatus && !systemStatus.gpu?.available && (
                          <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded">Recommended</span>
                        )}
                      </div>
                      <p className="text-slate-400 text-sm mt-1">Sample every 30 seconds for a rapid overview</p>
                    </div>
                    <div className="text-right">
                      <div className="text-cyan-400 text-sm font-medium">
                        {systemStatus?.gpu?.available
                          ? `~${systemStatus?.processing_estimates?.quick_preview?.gpu_minutes || 2} min`
                          : `~${systemStatus?.processing_estimates?.quick_preview?.cpu_minutes || 6} min`
                        }
                      </div>
                    </div>
                  </div>
                </button>

                {/* Standard */}
                <button
                  onClick={() => setProcessingMode('standard')}
                  className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
                    processingMode === 'standard'
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-slate-600 hover:border-slate-500'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">Standard Analysis</span>
                      </div>
                      <p className="text-slate-400 text-sm mt-1">Good balance of speed and detail (1 FPS)</p>
                    </div>
                    <div className="text-right">
                      <div className="text-slate-300 text-sm font-medium">
                        {systemStatus?.gpu?.available
                          ? `~${systemStatus?.processing_estimates?.standard?.gpu_minutes || 45} min`
                          : `~${Math.round((systemStatus?.processing_estimates?.standard?.cpu_minutes || 180) / 60)} hrs`
                        }
                      </div>
                    </div>
                  </div>
                </button>

                {/* Full Analysis */}
                <button
                  onClick={() => setProcessingMode('full')}
                  className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
                    processingMode === 'full'
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-slate-600 hover:border-slate-500'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">Full Analysis</span>
                        {systemStatus?.gpu?.available && (
                          <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded">Recommended</span>
                        )}
                      </div>
                      <p className="text-slate-400 text-sm mt-1">Maximum accuracy at 3 FPS - best for detailed review</p>
                    </div>
                    <div className="text-right">
                      <div className="text-slate-300 text-sm font-medium">
                        {systemStatus?.gpu?.available
                          ? `~${Math.round((systemStatus?.processing_estimates?.full?.gpu_minutes || 120) / 60)} hrs`
                          : `~${Math.round((systemStatus?.processing_estimates?.full?.cpu_minutes || 540) / 60)} hrs`
                        }
                      </div>
                      {!systemStatus?.gpu?.available && (
                        <div className="text-amber-400 text-xs mt-1">Overnight processing recommended</div>
                      )}
                    </div>
                  </div>
                </button>
              </div>
            </div>
          )}

          {uploading && (
            <div className="mt-6">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">Uploading...</span>
                <span className="text-cyan-400">{uploadProgress}%</span>
              </div>
              <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className={`w-full mt-6 py-4 font-semibold rounded-lg transition-all ${
              selectedFile && !uploading
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-600 hover:to-blue-700'
                : 'bg-slate-700 text-slate-400 cursor-not-allowed'
            }`}
          >
            {uploading ? 'Uploading...' : `Upload & Start ${processingMode === 'quick_preview' ? 'Quick Preview' : processingMode === 'standard' ? 'Standard' : 'Full'} Analysis`}
          </button>
        </div>
      </div>
    );
  }

  // Processing View
  return (
    <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center p-4">
      <div className="bg-[#111827] rounded-2xl p-8 max-w-xl w-full border border-slate-700/50 text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-6">
          <div className="w-12 h-12 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
        </div>

        <h2 className="text-xl font-bold text-white mb-2">Analyzing Match</h2>
        <p className="text-slate-400 mb-6">
          {metadata.homeTeam} vs {metadata.awayTeam}
        </p>

        <div className="mb-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-400">{processingStatus || 'Processing video...'}</span>
            <span className="text-cyan-400">{processingProgress.toFixed(1)}%</span>
          </div>
          <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-500"
              style={{ width: `${processingProgress}%` }}
            />
          </div>
        </div>

        <div className="text-slate-500 text-sm space-y-1">
          <p>• Detecting players and ball</p>
          <p>• Reading jersey numbers</p>
          <p>• Tracking movements</p>
          <p>• Calculating statistics</p>
        </div>
      </div>
    </div>
  );
}

// Available match info from backend
interface AvailableMatch {
  id: string;
  name: string;
  date: string;
  duration: string;
  status: 'ready' | 'processing' | 'failed';
}

function App() {
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('video');
  const [loading, setLoading] = useState(false); // Don't auto-load on startup
  const [error, setError] = useState<string | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<'full' | '1st' | '2nd'>('full');
  const [showUpload, setShowUpload] = useState(false);
  const [availableMatches, setAvailableMatches] = useState<AvailableMatch[]>([]);
  const [loadingMatches, setLoadingMatches] = useState(true);

  // Load available matches on startup
  useEffect(() => {
    loadAvailableMatches();
  }, []);

  const loadAvailableMatches = async () => {
    setLoadingMatches(true);
    try {
      // Try to get list of available analyses
      const response = await fetch('http://localhost:8000/api/matches/list');
      if (response.ok) {
        const data = await response.json();
        setAvailableMatches(data.matches || []);
      }
    } catch (err) {
      console.log('No matches list endpoint available');
    }

    // Also check if there's a current analysis available
    try {
      const response = await fetch('http://localhost:8000/api/video/full-analysis');
      if (response.ok) {
        const data = await response.json();
        if (data && (data.frame_analyses || data.frames)) {
          // There's a current analysis, add it to available matches if not already there
          const currentMatch: AvailableMatch = {
            id: 'current',
            name: data.video_path ? data.video_path.split(/[/\\]/).pop() || 'Current Analysis' : 'Current Analysis',
            date: data.start_time || new Date().toISOString().split('T')[0],
            duration: data.duration_seconds ? formatDuration(data.duration_seconds) : 'Unknown',
            status: 'ready'
          };
          setAvailableMatches(prev => {
            if (!prev.find(m => m.id === 'current')) {
              return [currentMatch, ...prev];
            }
            return prev;
          });
        }
      }
    } catch (err) {
      // No current analysis available
    }
    setLoadingMatches(false);
  };

  const loadAnalysis = async (matchId?: string) => {
    setLoading(true);
    setError(null);
    try {
      // Load the full analysis via API endpoint
      const url = matchId && matchId !== 'current'
        ? `http://localhost:8000/api/matches/${matchId}/analysis`
        : 'http://localhost:8000/api/video/full-analysis';

      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to load analysis file');
      const data = await response.json();

      // Transform to expected format if needed
      if (data.frame_analyses) {
        // Convert frame_analyses to frames format
        const transformedData: MatchAnalysis = {
          video_path: data.video_path || '',
          duration_seconds: data.duration_seconds || 0,
          total_frames: data.total_frames || data.frame_analyses.length,
          analyzed_frames: data.analyzed_frames || data.frame_analyses.length,
          fps_analyzed: data.fps_analyzed || 3,
          avg_home_players: data.avg_home_players || 0,
          avg_away_players: data.avg_away_players || 0,
          frames: data.frame_analyses.map((f: any) => ({
            frame_number: f.frame_number,
            timestamp: f.timestamp,
            player_count: f.player_count,
            home_players: f.home_players,
            away_players: f.away_players,
            ball_position: f.ball_position,
            detections: f.detections || []
          }))
        };
        setAnalysis(transformedData);
      } else if (data.frames) {
        // Already in correct format
        setAnalysis(data);
      } else {
        throw new Error('Invalid analysis format');
      }
      setLoading(false);
    } catch (err) {
      console.error('Failed to load analysis:', err);
      setError('Could not load match analysis.');
      setLoading(false);
    }
  };

  // Calculate all stats
  const stats = useMemo(() => {
    if (!analysis) return null;
    return calculateMatchStats(analysis, selectedPeriod);
  }, [analysis, selectedPeriod]);

  // Show upload view if requested
  if (showUpload) {
    return (
      <VideoUploadView
        onUploadComplete={() => {
          setShowUpload(false);
          loadAvailableMatches();
          loadAnalysis();
        }}
        onSkip={() => setShowUpload(false)}
        error={error}
        hasExistingAnalysis={!!analysis}
      />
    );
  }

  // Show loading spinner when loading analysis
  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-3 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400 text-sm">Loading match analysis...</p>
        </div>
      </div>
    );
  }

  // Show home dashboard when no analysis is selected
  if (!analysis || !stats) {
    return (
      <div className="min-h-screen bg-[#0a0f1a]">
        {/* Header */}
        <header className="bg-[#111827] border-b border-slate-700/50">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center">
                  <span className="text-white text-lg font-bold">V</span>
                </div>
                <div>
                  <h1 className="text-lg font-semibold text-white">Football Analyzer</h1>
                  <p className="text-slate-400 text-xs">AI-Powered Match Analysis</p>
                </div>
              </div>
              <button
                onClick={() => setShowUpload(true)}
                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-medium rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Match
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          {/* Welcome Section */}
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-3">Welcome to Football Analyzer</h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Upload match footage and get AI-powered insights including player tracking,
              tactical analysis, heatmaps, and coaching recommendations.
            </p>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <button
              onClick={() => setShowUpload(true)}
              className="bg-gradient-to-br from-cyan-500/10 to-blue-600/10 border border-cyan-500/30 rounded-xl p-6 text-left hover:border-cyan-500/50 transition-all group"
            >
              <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center mb-4 group-hover:bg-cyan-500/30 transition-colors">
                <svg className="w-6 h-6 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Upload New Match</h3>
              <p className="text-slate-400 text-sm">Start analyzing a new match video with AI</p>
            </button>

            <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-6 text-left">
              <div className="w-12 h-12 bg-slate-700/50 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Match Analysis</h3>
              <p className="text-slate-400 text-sm">View possession, passes, shots, and more</p>
            </div>

            <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-6 text-left">
              <div className="w-12 h-12 bg-slate-700/50 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">AI Coach</h3>
              <p className="text-slate-400 text-sm">Get tactical insights and recommendations</p>
            </div>
          </div>

          {/* Recent Matches */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">Recent Matches</h3>

            {loadingMatches ? (
              <div className="text-center py-12">
                <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                <p className="text-slate-400 text-sm">Loading matches...</p>
              </div>
            ) : availableMatches.length > 0 ? (
              <div className="grid gap-4">
                {availableMatches.map(match => (
                  <button
                    key={match.id}
                    onClick={() => loadAnalysis(match.id)}
                    className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 text-left hover:border-cyan-500/30 hover:bg-slate-800/70 transition-all flex items-center gap-4"
                  >
                    <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-white font-medium truncate">{match.name}</h4>
                      <p className="text-slate-400 text-sm">{match.date} • {match.duration}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {match.status === 'ready' && (
                        <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">Ready</span>
                      )}
                      {match.status === 'processing' && (
                        <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 text-xs rounded-full">Processing</span>
                      )}
                      {match.status === 'failed' && (
                        <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded-full">Failed</span>
                      )}
                      <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 bg-slate-800/30 rounded-xl border border-slate-700/30">
                <div className="w-16 h-16 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                <h4 className="text-white font-medium mb-2">No matches yet</h4>
                <p className="text-slate-400 text-sm mb-4">Upload your first match video to get started</p>
                <button
                  onClick={() => setShowUpload(true)}
                  className="px-4 py-2 bg-cyan-500/20 text-cyan-400 font-medium rounded-lg hover:bg-cyan-500/30 transition-colors"
                >
                  Upload Match
                </button>
              </div>
            )}
          </div>
        </main>
      </div>
    );
  }

  // Simplified navigation - 6 main tabs + live coaching
  const tabs: { id: Tab; label: string; icon: string; description: string }[] = [
    { id: 'video', label: 'Match Video', icon: '🎬', description: 'Video with tracking overlays' },
    { id: 'analysis', label: 'Match Analysis', icon: '📊', description: 'Stats, heatmaps & formations' },
    { id: 'aicoach', label: 'AI Coach', icon: '🧠', description: 'Tactical insights & recommendations' },
    { id: 'tactical', label: 'Tactical Events', icon: '⚡', description: 'Key moments & events timeline' },
    { id: 'players', label: 'Players', icon: '👤', description: 'Individual player statistics' },
    { id: 'training', label: 'Training Data', icon: '🎯', description: 'Annotate frames for ML training' },
    { id: 'live', label: 'Live Coaching', icon: '📡', description: 'Real-time game management' },
    { id: 'jersey', label: 'Jersey Detection', icon: '🔢', description: 'AI-powered jersey number detection' },
    { id: 'radar', label: '2D Radar', icon: '📍', description: 'Live tactical overhead view' },
    { id: 'spotlight', label: 'Player Spotlight', icon: '🌟', description: 'Individual player moments & clips' },
  ];

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      {/* VEO-style Header */}
      <header className="bg-[#111827] border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setAnalysis(null)}
                className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center hover:from-cyan-600 hover:to-blue-700 transition-all"
                title="Back to Library"
              >
                <span className="text-white text-lg font-bold">V</span>
              </button>
              <div>
                <button
                  onClick={() => setAnalysis(null)}
                  className="text-slate-400 hover:text-cyan-400 text-xs flex items-center gap-1 mb-0.5 transition-colors"
                >
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  Back to Library
                </button>
                <h1 className="text-lg font-semibold text-white">{analysis.video_path?.split(/[/\\]/).pop() || 'Match Analysis'}</h1>
                <p className="text-slate-400 text-xs">{formatDuration(analysis.duration_seconds)} duration</p>
              </div>
            </div>

            {/* Period Selector */}
            <div className="flex items-center gap-2 bg-slate-800/50 rounded-lg p-1">
              {(['full', '1st', '2nd'] as const).map(period => (
                <button
                  key={period}
                  onClick={() => setSelectedPeriod(period)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                    selectedPeriod === period
                      ? 'bg-cyan-500 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {period === 'full' ? 'Full Match' : `${period} Half`}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowUpload(true)}
                className="px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-white border border-slate-600 hover:border-slate-500 rounded-lg transition-all flex items-center gap-1.5"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Match
              </button>
              <ExportButton />
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex gap-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium transition-all border-b-2 ${
                  activeTab === tab.id
                    ? 'text-cyan-400 border-cyan-400'
                    : 'text-slate-400 border-transparent hover:text-slate-200'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === 'video' && <VideoPlayerWithTracking analysis={analysis} />}
        {activeTab === 'analysis' && <CombinedAnalysisView analysis={analysis} stats={stats} />}
        {activeTab === 'aicoach' && <AICoachView />}
        {activeTab === 'tactical' && <CombinedTacticalView analysis={analysis} />}
        {activeTab === 'players' && <PlayersView />}
        {activeTab === 'training' && <AnnotationUIView />}
        {activeTab === 'live' && <LiveCoaching />}
        {activeTab === 'jersey' && <JerseyDetectionView />}
        {activeTab === 'radar' && <Radar2DView analysis={analysis} />}
        {activeTab === 'spotlight' && <PlayerSpotlightView />}
      </main>
    </div>
  );
}

// ==================== VIDEO PLAYER VIEW ====================
function VideoPlayerView({ analysis }: { analysis: MatchAnalysis }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [events, setEvents] = useState<any[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(true);

  // Load tactical events
  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/analytics/tactical-events');
      if (response.ok) {
        const data = await response.json();
        setEvents(data.tactical_events?.events || []);
      }
    } catch (err) {
      console.error('Failed to load events:', err);
    } finally {
      setLoadingEvents(false);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const skip = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime + seconds);
    }
  };

  // Get current frame data based on video time
  const currentFrame = useMemo(() => {
    if (!analysis.frames || analysis.frames.length === 0) return null;
    const frameIndex = Math.floor(currentTime * (analysis.fps_analyzed || 3));
    return analysis.frames[Math.min(frameIndex, analysis.frames.length - 1)];
  }, [analysis, currentTime]);

  // Get nearby events
  const nearbyEvents = useMemo(() => {
    return events
      .filter(e => Math.abs(e.timestamp - currentTime) < 30)
      .sort((a, b) => a.timestamp - b.timestamp);
  }, [events, currentTime]);

  const eventTypeColors: Record<string, string> = {
    pressing_trigger: 'bg-yellow-500',
    dangerous_attack: 'bg-red-500',
    counter_attack: 'bg-orange-500',
    shape_warning: 'bg-purple-500',
    high_line_opportunity: 'bg-blue-500',
    overload: 'bg-cyan-500',
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-white">Match Video</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Player */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-[#111827] rounded-2xl overflow-hidden border border-slate-700/30">
            <video
              ref={videoRef}
              className="w-full aspect-video bg-black"
              src="http://127.0.0.1:8000/api/video/stream"
              onTimeUpdate={handleTimeUpdate}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            >
              Your browser does not support video playback.
            </video>

            {/* Video Controls */}
            <div className="p-4 bg-slate-800/50">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => skip(-10)}
                  className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors"
                  title="Back 10s"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.333 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
                  </svg>
                </button>

                <button
                  onClick={togglePlay}
                  className="p-3 rounded-full bg-cyan-500 hover:bg-cyan-600 text-white transition-colors"
                >
                  {isPlaying ? (
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  ) : (
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                </button>

                <button
                  onClick={() => skip(10)}
                  className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors"
                  title="Forward 10s"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z" />
                  </svg>
                </button>

                <div className="flex-1 mx-4">
                  <input
                    type="range"
                    min="0"
                    max={analysis.duration_seconds || 0}
                    value={currentTime}
                    onChange={(e) => handleSeek(Number(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                  />
                </div>

                <span className="text-white font-mono text-sm">
                  {formatTime(currentTime)} / {formatTime(analysis.duration_seconds)}
                </span>
              </div>
            </div>
          </div>

          {/* Current Frame Stats */}
          {currentFrame && (
            <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-red-400">{currentFrame.home_players}</div>
                  <div className="text-slate-400 text-sm">Your Players</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-cyan-400">{currentFrame.player_count}</div>
                  <div className="text-slate-400 text-sm">Total Players</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-slate-300">{currentFrame.away_players}</div>
                  <div className="text-slate-400 text-sm">Opposition</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Events Timeline */}
        <div className="space-y-4">
          <div className="bg-[#111827] rounded-2xl p-4 border border-slate-700/30">
            <h3 className="text-white font-medium mb-3">Nearby Events</h3>

            {loadingEvents ? (
              <div className="text-center py-4">
                <div className="w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
              </div>
            ) : nearbyEvents.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {nearbyEvents.map((event, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSeek(event.timestamp)}
                    className={`w-full text-left p-2 rounded-lg transition-colors ${
                      Math.abs(event.timestamp - currentTime) < 5
                        ? 'bg-cyan-500/20 border border-cyan-500/50'
                        : 'bg-slate-800/50 hover:bg-slate-700/50'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${eventTypeColors[event.type] || 'bg-slate-500'}`}></div>
                      <span className="text-slate-400 text-xs">{formatTime(event.timestamp)}</span>
                    </div>
                    <p className="text-white text-sm mt-1 truncate">{event.description}</p>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-slate-400 text-sm text-center py-4">No events near current time</p>
            )}
          </div>

          {/* Quick Jump */}
          <div className="bg-[#111827] rounded-2xl p-4 border border-slate-700/30">
            <h3 className="text-white font-medium mb-3">Quick Jump</h3>
            <div className="grid grid-cols-3 gap-2">
              {[0, 5, 10, 15, 20, 25].map(mins => (
                <button
                  key={mins}
                  onClick={() => handleSeek(mins * 60)}
                  className="px-3 py-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg text-white text-sm transition-colors"
                >
                  {mins}'
                </button>
              ))}
            </div>
          </div>

          {/* Key Moments */}
          <div className="bg-[#111827] rounded-2xl p-4 border border-slate-700/30">
            <h3 className="text-white font-medium mb-3">Key Moments</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {events
                .filter(e => e.priority >= 3)
                .slice(0, 10)
                .map((event, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSeek(event.timestamp)}
                    className="w-full text-left p-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-cyan-400 text-xs">{formatTime(event.timestamp)}</span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        event.priority === 4 ? 'bg-red-500/20 text-red-400' : 'bg-orange-500/20 text-orange-400'
                      }`}>
                        {event.priority_name}
                      </span>
                    </div>
                    <p className="text-white text-sm mt-1 truncate">{event.description}</p>
                  </button>
                ))}
              {events.filter(e => e.priority >= 3).length === 0 && (
                <p className="text-slate-400 text-sm text-center py-2">No critical events</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ==================== VEO-STYLE VIDEO PLAYER WITH TRACKING ====================
function VideoPlayerWithTracking({ analysis }: { analysis: MatchAnalysis }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showOverlays, setShowOverlays] = useState(true);
  const [overlayMode, setOverlayMode] = useState<'tracking' | 'heatmap' | 'lines'>('tracking');
  const [events, setEvents] = useState<any[]>([]);
  const [videoError, setVideoError] = useState(false);

  // Load tactical events
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/analytics/tactical-events')
      .then(res => res.ok ? res.json() : null)
      .then(data => setEvents(data?.tactical_events?.events || []))
      .catch(() => {});
  }, []);

  // Get current frame based on video time
  const currentFrame = useMemo(() => {
    if (!analysis.frames || analysis.frames.length === 0) return null;
    const frameIndex = Math.floor(currentTime * (analysis.fps_analyzed || 3));
    return analysis.frames[Math.min(frameIndex, analysis.frames.length - 1)];
  }, [analysis, currentTime]);

  // Draw tracking overlays on canvas
  useEffect(() => {
    if (!canvasRef.current || !currentFrame || !showOverlays) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const canvas = canvasRef.current;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (overlayMode === 'tracking' && currentFrame.detections) {
      currentFrame.detections.forEach((det, idx) => {
        const [x1, y1, x2, y2] = det.bbox;
        // Scale to canvas size
        const scaleX = canvas.width / 1920;
        const scaleY = canvas.height / 1080;

        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;

        // Draw player marker (VEO style - circle at feet)
        const centerX = sx1 + sw / 2;
        const bottomY = sy1 + sh;

        // Team color
        const color = det.team === 'home' ? '#ef4444' : det.team === 'away' ? '#3b82f6' : '#94a3b8';

        // Outer glow
        ctx.beginPath();
        ctx.arc(centerX, bottomY - 5, 12, 0, Math.PI * 2);
        ctx.fillStyle = color + '40';
        ctx.fill();

        // Inner circle
        ctx.beginPath();
        ctx.arc(centerX, bottomY - 5, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Player number/ID
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(idx + 1), centerX, bottomY - 2);
      });

      // Draw ball if detected
      if (currentFrame.ball_position) {
        const [bx, by] = currentFrame.ball_position;
        const scaleX = canvas.width / 1920;
        const scaleY = canvas.height / 1080;

        ctx.beginPath();
        ctx.arc(bx * scaleX, by * scaleY, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [currentFrame, showOverlays, overlayMode]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const skip = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime + seconds);
    }
  };

  // Get nearby events
  const nearbyEvents = useMemo(() => {
    return events
      .filter(e => Math.abs(e.timestamp - currentTime) < 60)
      .sort((a, b) => Math.abs(a.timestamp - currentTime) - Math.abs(b.timestamp - currentTime))
      .slice(0, 5);
  }, [events, currentTime]);

  const eventColors: Record<string, string> = {
    pressing_trigger: 'bg-yellow-500',
    dangerous_attack: 'bg-red-500',
    counter_attack: 'bg-orange-500',
    high_line_opportunity: 'bg-blue-500',
  };

  return (
    <div className="space-y-4">
      {/* Video Container */}
      <div className="bg-[#111827] rounded-2xl overflow-hidden border border-slate-700/30">
        <div className="relative aspect-video bg-black">
          {/* Video Element */}
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full object-contain"
            src="http://127.0.0.1:8000/api/video/stream"
            onTimeUpdate={handleTimeUpdate}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onError={() => setVideoError(true)}
            crossOrigin="anonymous"
          />

          {/* Canvas Overlay for Tracking */}
          {showOverlays && (
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
              width={1920}
              height={1080}
            />
          )}

          {/* Video Error State */}
          {videoError && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90">
              <div className="text-center">
                <div className="text-6xl mb-4">🎬</div>
                <h3 className="text-white font-semibold mb-2">Video Not Available</h3>
                <p className="text-slate-400 text-sm mb-4">Using analysis data for tracking visualization</p>
                <button
                  onClick={() => setVideoError(false)}
                  className="px-4 py-2 bg-cyan-500 text-white rounded-lg text-sm hover:bg-cyan-600"
                >
                  Retry
                </button>
              </div>
            </div>
          )}

          {/* Live Stats Overlay */}
          {currentFrame && !videoError && (
            <div className="absolute top-4 left-4 right-4 flex justify-between">
              <div className="bg-black/70 rounded-lg px-3 py-2 flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-red-500"></span>
                  <span className="text-white font-medium">{currentFrame.home_players}</span>
                </div>
                <span className="text-slate-400">vs</span>
                <div className="flex items-center gap-2">
                  <span className="text-white font-medium">{currentFrame.away_players}</span>
                  <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                </div>
              </div>
              <div className="bg-black/70 rounded-lg px-3 py-2">
                <span className="text-white font-mono">{formatTime(currentTime)}</span>
              </div>
            </div>
          )}

          {/* Overlay Controls */}
          <div className="absolute bottom-20 right-4 flex flex-col gap-2">
            <button
              onClick={() => setShowOverlays(!showOverlays)}
              className={`p-2 rounded-lg transition-colors ${showOverlays ? 'bg-cyan-500 text-white' : 'bg-black/70 text-slate-400'}`}
              title="Toggle overlays"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </button>
          </div>
        </div>

        {/* Video Controls Bar */}
        <div className="p-4 bg-slate-800/50">
          <div className="flex items-center gap-4">
            <button
              onClick={() => skip(-10)}
              className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0019 16V8a1 1 0 00-1.6-.8l-5.333 4zM4.066 11.2a1 1 0 000 1.6l5.334 4A1 1 0 0011 16V8a1 1 0 00-1.6-.8l-5.334 4z" />
              </svg>
            </button>

            <button
              onClick={togglePlay}
              className="p-3 rounded-full bg-cyan-500 hover:bg-cyan-600 text-white transition-colors"
            >
              {isPlaying ? (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>

            <button
              onClick={() => skip(10)}
              className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.933 12.8a1 1 0 000-1.6L6.6 7.2A1 1 0 005 8v8a1 1 0 001.6.8l5.333-4zM19.933 12.8a1 1 0 000-1.6l-5.333-4A1 1 0 0013 8v8a1 1 0 001.6.8l5.333-4z" />
              </svg>
            </button>

            {/* Progress Bar */}
            <div className="flex-1 mx-4">
              <div className="relative">
                <input
                  type="range"
                  min="0"
                  max={analysis.duration_seconds || 1}
                  value={currentTime}
                  onChange={(e) => handleSeek(Number(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
                {/* Event markers on timeline */}
                {events.filter(e => e.priority >= 3).slice(0, 20).map((event, idx) => (
                  <div
                    key={idx}
                    className="absolute top-0 w-1 h-2 bg-yellow-500 rounded-full cursor-pointer"
                    style={{ left: `${(event.timestamp / analysis.duration_seconds) * 100}%` }}
                    onClick={() => handleSeek(event.timestamp)}
                    title={event.description}
                  />
                ))}
              </div>
            </div>

            <span className="text-white font-mono text-sm min-w-[100px] text-right">
              {formatTime(currentTime)} / {formatTime(analysis.duration_seconds)}
            </span>
          </div>
        </div>
      </div>

      {/* Bottom Panel - 2D Pitch View + Events */}
      <div className="grid grid-cols-3 gap-4">
        {/* 2D Pitch Visualization - Aerial view from sideline camera */}
        <div className="col-span-2 bg-[#111827] rounded-2xl p-4 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
            <span className="text-green-400">⬛</span> Live Position Map
            <span className="text-slate-500 text-xs ml-2">(Aerial View - Home attacks up)</span>
          </h3>
          {/*
            VEO SIDELINE CAMERA TRANSFORMATION:
            - Camera is on the SIDE of the pitch near halfway line
            - Video X (0-1920) = Pitch LENGTH (left = one goal, right = other goal)
            - Video Y (0-1080) = Pitch WIDTH (top = far touchline, bottom = near touchline/camera)

            AERIAL VIEW MAPPING (looking down at pitch):
            - Video X -> Aerial Y (inverted: left of video = bottom of aerial = home goal)
            - Video Y -> Aerial X (far touchline top of video = left side of aerial)

            Home team defends LEFT of video, attacks RIGHT
            So in aerial view: Home defends BOTTOM, attacks TOP
          */}
          <div className="relative bg-gradient-to-b from-green-800/40 to-green-900/40 rounded-xl overflow-hidden border border-green-700/30" style={{ aspectRatio: '68/105' }}>
            {/* Pitch outline */}
            <div className="absolute inset-2 border-2 border-white/40 rounded-sm">
              {/* Center line - horizontal across pitch */}
              <div className="absolute left-0 right-0 top-1/2 h-px bg-white/40"></div>
              {/* Center circle */}
              <div className="absolute top-1/2 left-1/2 w-16 h-16 -translate-x-1/2 -translate-y-1/2 border-2 border-white/40 rounded-full"></div>
              {/* Center spot */}
              <div className="absolute top-1/2 left-1/2 w-2 h-2 -translate-x-1/2 -translate-y-1/2 bg-white/40 rounded-full"></div>

              {/* Top penalty area (opponent/away goal) */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2/5 h-[16%] border-b-2 border-l-2 border-r-2 border-white/40"></div>
              {/* Top goal box */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/5 h-[6%] border-b-2 border-l-2 border-r-2 border-white/40"></div>
              {/* Top penalty spot */}
              <div className="absolute top-[11%] left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-white/40 rounded-full"></div>

              {/* Bottom penalty area (home/your goal) */}
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-2/5 h-[16%] border-t-2 border-l-2 border-r-2 border-white/40"></div>
              {/* Bottom goal box */}
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/5 h-[6%] border-t-2 border-l-2 border-r-2 border-white/40"></div>
              {/* Bottom penalty spot */}
              <div className="absolute bottom-[11%] left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-white/40 rounded-full"></div>

              {/* Corner arcs */}
              <div className="absolute top-0 left-0 w-3 h-3 border-b-2 border-r-2 border-white/40 rounded-br-full"></div>
              <div className="absolute top-0 right-0 w-3 h-3 border-b-2 border-l-2 border-white/40 rounded-bl-full"></div>
              <div className="absolute bottom-0 left-0 w-3 h-3 border-t-2 border-r-2 border-white/40 rounded-tr-full"></div>
              <div className="absolute bottom-0 right-0 w-3 h-3 border-t-2 border-l-2 border-white/40 rounded-tl-full"></div>
            </div>

            {/* Player positions - using homography-transformed pitch coordinates from backend */}
            {currentFrame?.detections.map((det, idx) => {
              // Use pre-computed pitch coordinates from backend (with perspective correction)
              // pitch_x: 0-100 along pitch length (0=left goal, 100=right goal)
              // pitch_y: 0-100 along pitch width (0=near touchline/camera, 100=far touchline)
              //
              // If backend hasn't computed them, fall back to simple linear mapping
              const hasPitchCoords = det.pitch_x !== undefined && det.pitch_y !== undefined;
              const pitchX = hasPitchCoords ? det.pitch_x : ((det.bbox[0] + det.bbox[2]) / 2 / 1920 * 100);
              const pitchY = hasPitchCoords ? det.pitch_y : ((1 - det.bbox[3] / 1080) * 100);

              // Transform to aerial view display:
              // Aerial display has:
              // - Home goal at BOTTOM (aerialY = 100%)
              // - Away goal at TOP (aerialY = 0%)
              // - Camera/near touchline at one side
              //
              // pitch_x: 0 = left goal in video (home defends), 100 = right goal (home attacks)
              // pitch_y: 0 = near touchline (camera side), 100 = far touchline
              //
              // So for aerial view where home is at bottom:
              // aerialY = 100 - pitch_x (invert so home goal is at bottom)
              // aerialX = pitch_y (far touchline = top of aerial = left side... or we keep it natural)
              const aerialX = pitchY;  // Width across pitch
              const aerialY = 100 - pitchX;  // Depth: 0=away goal (top), 100=home goal (bottom)

              return (
                <div
                  key={idx}
                  className={`absolute w-5 h-5 rounded-full transform -translate-x-1/2 -translate-y-1/2 border-2 border-white shadow-lg ${
                    det.team === 'home' ? 'bg-red-500' : det.team === 'away' ? 'bg-blue-500' : 'bg-slate-500'
                  }`}
                  style={{ left: `${aerialX}%`, top: `${aerialY}%` }}
                  title={`${det.team} #${idx + 1} | Pitch: (${pitchX.toFixed(0)}m, ${pitchY.toFixed(0)}m)`}
                >
                  <span className="absolute inset-0 flex items-center justify-center text-white text-[8px] font-bold">
                    {idx + 1}
                  </span>
                </div>
              );
            })}

            {/* Ball position - using homography-transformed coordinates */}
            {currentFrame?.ball_position && (() => {
              const hasBallPitchCoords = currentFrame.ball_pitch_x !== undefined;
              const ballPitchX = hasBallPitchCoords ? currentFrame.ball_pitch_x : (currentFrame.ball_position[0] / 1920 * 100);
              const ballPitchY = hasBallPitchCoords ? currentFrame.ball_pitch_y : ((1 - currentFrame.ball_position[1] / 1080) * 100);
              return (
                <div
                  className="absolute w-4 h-4 rounded-full bg-yellow-400 border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2 z-10"
                  style={{
                    left: `${ballPitchY}%`,
                    top: `${100 - ballPitchX}%`
                  }}
                />
              );
            })()}

            {/* Goal indicators */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 bg-blue-500/80 text-white text-[10px] px-2 py-0.5 rounded-b font-medium">
              AWAY GOAL
            </div>
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 bg-red-500/80 text-white text-[10px] px-2 py-0.5 rounded-t font-medium">
              HOME GOAL
            </div>
          </div>
        </div>

        {/* Events Panel */}
        <div className="bg-[#111827] rounded-2xl p-4 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
            <span className="text-yellow-400">⚡</span> Nearby Events
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {nearbyEvents.length > 0 ? nearbyEvents.map((event, idx) => (
              <button
                key={idx}
                onClick={() => handleSeek(event.timestamp)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  Math.abs(event.timestamp - currentTime) < 3
                    ? 'bg-cyan-500/20 border border-cyan-500/50'
                    : 'bg-slate-800/50 hover:bg-slate-700/50'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-2 h-2 rounded-full ${eventColors[event.type] || 'bg-slate-500'}`}></div>
                  <span className="text-cyan-400 text-xs font-mono">{formatTime(event.timestamp)}</span>
                </div>
                <p className="text-white text-sm">{event.description}</p>
              </button>
            )) : (
              <p className="text-slate-400 text-sm text-center py-4">No events near current time</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ==================== COMBINED ANALYSIS VIEW ====================
function CombinedAnalysisView({ analysis, stats }: { analysis: MatchAnalysis; stats: MatchStats }) {
  const [activeSection, setActiveSection] = useState<'overview' | 'heatmap' | 'possession' | 'shots' | 'formations'>('overview');

  return (
    <div className="space-y-6">
      {/* Section Tabs */}
      <div className="flex gap-2 bg-[#111827] p-2 rounded-xl border border-slate-700/30">
        {[
          { id: 'overview', label: 'Overview', icon: '📊' },
          { id: 'heatmap', label: 'Heatmaps', icon: '🗺️' },
          { id: 'possession', label: 'Possession', icon: '🎯' },
          { id: 'shots', label: 'Shots', icon: '🥅' },
          { id: 'formations', label: 'Formations', icon: '📋' },
        ].map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id as any)}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeSection === section.id
                ? 'bg-cyan-500 text-white'
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            <span className="mr-2">{section.icon}</span>
            {section.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeSection === 'overview' && <MatchOverview analysis={analysis} stats={stats} />}
      {activeSection === 'heatmap' && <HeatmapView analysis={analysis} />}
      {activeSection === 'possession' && <PossessionView analysis={analysis} stats={stats} />}
      {activeSection === 'shots' && <ShotsView analysis={analysis} />}
      {activeSection === 'formations' && <FormationsView />}
    </div>
  );
}

// ==================== COMBINED TACTICAL VIEW ====================
function CombinedTacticalView({ analysis }: { analysis: MatchAnalysis }) {
  const [activeSection, setActiveSection] = useState<'events' | 'momentum' | 'tracking' | 'radar'>('events');

  return (
    <div className="space-y-6">
      {/* Section Tabs */}
      <div className="flex gap-2 bg-[#111827] p-2 rounded-xl border border-slate-700/30">
        {[
          { id: 'events', label: 'Events Timeline', icon: '⚡' },
          { id: 'momentum', label: 'Momentum', icon: '📈' },
          { id: 'tracking', label: 'Player Tracking', icon: '🎯' },
          { id: 'radar', label: '2D Radar', icon: '📍' },
        ].map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id as any)}
            className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeSection === section.id
                ? 'bg-cyan-500 text-white'
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            <span className="mr-2">{section.icon}</span>
            {section.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeSection === 'events' && <TacticalEventsView />}
      {activeSection === 'momentum' && <MomentumView analysis={analysis} stats={calculateMatchStats(analysis, 'full')} />}
      {activeSection === 'tracking' && <PredictiveTrackingView />}
      {activeSection === 'radar' && <TimelineView analysis={analysis} />}
    </div>
  );
}

// ==================== MATCH OVERVIEW ====================
function MatchOverview({ analysis, stats }: { analysis: MatchAnalysis; stats: MatchStats }) {
  return (
    <div className="space-y-6">
      {/* Scoreboard Style Header */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center justify-between">
          <TeamDisplay name="Your Team" color="red" players={stats.avgHomePlayers} isHome />
          <div className="text-center">
            <div className="text-4xl font-bold text-white mb-1">
              {stats.homeScore} - {stats.awayScore}
            </div>
            <div className="text-slate-400 text-sm">{formatDuration(analysis.duration_seconds)}</div>
          </div>
          <TeamDisplay name="Opposition" color="slate" players={stats.avgAwayPlayers} isHome={false} />
        </div>
      </div>

      {/* Key Stats Grid */}
      <div className="grid grid-cols-4 gap-4">
        <StatBox label="Possession" homeValue={`${stats.possession.home}%`} awayValue={`${stats.possession.away}%`} />
        <StatBox label="Territorial Presence" homeValue={`${stats.territorialAdvantage.home}%`} awayValue={`${stats.territorialAdvantage.away}%`} />
        <StatBox label="Pressing Actions" homeValue={stats.pressingActions.home.toString()} awayValue={stats.pressingActions.away.toString()} />
        <StatBox label="High Intensity Moments" homeValue={stats.highIntensityMoments.home.toString()} awayValue={stats.highIntensityMoments.away.toString()} />
      </div>

      {/* Momentum Preview */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <span className="text-cyan-400">📈</span> Match Momentum
        </h3>
        <MomentumGraph analysis={analysis} height={120} />
        <div className="flex justify-between mt-3 text-xs text-slate-500">
          <span>0'</span>
          <span>{Math.floor(analysis.duration_seconds / 60)}'</span>
        </div>
      </div>

      {/* Mini Heatmaps */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[#111827] rounded-2xl p-5 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-4 text-sm">Your Team Positioning</h3>
          <MiniPitchHeatmap analysis={analysis} team="home" />
        </div>
        <div className="bg-[#111827] rounded-2xl p-5 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-4 text-sm">Opposition Positioning</h3>
          <MiniPitchHeatmap analysis={analysis} team="away" />
        </div>
      </div>

      {/* AI Insights */}
      <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-2xl p-6 border border-cyan-500/20">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <span className="text-cyan-400">🤖</span> AI Coach Insights
        </h3>
        <div className="space-y-3">
          {stats.insights.map((insight, i) => (
            <div key={i} className="flex items-start gap-3">
              <span className={`text-lg ${insight.type === 'positive' ? 'text-emerald-400' : insight.type === 'warning' ? 'text-amber-400' : 'text-cyan-400'}`}>
                {insight.type === 'positive' ? '✓' : insight.type === 'warning' ? '⚠' : '💡'}
              </span>
              <div>
                <p className="text-white text-sm font-medium">{insight.title}</p>
                <p className="text-slate-400 text-xs mt-0.5">{insight.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ==================== MOMENTUM VIEW ====================
function MomentumView({ analysis, stats }: { analysis: MatchAnalysis; stats: MatchStats }) {
  return (
    <div className="space-y-6">
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white font-semibold">Match Momentum</h3>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-2"><span className="w-3 h-3 bg-red-500 rounded"></span> Your Team</span>
            <span className="flex items-center gap-2"><span className="w-3 h-3 bg-slate-400 rounded"></span> Opposition</span>
          </div>
        </div>
        <MomentumGraph analysis={analysis} height={200} detailed />
        <div className="flex justify-between mt-4 text-xs text-slate-500">
          <span>0'</span>
          <span>{Math.floor(analysis.duration_seconds / 4 / 60)}'</span>
          <span>{Math.floor(analysis.duration_seconds / 2 / 60)}'</span>
          <span>{Math.floor(analysis.duration_seconds * 3 / 4 / 60)}'</span>
          <span>{Math.floor(analysis.duration_seconds / 60)}'</span>
        </div>
      </div>

      {/* Momentum Periods Breakdown */}
      <div className="grid grid-cols-3 gap-4">
        {stats.momentumPeriods.map((period, i) => (
          <div key={i} className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
            <div className="text-slate-400 text-xs mb-2">{period.label}</div>
            <div className="flex items-center gap-3">
              <div className={`text-2xl font-bold ${period.dominant === 'home' ? 'text-red-400' : 'text-slate-400'}`}>
                {period.homeControl}%
              </div>
              <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-full"
                  style={{ width: `${period.homeControl}%` }}
                />
              </div>
              <div className={`text-2xl font-bold ${period.dominant === 'away' ? 'text-slate-300' : 'text-slate-500'}`}>
                {period.awayControl}%
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Key Momentum Shifts */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Key Momentum Shifts</h3>
        <div className="space-y-3">
          {stats.momentumShifts.map((shift, i) => (
            <div key={i} className="flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg">
              <div className="text-cyan-400 font-mono text-sm">{shift.time}</div>
              <div className="flex-1">
                <p className="text-white text-sm">{shift.description}</p>
              </div>
              <div className={`px-2 py-1 rounded text-xs ${shift.favor === 'home' ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'}`}>
                {shift.favor === 'home' ? 'Your Team' : 'Opposition'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ==================== HEATMAP VIEW ====================
function HeatmapView({ analysis }: { analysis: MatchAnalysis }) {
  const [viewMode, setViewMode] = useState<'both' | 'home' | 'away'>('both');

  return (
    <div className="space-y-6">
      {/* View Mode Toggle */}
      <div className="flex items-center gap-2">
        {(['both', 'home', 'away'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`px-4 py-2 text-sm rounded-lg transition-all ${
              viewMode === mode ? 'bg-cyan-500 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            {mode === 'both' ? 'Both Teams' : mode === 'home' ? 'Your Team' : 'Opposition'}
          </button>
        ))}
      </div>

      {/* Heatmaps */}
      {viewMode === 'both' ? (
        <div className="grid grid-cols-2 gap-6">
          <FullPitchHeatmap analysis={analysis} team="home" title="Your Team (Red)" />
          <FullPitchHeatmap analysis={analysis} team="away" title="Opposition" />
        </div>
      ) : (
        <FullPitchHeatmap analysis={analysis} team={viewMode} title={viewMode === 'home' ? 'Your Team (Red)' : 'Opposition'} large />
      )}

      {/* Zone Analysis */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Pitch Zone Analysis</h3>
        <ZoneBreakdown analysis={analysis} />
      </div>
    </div>
  );
}

// ==================== POSSESSION VIEW ====================
function PossessionView({ analysis, stats }: { analysis: MatchAnalysis; stats: MatchStats }) {
  return (
    <div className="space-y-6">
      {/* Possession Donut */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-6">Possession</h3>
        <div className="flex items-center justify-around">
          <div className="text-center">
            <div className="text-5xl font-bold text-red-400">{stats.possession.home}%</div>
            <div className="text-slate-400 text-sm mt-1">Your Team</div>
          </div>
          <PossessionDonut home={stats.possession.home} away={stats.possession.away} />
          <div className="text-center">
            <div className="text-5xl font-bold text-slate-400">{stats.possession.away}%</div>
            <div className="text-slate-400 text-sm mt-1">Opposition</div>
          </div>
        </div>
      </div>

      {/* Possession Location Map */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Possession Location Map</h3>
        <p className="text-slate-400 text-sm mb-4">Where each team controlled the ball</p>
        <PossessionLocationMap analysis={analysis} />
      </div>

      {/* Possession by Zone */}
      <div className="grid grid-cols-3 gap-4">
        {['Defensive Third', 'Middle Third', 'Attacking Third'].map((zone, i) => (
          <div key={zone} className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
            <div className="text-slate-400 text-xs mb-3">{zone}</div>
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <div className="h-20 flex items-end">
                  <div
                    className="w-full bg-gradient-to-t from-red-600 to-red-400 rounded-t"
                    style={{ height: `${stats.possessionByZone[i].home}%` }}
                  />
                </div>
                <div className="text-center mt-2">
                  <div className="text-red-400 font-bold">{stats.possessionByZone[i].home}%</div>
                  <div className="text-slate-500 text-xs">You</div>
                </div>
              </div>
              <div className="flex-1">
                <div className="h-20 flex items-end">
                  <div
                    className="w-full bg-gradient-to-t from-slate-600 to-slate-400 rounded-t"
                    style={{ height: `${stats.possessionByZone[i].away}%` }}
                  />
                </div>
                <div className="text-center mt-2">
                  <div className="text-slate-400 font-bold">{stats.possessionByZone[i].away}%</div>
                  <div className="text-slate-500 text-xs">Opp</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ==================== TIMELINE VIEW (2D RADAR) ====================
function TimelineView({ analysis }: { analysis: MatchAnalysis }) {
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const currentFrame = analysis.frames[currentFrameIndex];

  // Auto-advance when playing
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentFrameIndex(prev => {
        if (prev >= analysis.frames.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000 / playbackSpeed);
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, analysis.frames.length]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentFrameIndex(Number(e.target.value));
    setIsPlaying(false);
  };

  return (
    <div className="space-y-6">
      {/* 2D Pitch with Player Positions */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">2D Radar View</h3>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-2"><span className="w-3 h-3 bg-red-500 rounded-full"></span> Your Team ({currentFrame.home_players})</span>
            <span className="flex items-center gap-2"><span className="w-3 h-3 bg-slate-400 rounded-full"></span> Opposition ({currentFrame.away_players})</span>
            {currentFrame.ball_position && (
              <span className="flex items-center gap-2"><span className="w-3 h-3 bg-yellow-400 rounded-full"></span> Ball</span>
            )}
          </div>
        </div>

        {/* Pitch with players */}
        <div className="relative aspect-[3/2] bg-emerald-800/40 rounded-xl overflow-hidden">
          <PitchSVGDetailed />

          {/* Player dots */}
          {currentFrame.detections.map((det, i) => {
            const x = ((det.bbox[0] + det.bbox[2]) / 2 / VIDEO_WIDTH) * 100;
            const y = ((det.bbox[1] + det.bbox[3]) / 2 / VIDEO_HEIGHT) * 100;
            const color = det.team === 'home' ? 'bg-red-500' :
                          det.team === 'away' ? 'bg-slate-300' :
                          det.team === 'referee' ? 'bg-yellow-600' : 'bg-slate-600';
            return (
              <div
                key={i}
                className={`absolute w-4 h-4 ${color} rounded-full transform -translate-x-1/2 -translate-y-1/2 border-2 border-white shadow-lg transition-all duration-300`}
                style={{ left: `${x}%`, top: `${y}%` }}
                title={`${det.team} player (${Math.round(det.confidence * 100)}% conf)`}
              />
            );
          })}

          {/* Ball position */}
          {currentFrame.ball_position && (
            <div
              className="absolute w-3 h-3 bg-yellow-400 rounded-full transform -translate-x-1/2 -translate-y-1/2 border-2 border-white shadow-lg animate-pulse"
              style={{
                left: `${(currentFrame.ball_position[0] / VIDEO_WIDTH) * 100}%`,
                top: `${(currentFrame.ball_position[1] / VIDEO_HEIGHT) * 100}%`
              }}
            />
          )}

          {/* Timestamp overlay */}
          <div className="absolute bottom-3 left-3 bg-black/50 rounded-lg px-3 py-1.5 backdrop-blur-sm">
            <span className="text-white font-mono text-lg">{formatTime(currentFrame.timestamp)}</span>
          </div>

          {/* Player count overlay */}
          <div className="absolute top-3 left-3 bg-black/50 rounded-lg px-3 py-1.5 backdrop-blur-sm">
            <span className="text-white text-sm">{currentFrame.player_count} players detected</span>
          </div>
        </div>

        {/* Timeline Controls */}
        <div className="mt-6 space-y-4">
          {/* Progress bar */}
          <div className="relative">
            <input
              type="range"
              min={0}
              max={analysis.frames.length - 1}
              value={currentFrameIndex}
              onChange={handleSliderChange}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <div className="flex justify-between mt-1 text-xs text-slate-500">
              <span>0:00</span>
              <span>{formatTime(analysis.duration_seconds)}</span>
            </div>
          </div>

          {/* Playback controls */}
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => setCurrentFrameIndex(Math.max(0, currentFrameIndex - 10))}
              className="p-2 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors text-white"
              title="Back 10 frames"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </button>

            <button
              onClick={() => setCurrentFrameIndex(Math.max(0, currentFrameIndex - 1))}
              className="p-2 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors text-white"
              title="Previous frame"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-3 bg-cyan-500 rounded-full hover:bg-cyan-600 transition-colors text-white"
            >
              {isPlaying ? (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
              )}
            </button>

            <button
              onClick={() => setCurrentFrameIndex(Math.min(analysis.frames.length - 1, currentFrameIndex + 1))}
              className="p-2 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors text-white"
              title="Next frame"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>

            <button
              onClick={() => setCurrentFrameIndex(Math.min(analysis.frames.length - 1, currentFrameIndex + 10))}
              className="p-2 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors text-white"
              title="Forward 10 frames"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </button>

            {/* Speed control */}
            <div className="ml-4 flex items-center gap-2">
              <span className="text-slate-400 text-sm">Speed:</span>
              {[0.5, 1, 2, 4].map(speed => (
                <button
                  key={speed}
                  onClick={() => setPlaybackSpeed(speed)}
                  className={`px-2 py-1 text-xs rounded ${
                    playbackSpeed === speed ? 'bg-cyan-500 text-white' : 'bg-slate-700/50 text-slate-400 hover:text-white'
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Frame Info Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Frame</div>
          <div className="text-white text-2xl font-bold">{currentFrameIndex + 1}</div>
          <div className="text-slate-500 text-xs">of {analysis.frames.length}</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Your Team</div>
          <div className="text-red-400 text-2xl font-bold">{currentFrame.home_players}</div>
          <div className="text-slate-500 text-xs">players visible</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Opposition</div>
          <div className="text-slate-400 text-2xl font-bold">{currentFrame.away_players}</div>
          <div className="text-slate-500 text-xs">players visible</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Ball Detected</div>
          <div className={`text-2xl font-bold ${currentFrame.ball_position ? 'text-yellow-400' : 'text-slate-600'}`}>
            {currentFrame.ball_position ? 'Yes' : 'No'}
          </div>
          <div className="text-slate-500 text-xs">in this frame</div>
        </div>
      </div>

      {/* Quick Jump Moments */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Quick Jump</h3>
        <div className="grid grid-cols-6 gap-2">
          {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1].map((pct, i) => {
            const frameIndex = Math.min(analysis.frames.length - 1, Math.floor(pct * (analysis.frames.length - 1)));
            const frame = analysis.frames[frameIndex];
            return (
              <button
                key={i}
                onClick={() => setCurrentFrameIndex(frameIndex)}
                className={`p-3 rounded-lg text-center transition-all ${
                  currentFrameIndex === frameIndex ? 'bg-cyan-500 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
                }`}
              >
                <div className="text-sm font-mono">{formatTime(frame.timestamp)}</div>
                <div className="text-xs mt-1 opacity-70">{frame.home_players}v{frame.away_players}</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ==================== SHOTS VIEW ====================
function ShotsView({ analysis }: { analysis: MatchAnalysis }) {
  // Detect shots from ball movement and player positions
  const shots = useMemo(() => {
    return detectShots(analysis);
  }, [analysis]);

  const homeShots = shots.filter(s => s.team === 'home');
  const awayShots = shots.filter(s => s.team === 'away');
  const homeShotsOnTarget = homeShots.filter(s => s.on_target);
  const awayShotsOnTarget = awayShots.filter(s => s.on_target);

  return (
    <div className="space-y-6">
      {/* Shot Stats Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Your Shots</div>
          <div className="text-red-400 text-3xl font-bold">{homeShots.length}</div>
          <div className="text-slate-500 text-xs">{homeShotsOnTarget.length} on target</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Your Accuracy</div>
          <div className="text-red-400 text-3xl font-bold">
            {homeShots.length > 0 ? Math.round((homeShotsOnTarget.length / homeShots.length) * 100) : 0}%
          </div>
          <div className="text-slate-500 text-xs">shots on target</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Opp Shots</div>
          <div className="text-slate-400 text-3xl font-bold">{awayShots.length}</div>
          <div className="text-slate-500 text-xs">{awayShotsOnTarget.length} on target</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Opp Accuracy</div>
          <div className="text-slate-400 text-3xl font-bold">
            {awayShots.length > 0 ? Math.round((awayShotsOnTarget.length / awayShots.length) * 100) : 0}%
          </div>
          <div className="text-slate-500 text-xs">shots on target</div>
        </div>
      </div>

      {/* Shot Map */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Shot Map</h3>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 bg-red-500 rounded-full"></span> Your Team ({homeShots.length})
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 bg-slate-400 rounded-full"></span> Opposition ({awayShots.length})
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 border-2 border-white rounded-full"></span> On Target
            </span>
          </div>
        </div>

        {/* Pitch with shot positions */}
        <div className="relative aspect-[3/2] bg-emerald-800/40 rounded-xl overflow-hidden">
          <PitchSVGDetailed />

          {/* Shot markers */}
          {shots.map((shot, i) => {
            const x = (shot.position[0] / VIDEO_WIDTH) * 100;
            const y = (shot.position[1] / VIDEO_HEIGHT) * 100;
            const isHome = shot.team === 'home';
            const onTarget = shot.on_target;

            return (
              <div
                key={i}
                className={`absolute w-5 h-5 rounded-full transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center transition-all hover:scale-125 cursor-pointer ${
                  isHome ? 'bg-red-500' : 'bg-slate-400'
                } ${onTarget ? 'ring-2 ring-white ring-offset-2 ring-offset-emerald-800' : ''}`}
                style={{ left: `${x}%`, top: `${y}%` }}
                title={`${formatTime(shot.timestamp)} - ${isHome ? 'Your Team' : 'Opposition'} - ${onTarget ? 'On Target' : 'Off Target'}`}
              >
                <span className="text-white text-xs font-bold">{i + 1}</span>
              </div>
            );
          })}

          {/* Goal areas highlight */}
          <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[15%] h-[40%] border-r-2 border-yellow-400/30 bg-yellow-400/5"></div>
          <div className="absolute right-0 top-1/2 -translate-y-1/2 w-[15%] h-[40%] border-l-2 border-yellow-400/30 bg-yellow-400/5"></div>
        </div>

        <div className="flex items-center justify-center gap-6 mt-4 text-xs text-slate-400">
          <span>Attacking left goal</span>
          <span className="text-slate-600">|</span>
          <span>Attacking right goal</span>
        </div>
      </div>

      {/* Shot List */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Shot Timeline</h3>
        {shots.length === 0 ? (
          <p className="text-slate-400 text-sm text-center py-8">
            No shots detected in this match. Shot detection requires ball tracking data.
          </p>
        ) : (
          <div className="space-y-2">
            {shots.map((shot, i) => (
              <div
                key={i}
                className={`flex items-center gap-4 p-3 rounded-lg ${
                  shot.team === 'home' ? 'bg-red-500/10' : 'bg-slate-500/10'
                }`}
              >
                <div className="text-cyan-400 font-mono text-sm w-16">{formatTime(shot.timestamp)}</div>
                <div className={`w-3 h-3 rounded-full ${shot.team === 'home' ? 'bg-red-500' : 'bg-slate-400'}`}></div>
                <div className="flex-1">
                  <span className="text-white text-sm">
                    {shot.team === 'home' ? 'Your Team' : 'Opposition'} shot
                  </span>
                  <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                    shot.on_target ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-500/20 text-slate-400'
                  }`}>
                    {shot.on_target ? 'On Target' : 'Off Target'}
                  </span>
                </div>
                <div className="text-slate-500 text-xs">
                  ~{Math.round(shot.distance_estimate / 10)}m from goal
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Shot Detection Info */}
      <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-2xl p-6 border border-cyan-500/20">
        <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
          <span className="text-cyan-400">ℹ️</span> How Shot Detection Works
        </h3>
        <div className="text-slate-400 text-sm space-y-2">
          <p>Shots are detected by analyzing:</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li>Ball position moving rapidly toward goal areas</li>
            <li>Player clusters in the attacking third</li>
            <li>Sudden changes in ball trajectory</li>
          </ul>
          <p className="text-slate-500 mt-3">
            Note: Shot detection accuracy depends on ball tracking quality. For best results,
            ensure good camera coverage of both goal areas.
          </p>
        </div>
      </div>
    </div>
  );
}

// Shot detection algorithm
function detectShots(analysis: MatchAnalysis): DetectedShot[] {
  const shots: DetectedShot[] = [];
  const GOAL_ZONE_LEFT = VIDEO_WIDTH * 0.15;  // Left 15% = goal zone
  const GOAL_ZONE_RIGHT = VIDEO_WIDTH * 0.85; // Right 15% = goal zone
  const GOAL_Y_TOP = VIDEO_HEIGHT * 0.3;
  const GOAL_Y_BOTTOM = VIDEO_HEIGHT * 0.7;

  // Track ball movement between frames
  let prevBallPos: [number, number] | null = null;
  let prevPrevBallPos: [number, number] | null = null;
  let lastShotFrame = -100; // Prevent duplicate shots

  for (let i = 0; i < analysis.frames.length; i++) {
    const frame = analysis.frames[i];
    const ballPos = frame.ball_position;

    if (ballPos && prevBallPos) {
      // Calculate ball movement
      const dx = ballPos[0] - prevBallPos[0];
      const dy = ballPos[1] - prevBallPos[1];
      const speed = Math.sqrt(dx * dx + dy * dy);

      // Check if ball is moving fast toward a goal
      const movingTowardLeftGoal = dx < -50 && ballPos[0] < VIDEO_WIDTH * 0.4;
      const movingTowardRightGoal = dx > 50 && ballPos[0] > VIDEO_WIDTH * 0.6;

      // Check if in shooting position (attacking third)
      const inLeftAttackingThird = ballPos[0] < VIDEO_WIDTH * 0.35;
      const inRightAttackingThird = ballPos[0] > VIDEO_WIDTH * 0.65;

      // Minimum frames between shots to avoid duplicates
      const framesSinceLastShot = frame.frame_number - lastShotFrame;

      if (speed > 80 && framesSinceLastShot > 30) { // Fast movement, not too close to previous shot
        let shotDetected = false;
        let team: 'home' | 'away' = 'home';
        let onTarget = false;

        if (movingTowardLeftGoal && inLeftAttackingThird) {
          // Shot toward left goal - likely away team attacking
          team = 'away';
          shotDetected = true;
          // Check if ball trajectory would hit goal area
          onTarget = ballPos[1] > GOAL_Y_TOP && ballPos[1] < GOAL_Y_BOTTOM;
        } else if (movingTowardRightGoal && inRightAttackingThird) {
          // Shot toward right goal - likely home team attacking
          team = 'home';
          shotDetected = true;
          onTarget = ballPos[1] > GOAL_Y_TOP && ballPos[1] < GOAL_Y_BOTTOM;
        }

        if (shotDetected) {
          // Calculate distance from goal
          const distanceFromGoal = team === 'home'
            ? VIDEO_WIDTH - ballPos[0]
            : ballPos[0];

          shots.push({
            timestamp: frame.timestamp,
            frame_number: frame.frame_number,
            position: ballPos as [number, number],
            team,
            on_target: onTarget,
            distance_estimate: distanceFromGoal
          });
          lastShotFrame = frame.frame_number;
        }
      }
    }

    prevPrevBallPos = prevBallPos;
    prevBallPos = ballPos;
  }

  return shots;
}

// ==================== FORMATIONS VIEW ====================
function FormationsView() {
  const [formations, setFormations] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadFormations();
  }, []);

  const loadFormations = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/analytics/formations');
      if (response.ok) {
        const data = await response.json();
        setFormations(data.formations);
      } else {
        setError('Failed to load formation data');
      }
    } catch (err) {
      setError('Could not connect to server');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error || !formations) {
    return (
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30 text-center">
        <p className="text-slate-400">{error || 'No formation data available'}</p>
        <button onClick={loadFormations} className="mt-4 px-4 py-2 bg-cyan-500 text-white rounded-lg">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-white">Formation Analysis</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Home Team Formation */}
        <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-4 h-4 bg-red-500 rounded-full"></div>
            <h3 className="text-lg font-medium text-white">Your Team</h3>
          </div>

          <div className="text-center mb-6">
            <div className="text-4xl font-bold text-cyan-400 mb-2">
              {formations.home?.primary_formation || 'N/A'}
            </div>
            <p className="text-slate-400 text-sm">Primary Formation</p>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-400">Defensive Line Height</span>
              <span className="text-white font-medium">{formations.home?.avg_defensive_line || 0}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Team Compactness</span>
              <span className="text-white font-medium">{formations.home?.avg_compactness || 0}px</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Formation Changes</span>
              <span className="text-white font-medium">{formations.home?.formation_changes || 0}</span>
            </div>
          </div>

          {formations.home?.formation_counts && Object.keys(formations.home.formation_counts).length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-700/50">
              <p className="text-slate-400 text-sm mb-2">Formation Distribution</p>
              <div className="space-y-2">
                {Object.entries(formations.home.formation_counts).slice(0, 5).map(([formation, count]: [string, any]) => (
                  <div key={formation} className="flex items-center gap-2">
                    <span className="text-white text-sm w-16">{formation}</span>
                    <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                      <div
                        className="bg-red-500 h-2 rounded-full"
                        style={{
                          width: `${(count / Object.values(formations.home.formation_counts).reduce((a: number, b: any) => a + b, 0)) * 100}%`
                        }}
                      ></div>
                    </div>
                    <span className="text-slate-400 text-xs w-8 text-right">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Away Team Formation */}
        <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-4 h-4 bg-slate-400 rounded-full"></div>
            <h3 className="text-lg font-medium text-white">Opposition</h3>
          </div>

          <div className="text-center mb-6">
            <div className="text-4xl font-bold text-cyan-400 mb-2">
              {formations.away?.primary_formation || 'N/A'}
            </div>
            <p className="text-slate-400 text-sm">Primary Formation</p>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-400">Defensive Line Height</span>
              <span className="text-white font-medium">{formations.away?.avg_defensive_line || 0}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Team Compactness</span>
              <span className="text-white font-medium">{formations.away?.avg_compactness || 0}px</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Formation Changes</span>
              <span className="text-white font-medium">{formations.away?.formation_changes || 0}</span>
            </div>
          </div>

          {formations.away?.formation_counts && Object.keys(formations.away.formation_counts).length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-700/50">
              <p className="text-slate-400 text-sm mb-2">Formation Distribution</p>
              <div className="space-y-2">
                {Object.entries(formations.away.formation_counts).slice(0, 5).map(([formation, count]: [string, any]) => (
                  <div key={formation} className="flex items-center gap-2">
                    <span className="text-white text-sm w-16">{formation}</span>
                    <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                      <div
                        className="bg-slate-400 h-2 rounded-full"
                        style={{
                          width: `${(count / Object.values(formations.away.formation_counts).reduce((a: number, b: any) => a + b, 0)) * 100}%`
                        }}
                      ></div>
                    </div>
                    <span className="text-slate-400 text-xs w-8 text-right">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Formation Timeline */}
      {(formations.home?.timeline?.length > 0 || formations.away?.timeline?.length > 0) && (
        <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <h3 className="text-lg font-medium text-white mb-4">Formation Changes Timeline</h3>
          <div className="space-y-2">
            {[...(formations.home?.timeline || []).map((t: any) => ({ ...t, team: 'home' })),
              ...(formations.away?.timeline || []).map((t: any) => ({ ...t, team: 'away' }))]
              .sort((a, b) => a.timestamp - b.timestamp)
              .slice(0, 20)
              .map((change: any, idx: number) => (
                <div key={idx} className="flex items-center gap-4 py-2 border-b border-slate-700/30">
                  <span className="text-slate-400 text-sm w-16">{formatTime(change.timestamp)}</span>
                  <div className={`w-3 h-3 rounded-full ${change.team === 'home' ? 'bg-red-500' : 'bg-slate-400'}`}></div>
                  <span className="text-white font-medium">{change.formation}</span>
                  <span className="text-slate-400 text-sm">{change.team === 'home' ? 'Your Team' : 'Opposition'}</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ==================== TACTICAL EVENTS VIEW ====================
function TacticalEventsView() {
  const [events, setEvents] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/analytics/tactical-events');
      if (response.ok) {
        const data = await response.json();
        setEvents(data.tactical_events);
      } else {
        setError('Failed to load tactical events');
      }
    } catch (err) {
      setError('Could not connect to server');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error || !events) {
    return (
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30 text-center">
        <p className="text-slate-400">{error || 'No tactical events available'}</p>
        <button onClick={loadEvents} className="mt-4 px-4 py-2 bg-cyan-500 text-white rounded-lg">
          Retry
        </button>
      </div>
    );
  }

  const eventTypeColors: Record<string, string> = {
    pressing_trigger: 'bg-yellow-500',
    pressing_opportunity: 'bg-yellow-400',
    dangerous_attack: 'bg-red-500',
    counter_attack: 'bg-orange-500',
    shape_warning: 'bg-purple-500',
    high_line_opportunity: 'bg-blue-500',
    transition_moment: 'bg-green-500',
    overload: 'bg-cyan-500',
  };

  const priorityColors: Record<number, string> = {
    1: 'text-slate-400',
    2: 'text-yellow-400',
    3: 'text-orange-400',
    4: 'text-red-400',
  };

  const filteredEvents = filter === 'all'
    ? events.events
    : events.events?.filter((e: any) => e.type === filter);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white">Tactical Events</h2>
        <span className="text-slate-400">{events.total_events || 0} events detected</span>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-2xl font-bold text-red-400">{events.priority_counts?.critical || 0}</div>
          <div className="text-slate-400 text-sm">Critical</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-2xl font-bold text-orange-400">{events.priority_counts?.high || 0}</div>
          <div className="text-slate-400 text-sm">High Priority</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-2xl font-bold text-yellow-400">{events.priority_counts?.medium || 0}</div>
          <div className="text-slate-400 text-sm">Medium</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
          <div className="text-2xl font-bold text-slate-400">{events.priority_counts?.low || 0}</div>
          <div className="text-slate-400 text-sm">Low</div>
        </div>
      </div>

      {/* Event Type Breakdown */}
      {events.event_counts && (
        <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <h3 className="text-lg font-medium text-white mb-4">Event Breakdown</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(events.event_counts).map(([type, count]: [string, any]) => (
              <button
                key={type}
                onClick={() => setFilter(filter === type ? 'all' : type)}
                className={`p-3 rounded-lg border transition-all ${
                  filter === type
                    ? 'border-cyan-500 bg-cyan-500/10'
                    : 'border-slate-700/50 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-2 h-2 rounded-full ${eventTypeColors[type] || 'bg-slate-500'}`}></div>
                  <span className="text-white text-sm font-medium">{count}</span>
                </div>
                <div className="text-slate-400 text-xs capitalize">{type.replace(/_/g, ' ')}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Events List */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">Event Timeline</h3>
          {filter !== 'all' && (
            <button
              onClick={() => setFilter('all')}
              className="text-cyan-400 text-sm hover:underline"
            >
              Clear filter
            </button>
          )}
        </div>

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredEvents?.slice(0, 50).map((event: any, idx: number) => (
            <div
              key={idx}
              className="flex items-start gap-4 p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
            >
              <span className="text-slate-400 text-sm w-16 flex-shrink-0">{formatTime(event.timestamp)}</span>
              <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${eventTypeColors[event.type] || 'bg-slate-500'}`}></div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${priorityColors[event.priority] || 'text-white'}`}>
                    {event.priority_name}
                  </span>
                  <span className="text-slate-500">•</span>
                  <span className="text-slate-400 text-sm capitalize">{event.type.replace(/_/g, ' ')}</span>
                </div>
                <p className="text-white text-sm mt-1">{event.description}</p>
                <span className="text-slate-500 text-xs">{event.team === 'home' ? 'Your Team' : 'Opposition'}</span>
              </div>
            </div>
          ))}

          {(!filteredEvents || filteredEvents.length === 0) && (
            <p className="text-slate-400 text-center py-4">No events match the filter</p>
          )}
        </div>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ==================== HUDL-STYLE PLAYERS VIEW ====================
// Match player type for clip generation
interface MatchPlayer {
  jersey_number: number;
  player_name: string;
  team: string;
  track_ids: number[];
  stats: {
    touches: number;
    passes_attempted: number;
    passes_completed: number;
    shots: number;
    goals: number;
    tackles: number;
    interceptions: number;
  };
  moments_count: number;
  clips_available: number;
  can_generate_clips: boolean;
}

interface PlayerClip {
  clip_id: string;
  jersey_number: number;
  player_name: string;
  team: string;
  event_type: string;
  timestamp_start_ms: number;
  duration_seconds: number;
  clip_path: string;
}

function PlayersView() {
  const [players, setPlayers] = useState<PlayerListItem[]>([]);
  const [matchPlayers, setMatchPlayers] = useState<MatchPlayer[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerStats | null>(null);
  const [selectedMatchPlayer, setSelectedMatchPlayer] = useState<MatchPlayer | null>(null);
  const [playerClips, setPlayerClips] = useState<PlayerClip[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analyzingClip, setAnalyzingClip] = useState(false);
  const [buildingHighlights, setBuildingHighlights] = useState(false);
  const [generatingClips, setGeneratingClips] = useState(false);
  const [clipPath, setClipPath] = useState('');
  const [playerName, setPlayerName] = useState('');
  const [activeTab, setActiveTab] = useState<'radar' | 'stats' | 'feedback' | 'clips'>('clips');
  const [viewMode, setViewMode] = useState<'match' | 'external'>('match');

  const loadPlayers = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/player/list');
      if (response.ok) {
        const data = await response.json();
        setPlayers(data.players || []);
      }
    } catch (err) {
      console.error('Could not load external player data');
    }
  };

  const loadMatchPlayers = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/highlights/match-players');
      if (response.ok) {
        const data = await response.json();
        setMatchPlayers(data.players || []);
      }
      setLoading(false);
    } catch (err) {
      setError('Could not load match player data');
      setLoading(false);
    }
  };

  const buildHighlightsFromDetections = async () => {
    setBuildingHighlights(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/highlights/build-from-detections', {
        method: 'POST'
      });
      if (response.ok) {
        await loadMatchPlayers();
      }
    } catch (err) {
      console.error('Failed to build highlights:', err);
    }
    setBuildingHighlights(false);
  };

  const loadPlayerClips = async (jerseyNumber: number) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/clips/player/${jerseyNumber}`);
      if (response.ok) {
        const data = await response.json();
        setPlayerClips(data.clips || []);
      }
    } catch (err) {
      console.error('Failed to load player clips:', err);
    }
  };

  const generatePlayerClips = async (jerseyNumber: number) => {
    setGeneratingClips(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/clips/player/${jerseyNumber}/extract?min_importance=0.5`, {
        method: 'POST'
      });
      if (response.ok) {
        await loadPlayerClips(jerseyNumber);
        await loadMatchPlayers();
      }
    } catch (err) {
      console.error('Failed to generate clips:', err);
    }
    setGeneratingClips(false);
  };

  useEffect(() => {
    loadPlayers();
    loadMatchPlayers();
  }, []);

  const loadPlayerStats = async (name: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/player/${encodeURIComponent(name)}/stats`);
      if (response.ok) {
        const data = await response.json();
        setSelectedPlayer(data);
      }
    } catch (err) {
      console.error('Failed to load player stats:', err);
    }
  };

  const analyzeClip = async () => {
    if (!clipPath || !playerName) return;
    setAnalyzingClip(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/player/analyze-clip?clip_path=${encodeURIComponent(clipPath)}&player_name=${encodeURIComponent(playerName)}`, {
        method: 'POST'
      });
      if (response.ok) {
        await loadPlayers();
        await loadPlayerStats(playerName);
        setClipPath('');
      }
    } catch (err) {
      console.error('Failed to analyze clip:', err);
    }
    setAnalyzingClip(false);
  };

  // Generate coaching feedback based on stats
  const generateFeedback = (player: PlayerStats) => {
    const feedback: { category: string; type: 'strength' | 'improvement' | 'focus'; message: string; detail: string }[] = [];

    // Passing feedback
    if (player.attacking.pass_accuracy >= 80) {
      feedback.push({ category: 'Passing', type: 'strength', message: 'Excellent distribution', detail: `${player.attacking.pass_accuracy.toFixed(0)}% accuracy shows reliable ball retention. Continue to take on more progressive passes.` });
    } else if (player.attacking.pass_accuracy < 65) {
      feedback.push({ category: 'Passing', type: 'improvement', message: 'Passing accuracy needs work', detail: `At ${player.attacking.pass_accuracy.toFixed(0)}%, focus on simpler passes and scanning before receiving.` });
    }

    // Shooting feedback
    if (player.attacking.shots > 0) {
      if (player.attacking.shot_accuracy >= 50) {
        feedback.push({ category: 'Shooting', type: 'strength', message: 'Clinical finisher', detail: `${player.attacking.shots_on_target}/${player.attacking.shots} shots on target. Good shot selection.` });
      } else {
        feedback.push({ category: 'Shooting', type: 'focus', message: 'Shot placement drill needed', detail: `Only ${player.attacking.shot_accuracy.toFixed(0)}% of shots hit the target. Practice finishing under pressure.` });
      }
    }

    // Defensive feedback
    if (player.defensive.tackle_success_rate >= 70) {
      feedback.push({ category: 'Defending', type: 'strength', message: 'Strong in the tackle', detail: `${player.defensive.tackle_success_rate.toFixed(0)}% tackle success rate demonstrates good timing and technique.` });
    } else if (player.defensive.tackles_attempted > 3 && player.defensive.tackle_success_rate < 50) {
      feedback.push({ category: 'Defending', type: 'improvement', message: 'Tackling technique', detail: `${player.defensive.tackle_success_rate.toFixed(0)}% success rate. Focus on body position and patience before committing.` });
    }

    // Activity feedback
    const touchesPerMin = player.attacking.ball_touches / Math.max(1, player.total_play_time_seconds / 60);
    if (touchesPerMin > 2) {
      feedback.push({ category: 'Involvement', type: 'strength', message: 'High work rate', detail: `${touchesPerMin.toFixed(1)} touches per minute shows excellent involvement and availability.` });
    } else if (touchesPerMin < 1) {
      feedback.push({ category: 'Involvement', type: 'focus', message: 'Get more involved', detail: `${touchesPerMin.toFixed(1)} touches per minute. Make yourself more available, check to the ball.` });
    }

    // Physical feedback
    if (player.physical.sprints > 5) {
      feedback.push({ category: 'Physical', type: 'strength', message: 'Good intensity', detail: `${player.physical.sprints} high-intensity runs shows commitment to pressing and attacking.` });
    }

    return feedback;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* HUDL-style Header */}
      <div className="bg-gradient-to-r from-[#0f172a] to-[#1e293b] rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">Player Performance & Clips</h2>
            <p className="text-slate-400 text-sm mt-1">View match players, generate individual clips, and analyze performance</p>
          </div>
          <div className="flex items-center gap-3">
            {/* View Mode Toggle */}
            <div className="flex bg-slate-800 rounded-lg p-1">
              <button
                type="button"
                onClick={() => setViewMode('match')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'match' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                Match Players
              </button>
              <button
                type="button"
                onClick={() => setViewMode('external')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'external' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                External Clips
              </button>
            </div>
            <div className="bg-slate-800 rounded-lg px-4 py-2 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span className="text-white text-sm font-medium">
                {viewMode === 'match' ? matchPlayers.length : players.length} Players
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Match Players View */}
      {viewMode === 'match' && (
        <>
          {/* Build Highlights Button */}
          {matchPlayers.length === 0 && (
            <div className="bg-[#111827] rounded-xl p-6 border border-slate-700/30 text-center">
              <div className="w-16 h-16 bg-cyan-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-white font-semibold mb-2">Build Player Highlights</h3>
              <p className="text-slate-400 text-sm mb-4">
                Extract player tracking data from your processed match to generate individual clips
              </p>
              <button
                type="button"
                onClick={buildHighlightsFromDetections}
                disabled={buildingHighlights}
                className="px-6 py-3 bg-cyan-500 text-white rounded-lg font-medium hover:bg-cyan-600 transition-colors disabled:opacity-50 flex items-center gap-2 mx-auto"
              >
                {buildingHighlights ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                    Building...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                    Build from Match Data
                  </>
                )}
              </button>
            </div>
          )}

          {/* Match Players Grid */}
          {matchPlayers.length > 0 && (
            <div className="grid grid-cols-4 gap-6">
              {/* Player List */}
              <div className="bg-[#111827] rounded-2xl border border-slate-700/30 overflow-hidden">
                <div className="p-4 border-b border-slate-700/30 flex items-center justify-between">
                  <h3 className="text-white font-semibold">Match Players</h3>
                  <button
                    type="button"
                    onClick={buildHighlightsFromDetections}
                    disabled={buildingHighlights}
                    className="text-xs px-2 py-1 bg-slate-700 text-slate-300 rounded hover:bg-slate-600"
                    title="Rebuild highlights from detection data"
                  >
                    {buildingHighlights ? '...' : '↻'}
                  </button>
                </div>
                <div className="p-2 max-h-[600px] overflow-y-auto">
                  {matchPlayers.map((player) => (
                    <button
                      type="button"
                      key={player.jersey_number}
                      onClick={() => {
                        setSelectedMatchPlayer(player);
                        loadPlayerClips(player.jersey_number);
                      }}
                      className={`w-full text-left p-3 rounded-xl transition-all ${
                        selectedMatchPlayer?.jersey_number === player.jersey_number
                          ? 'bg-cyan-500/20 border-l-4 border-cyan-500'
                          : 'hover:bg-slate-800/50'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                          player.team === 'home' ? 'bg-gradient-to-br from-blue-500 to-blue-600' : 'bg-gradient-to-br from-red-500 to-red-600'
                        }`}>
                          <span className="text-white font-bold text-sm">
                            {player.jersey_number > 0 ? `#${player.jersey_number}` : 'T'}
                          </span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-white font-medium truncate text-sm">{player.player_name}</div>
                          <div className="text-slate-500 text-xs">
                            {player.stats.touches} touches • {player.moments_count} moments
                          </div>
                        </div>
                        {player.clips_available > 0 && (
                          <span className="bg-green-500/20 text-green-400 text-xs px-2 py-0.5 rounded">
                            {player.clips_available}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Player Detail */}
              <div className="col-span-3 space-y-4">
                {selectedMatchPlayer ? (
                  <>
                    {/* Player Header */}
                    <div className="bg-gradient-to-r from-[#111827] to-[#1a2744] rounded-2xl p-6 border border-slate-700/30">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-5">
                          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center shadow-lg ${
                            selectedMatchPlayer.team === 'home'
                              ? 'bg-gradient-to-br from-blue-500 to-blue-600 shadow-blue-500/20'
                              : 'bg-gradient-to-br from-red-500 to-red-600 shadow-red-500/20'
                          }`}>
                            <span className="text-white text-2xl font-bold">
                              {selectedMatchPlayer.jersey_number > 0 ? `#${selectedMatchPlayer.jersey_number}` : 'T'}
                            </span>
                          </div>
                          <div>
                            <h2 className="text-2xl font-bold text-white">{selectedMatchPlayer.player_name}</h2>
                            <p className="text-slate-400 text-sm mt-1">
                              {selectedMatchPlayer.team === 'home' ? 'Home Team' : 'Away Team'} •
                              Track IDs: {selectedMatchPlayer.track_ids.slice(0, 3).join(', ')}{selectedMatchPlayer.track_ids.length > 3 ? '...' : ''}
                            </p>
                            <div className="flex items-center gap-4 mt-3">
                              <span className="text-slate-300 text-sm">{selectedMatchPlayer.moments_count} moments tracked</span>
                              <span className="text-slate-300 text-sm">{selectedMatchPlayer.clips_available} clips available</span>
                            </div>
                          </div>
                        </div>
                        {/* Quick Stats */}
                        <div className="flex gap-6">
                          <div className="text-center">
                            <div className="text-3xl font-bold text-white">{selectedMatchPlayer.stats.touches}</div>
                            <div className="text-slate-500 text-xs mt-1">TOUCHES</div>
                          </div>
                          <div className="text-center">
                            <div className="text-3xl font-bold text-cyan-400">{selectedMatchPlayer.stats.passes_completed}</div>
                            <div className="text-slate-500 text-xs mt-1">PASSES</div>
                          </div>
                          <div className="text-center">
                            <div className="text-3xl font-bold text-amber-400">{selectedMatchPlayer.stats.shots}</div>
                            <div className="text-slate-500 text-xs mt-1">SHOTS</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Generate Clips Button */}
                    {selectedMatchPlayer.can_generate_clips && selectedMatchPlayer.clips_available === 0 && (
                      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center">
                              <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <div>
                              <p className="text-white font-medium">Generate Player Clips</p>
                              <p className="text-slate-400 text-sm">{selectedMatchPlayer.moments_count} moments ready to extract</p>
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => generatePlayerClips(selectedMatchPlayer.jersey_number)}
                            disabled={generatingClips}
                            className="px-4 py-2 bg-amber-500 text-white rounded-lg font-medium hover:bg-amber-600 transition-colors disabled:opacity-50 flex items-center gap-2"
                          >
                            {generatingClips ? (
                              <>
                                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                                Extracting...
                              </>
                            ) : (
                              'Generate Clips'
                            )}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Player Clips Grid */}
                    <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-white font-semibold">Player Clips</h3>
                        {playerClips.length > 0 && (
                          <span className="text-slate-400 text-sm">{playerClips.length} clips</span>
                        )}
                      </div>

                      {playerClips.length === 0 ? (
                        <div className="text-center py-8">
                          <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-3">
                            <svg className="w-6 h-6 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <p className="text-slate-400 text-sm">No clips extracted yet</p>
                          <p className="text-slate-500 text-xs mt-1">Click "Generate Clips" to extract player moments</p>
                        </div>
                      ) : (
                        <div className="grid grid-cols-3 gap-4">
                          {playerClips.map((clip) => (
                            <div key={clip.clip_id} className="bg-slate-800/50 rounded-xl overflow-hidden hover:bg-slate-800 transition-colors">
                              <div className="aspect-video bg-slate-900 flex items-center justify-center relative">
                                <svg className="w-12 h-12 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-0.5 rounded">
                                  {clip.duration_seconds.toFixed(1)}s
                                </span>
                              </div>
                              <div className="p-3">
                                <div className="flex items-center justify-between">
                                  <span className={`text-xs px-2 py-0.5 rounded ${
                                    clip.event_type === 'goal' ? 'bg-green-500/20 text-green-400' :
                                    clip.event_type === 'shot' ? 'bg-red-500/20 text-red-400' :
                                    clip.event_type === 'pass' ? 'bg-blue-500/20 text-blue-400' :
                                    'bg-slate-700 text-slate-300'
                                  }`}>
                                    {clip.event_type}
                                  </span>
                                  <span className="text-slate-500 text-xs">
                                    {Math.floor(clip.timestamp_start_ms / 60000)}:{String(Math.floor((clip.timestamp_start_ms % 60000) / 1000)).padStart(2, '0')}
                                  </span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Stats Summary */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
                        <div className="text-slate-400 text-xs mb-1">Passes</div>
                        <div className="text-2xl font-bold text-white">{selectedMatchPlayer.stats.passes_completed}/{selectedMatchPlayer.stats.passes_attempted}</div>
                      </div>
                      <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
                        <div className="text-slate-400 text-xs mb-1">Shots</div>
                        <div className="text-2xl font-bold text-white">{selectedMatchPlayer.stats.shots}</div>
                      </div>
                      <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
                        <div className="text-slate-400 text-xs mb-1">Tackles</div>
                        <div className="text-2xl font-bold text-white">{selectedMatchPlayer.stats.tackles}</div>
                      </div>
                      <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
                        <div className="text-slate-400 text-xs mb-1">Interceptions</div>
                        <div className="text-2xl font-bold text-white">{selectedMatchPlayer.stats.interceptions}</div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="bg-[#111827] rounded-2xl p-16 border border-slate-700/30 text-center">
                    <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-6">
                      <svg className="w-12 h-12 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">Select a Player</h3>
                    <p className="text-slate-400 text-sm max-w-md mx-auto">
                      Choose a player from the list to view their stats and generate individual video clips
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* External Clips View (Original) */}
      {viewMode === 'external' && (
        <>
          {/* Import Section - Compact */}
          <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <input
                  type="text"
                  value={clipPath}
                  onChange={(e) => setClipPath(e.target.value)}
                  placeholder="Path to player clip (e.g., C:/clips/player.mp4)"
                  className="w-full bg-slate-800 text-white rounded-lg px-4 py-2.5 text-sm border border-slate-700 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div className="w-48">
                <input
                  type="text"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  placeholder="Player Name"
                  className="w-full bg-slate-800 text-white rounded-lg px-4 py-2.5 text-sm border border-slate-700 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <button
                type="button"
                onClick={analyzeClip}
                disabled={analyzingClip || !clipPath || !playerName}
                className="px-6 py-2.5 bg-cyan-500 text-white rounded-lg text-sm font-medium hover:bg-cyan-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {analyzingClip ? (
                  <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                ) : (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                )}
                Import Clip
              </button>
            </div>
          </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-4 gap-6">
        {/* Player List - Left Sidebar */}
        <div className="bg-[#111827] rounded-2xl border border-slate-700/30 overflow-hidden">
          <div className="p-4 border-b border-slate-700/30">
            <h3 className="text-white font-semibold">Squad</h3>
          </div>
          <div className="p-2 max-h-[600px] overflow-y-auto">
            {players.length === 0 ? (
              <div className="text-center py-12 px-4">
                <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <p className="text-slate-400 text-sm">No players yet</p>
                <p className="text-slate-500 text-xs mt-1">Import a clip to add players</p>
              </div>
            ) : (
              <div className="space-y-1">
                {players.map((player) => (
                  <button
                    key={player.name}
                    onClick={() => loadPlayerStats(player.name)}
                    className={`w-full text-left p-3 rounded-xl transition-all ${
                      selectedPlayer?.player_name === player.name
                        ? 'bg-cyan-500/20 border-l-4 border-cyan-500'
                        : 'hover:bg-slate-800/50'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-red-600 rounded-lg flex items-center justify-center flex-shrink-0">
                        <span className="text-white font-bold">{player.name.charAt(0)}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-white font-medium truncate">{player.name}</div>
                        <div className="text-slate-500 text-xs">{player.ball_touches} touches • {Math.round(player.play_time_seconds / 60)}m</div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Player Detail - Main Content */}
        <div className="col-span-3 space-y-4">
          {selectedPlayer ? (
            <>
              {/* Player Header Card */}
              <div className="bg-gradient-to-r from-[#111827] to-[#1a2744] rounded-2xl p-6 border border-slate-700/30">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-5">
                    <div className="w-20 h-20 bg-gradient-to-br from-red-500 to-red-600 rounded-2xl flex items-center justify-center shadow-lg shadow-red-500/20">
                      <span className="text-white text-3xl font-bold">{selectedPlayer.player_name.charAt(0)}</span>
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-white">{selectedPlayer.player_name}</h2>
                      <p className="text-slate-400 text-sm mt-1">Midfielder • Your Team</p>
                      <div className="flex items-center gap-4 mt-3">
                        <div className="flex items-center gap-1.5">
                          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                          <span className="text-slate-300 text-sm">{selectedPlayer.total_clips} clips</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="text-slate-300 text-sm">{Math.round(selectedPlayer.total_play_time_seconds / 60)} minutes analyzed</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="flex gap-6">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-white">{selectedPlayer.attacking.ball_touches}</div>
                      <div className="text-slate-500 text-xs mt-1">TOUCHES</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-cyan-400">{selectedPlayer.attacking.pass_accuracy.toFixed(0)}%</div>
                      <div className="text-slate-500 text-xs mt-1">PASS ACC</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-400">{selectedPlayer.defensive.tackle_success_rate.toFixed(0)}%</div>
                      <div className="text-slate-500 text-xs mt-1">TACKLE</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Tab Navigation */}
              <div className="flex gap-1 bg-[#111827] p-1 rounded-xl border border-slate-700/30">
                {[
                  { id: 'radar', label: 'Performance Radar', icon: '📊' },
                  { id: 'stats', label: 'Detailed Stats', icon: '📈' },
                  { id: 'feedback', label: 'Coaching Feedback', icon: '💬' },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex-1 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                      activeTab === tab.id
                        ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/20'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              {activeTab === 'radar' && (
                <div className="grid grid-cols-2 gap-4">
                  {/* HUDL-style Player Radar */}
                  <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
                    <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
                      <span className="text-cyan-400">◆</span> Trait Radar
                    </h3>
                    <div className="relative aspect-square max-w-[280px] mx-auto">
                      <PlayerRadarChart player={selectedPlayer} />
                    </div>
                    <div className="mt-4 text-center">
                      <p className="text-slate-500 text-xs">Compared to squad average</p>
                    </div>
                  </div>

                  {/* Performance Summary Cards */}
                  <div className="space-y-4">
                    <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-white font-semibold">Attacking Output</h4>
                        <span className="text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400">Offensive</span>
                      </div>
                      <div className="space-y-3">
                        <ProgressBar label="Pass Accuracy" value={selectedPlayer.attacking.pass_accuracy} color="cyan" />
                        <ProgressBar label="Shot Accuracy" value={selectedPlayer.attacking.shot_accuracy} color="red" />
                        <ProgressBar label="Chance Creation" value={Math.min(100, (selectedPlayer.attacking.passes_completed / Math.max(1, selectedPlayer.attacking.passes_attempted)) * 100)} color="amber" />
                      </div>
                    </div>

                    <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-white font-semibold">Defensive Output</h4>
                        <span className="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-400">Defensive</span>
                      </div>
                      <div className="space-y-3">
                        <ProgressBar label="Tackle Success" value={selectedPlayer.defensive.tackle_success_rate} color="blue" />
                        <ProgressBar label="Aerial Duels" value={Math.min(100, selectedPlayer.defensive.headers * 20)} color="purple" />
                        <ProgressBar label="Interceptions" value={Math.min(100, selectedPlayer.defensive.interceptions * 15)} color="green" />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'stats' && (
                <div className="grid grid-cols-3 gap-4">
                  {/* Attacking Stats */}
                  <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
                    <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-700/50">
                      <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center">
                        <span className="text-red-400 text-sm">⚔</span>
                      </div>
                      <h4 className="text-white font-semibold">Attacking</h4>
                    </div>
                    <div className="space-y-4">
                      <StatRow label="Ball Touches" value={selectedPlayer.attacking.ball_touches} />
                      <StatRow label="Passes Attempted" value={selectedPlayer.attacking.passes_attempted} />
                      <StatRow label="Passes Completed" value={selectedPlayer.attacking.passes_completed} />
                      <StatRow label="Pass Accuracy" value={`${selectedPlayer.attacking.pass_accuracy.toFixed(1)}%`} highlight />
                      <div className="pt-3 border-t border-slate-700/50">
                        <StatRow label="Shots" value={selectedPlayer.attacking.shots} />
                        <StatRow label="On Target" value={selectedPlayer.attacking.shots_on_target} />
                        <StatRow label="Shot Accuracy" value={`${selectedPlayer.attacking.shot_accuracy.toFixed(1)}%`} highlight />
                      </div>
                    </div>
                  </div>

                  {/* Defensive Stats */}
                  <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
                    <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-700/50">
                      <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                        <span className="text-blue-400 text-sm">🛡</span>
                      </div>
                      <h4 className="text-white font-semibold">Defensive</h4>
                    </div>
                    <div className="space-y-4">
                      <StatRow label="Tackles Attempted" value={selectedPlayer.defensive.tackles_attempted} />
                      <StatRow label="Tackles Won" value={selectedPlayer.defensive.tackles_won} />
                      <StatRow label="Tackle Success" value={`${selectedPlayer.defensive.tackle_success_rate.toFixed(1)}%`} highlight />
                      <div className="pt-3 border-t border-slate-700/50">
                        <StatRow label="Headers Won" value={selectedPlayer.defensive.headers} />
                        <StatRow label="Interceptions" value={selectedPlayer.defensive.interceptions} />
                      </div>
                    </div>
                  </div>

                  {/* Physical Stats */}
                  <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
                    <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-700/50">
                      <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center">
                        <span className="text-green-400 text-sm">🏃</span>
                      </div>
                      <h4 className="text-white font-semibold">Physical</h4>
                    </div>
                    <div className="space-y-4">
                      <StatRow label="Distance (est.)" value={`${selectedPlayer.physical.distance_covered_meters_estimate}m`} highlight />
                      <StatRow label="Sprints" value={selectedPlayer.physical.sprints} />
                      <StatRow label="High Intensity Runs" value={Math.round(selectedPlayer.physical.sprints * 1.5)} />
                    </div>
                    <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                      <p className="text-slate-500 text-xs">
                        Physical metrics estimated from tracking data. GPS data would provide more accuracy.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'feedback' && (
                <div className="space-y-4">
                  {/* Coach's Notes Header */}
                  <div className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 rounded-xl p-4 border border-amber-500/20">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-amber-500/20 rounded-lg flex items-center justify-center">
                        <span className="text-amber-400 text-lg">📋</span>
                      </div>
                      <div>
                        <h3 className="text-white font-semibold">Coaching Analysis</h3>
                        <p className="text-slate-400 text-sm">AI-generated feedback based on performance data</p>
                      </div>
                    </div>
                  </div>

                  {/* Feedback Cards */}
                  <div className="grid grid-cols-1 gap-3">
                    {generateFeedback(selectedPlayer).map((item, idx) => (
                      <div
                        key={idx}
                        className={`bg-[#111827] rounded-xl p-4 border ${
                          item.type === 'strength' ? 'border-green-500/30' :
                          item.type === 'improvement' ? 'border-red-500/30' :
                          'border-amber-500/30'
                        }`}
                      >
                        <div className="flex items-start gap-4">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                            item.type === 'strength' ? 'bg-green-500/20' :
                            item.type === 'improvement' ? 'bg-red-500/20' :
                            'bg-amber-500/20'
                          }`}>
                            {item.type === 'strength' ? (
                              <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            ) : item.type === 'improvement' ? (
                              <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                              </svg>
                            ) : (
                              <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                              </svg>
                            )}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className={`text-xs px-2 py-0.5 rounded ${
                                item.type === 'strength' ? 'bg-green-500/20 text-green-400' :
                                item.type === 'improvement' ? 'bg-red-500/20 text-red-400' :
                                'bg-amber-500/20 text-amber-400'
                              }`}>
                                {item.category}
                              </span>
                              <span className={`text-xs ${
                                item.type === 'strength' ? 'text-green-400' :
                                item.type === 'improvement' ? 'text-red-400' :
                                'text-amber-400'
                              }`}>
                                {item.type === 'strength' ? 'Strength' : item.type === 'improvement' ? 'Needs Work' : 'Focus Area'}
                              </span>
                            </div>
                            <h4 className="text-white font-medium">{item.message}</h4>
                            <p className="text-slate-400 text-sm mt-1">{item.detail}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Training Recommendations */}
                  <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
                    <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
                      <span className="text-cyan-400">🎯</span> Recommended Drills
                    </h4>
                    <div className="grid grid-cols-2 gap-3">
                      {selectedPlayer.attacking.pass_accuracy < 75 && (
                        <DrillCard title="Passing Circuits" duration="15 min" focus="Short passing under pressure" />
                      )}
                      {selectedPlayer.attacking.shot_accuracy < 50 && (
                        <DrillCard title="Finishing Practice" duration="20 min" focus="Shot placement and composure" />
                      )}
                      {selectedPlayer.defensive.tackle_success_rate < 60 && (
                        <DrillCard title="1v1 Defending" duration="15 min" focus="Tackling technique and timing" />
                      )}
                      <DrillCard title="Ball Retention" duration="10 min" focus="Keeping possession under pressure" />
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-[#111827] rounded-2xl p-16 border border-slate-700/30 text-center">
              <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-6">
                <svg className="w-12 h-12 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Select a Player</h3>
              <p className="text-slate-400 text-sm max-w-md mx-auto">
                Choose a player from the squad list to view their performance radar, detailed statistics, and coaching feedback.
              </p>
            </div>
          )}
        </div>
      </div>
        </>
      )}
    </div>
  );
}

// HUDL-style Player Radar Chart Component
function PlayerRadarChart({ player }: { player: PlayerStats }) {
  const metrics = [
    { label: 'Passing', value: player.attacking.pass_accuracy, max: 100 },
    { label: 'Shooting', value: player.attacking.shot_accuracy, max: 100 },
    { label: 'Tackling', value: player.defensive.tackle_success_rate, max: 100 },
    { label: 'Headers', value: Math.min(100, player.defensive.headers * 15), max: 100 },
    { label: 'Work Rate', value: Math.min(100, player.physical.sprints * 8), max: 100 },
    { label: 'Involvement', value: Math.min(100, (player.attacking.ball_touches / Math.max(1, player.total_play_time_seconds / 60)) * 30), max: 100 },
  ];

  const numPoints = metrics.length;
  const angleStep = (2 * Math.PI) / numPoints;
  const centerX = 140;
  const centerY = 140;
  const maxRadius = 100;

  // Generate polygon points
  const points = metrics.map((m, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const radius = (m.value / m.max) * maxRadius;
    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  });

  const polygonPoints = points.map(p => `${p.x},${p.y}`).join(' ');

  // Grid lines
  const gridLevels = [0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox="0 0 280 280" className="w-full h-full">
      {/* Background grid */}
      {gridLevels.map((level, i) => (
        <polygon
          key={i}
          points={metrics.map((_, idx) => {
            const angle = idx * angleStep - Math.PI / 2;
            const radius = level * maxRadius;
            return `${centerX + radius * Math.cos(angle)},${centerY + radius * Math.sin(angle)}`;
          }).join(' ')}
          fill="none"
          stroke="#334155"
          strokeWidth="1"
          opacity={0.5}
        />
      ))}

      {/* Axis lines */}
      {metrics.map((_, i) => {
        const angle = i * angleStep - Math.PI / 2;
        return (
          <line
            key={i}
            x1={centerX}
            y1={centerY}
            x2={centerX + maxRadius * Math.cos(angle)}
            y2={centerY + maxRadius * Math.sin(angle)}
            stroke="#334155"
            strokeWidth="1"
            opacity={0.5}
          />
        );
      })}

      {/* Data polygon */}
      <polygon
        points={polygonPoints}
        fill="rgba(6, 182, 212, 0.2)"
        stroke="#06b6d4"
        strokeWidth="2"
      />

      {/* Data points */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r="4"
          fill="#06b6d4"
          stroke="#0f172a"
          strokeWidth="2"
        />
      ))}

      {/* Labels */}
      {metrics.map((m, i) => {
        const angle = i * angleStep - Math.PI / 2;
        const labelRadius = maxRadius + 25;
        const x = centerX + labelRadius * Math.cos(angle);
        const y = centerY + labelRadius * Math.sin(angle);
        return (
          <text
            key={i}
            x={x}
            y={y}
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-slate-400 text-[10px]"
          >
            {m.label}
          </text>
        );
      })}
    </svg>
  );
}

// Progress Bar Component
function ProgressBar({ label, value, color }: { label: string; value: number; color: string }) {
  const colorClasses: Record<string, string> = {
    cyan: 'bg-cyan-500',
    red: 'bg-red-500',
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    amber: 'bg-amber-500',
    purple: 'bg-purple-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-medium">{value.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClasses[color]} rounded-full transition-all duration-500`}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
    </div>
  );
}

// Stat Row Component
function StatRow({ label, value, highlight }: { label: string; value: string | number; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-slate-400 text-sm">{label}</span>
      <span className={`font-semibold ${highlight ? 'text-cyan-400' : 'text-white'}`}>{value}</span>
    </div>
  );
}

// Drill Card Component
function DrillCard({ title, duration, focus }: { title: string; duration: string; focus: string }) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30">
      <div className="flex items-center justify-between mb-1">
        <span className="text-white font-medium text-sm">{title}</span>
        <span className="text-slate-500 text-xs">{duration}</span>
      </div>
      <p className="text-slate-400 text-xs">{focus}</p>
    </div>
  );
}

function StatItem({ label, value, highlight }: { label: string; value: string | number; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400 text-sm">{label}</span>
      <span className={`font-semibold ${highlight ? 'text-cyan-400' : 'text-white'}`}>{value}</span>
    </div>
  );
}

function PerformanceGauge({ label, value, color }: { label: string; value: number; color: string }) {
  const colorMap: Record<string, string> = {
    green: 'from-green-500 to-green-400',
    red: 'from-red-500 to-red-400',
    blue: 'from-blue-500 to-blue-400',
    yellow: 'from-yellow-500 to-yellow-400',
  };

  return (
    <div className="text-center">
      <div className="relative w-20 h-20 mx-auto mb-2">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
          <circle
            cx="18" cy="18" r="15.5"
            fill="none" stroke="#1e293b" strokeWidth="3"
          />
          <circle
            cx="18" cy="18" r="15.5"
            fill="none"
            stroke="url(#gradient)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={`${value * 0.97} 97`}
          />
          <defs>
            <linearGradient id="gradient">
              <stop offset="0%" className={`[stop-color:theme(colors.${color}.500)]`} />
              <stop offset="100%" className={`[stop-color:theme(colors.${color}.400)]`} />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-white font-bold text-sm">{Math.round(value)}%</span>
        </div>
      </div>
      <span className="text-slate-400 text-xs">{label}</span>
    </div>
  );
}

// ==================== PREDICTIVE TRACKING VIEW ====================
interface TrackedPlayerData {
  track_id: number;
  team: string;
  state: 'visible' | 'predicted' | 'occluded' | 'lost';
  last_seen_frame: number;
  frames_missing: number;
  current_position: [number, number] | null;
  predicted_position: [number, number] | null;
  velocity: [number, number];
  confidence: number;
  reentry_prediction: {
    predicted_frame: number;
    predicted_position: [number, number];
    entry_side: string;
    confidence: number;
  } | null;
  trajectory: Array<[number, number]>;
}

// ==================== JERSEY DETECTION VIEW ====================
interface JerseyDetection {
  jersey_number: number;
  team: string;
  confidence: number;
  observation_count: number;
  confirmed: boolean;
  manually_corrected: boolean;
  pending?: boolean;
}

// ==================== 2D RADAR VIEW (VEO-STYLE) ====================
interface RadarPlayer {
  jersey_number: number;
  x: number;
  y: number;
  has_ball?: boolean;
}

interface RadarState {
  frame_number: number;
  timestamp_ms?: number;
  players: {
    home: RadarPlayer[];
    away: RadarPlayer[];
  };
  ball: { x: number; y: number } | null;
}

interface TeamShape {
  players: { jersey: number; x: number; y: number }[];
  centroid: { x: number; y: number } | null;
  width: number;
  depth: number;
  compactness: number;
}

function Radar2DView({ analysis }: { analysis: MatchAnalysis | null }) {
  const [radarState, setRadarState] = useState<RadarState | null>(null);
  const [homeShape, setHomeShape] = useState<TeamShape | null>(null);
  const [awayShape, setAwayShape] = useState<TeamShape | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTrails, setShowTrails] = useState(false);
  const [showFormationLines, setShowFormationLines] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  // Pitch dimensions for drawing (normalized to 0-100)
  const PITCH_WIDTH = 400;
  const PITCH_HEIGHT = 260;
  const PADDING = 20;

  const fetchRadarState = async (frame?: number) => {
    try {
      const url = frame !== undefined
        ? `http://localhost:8000/api/visualization/2d-radar?frame_number=${frame}`
        : 'http://localhost:8000/api/visualization/2d-radar';
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        setRadarState(data);
        if (data.frame_number) {
          setCurrentFrame(data.frame_number);
        }
      }
    } catch (err) {
      console.error('Failed to fetch radar state:', err);
    }
  };

  const fetchTeamShapes = async (frame: number) => {
    try {
      const [homeRes, awayRes] = await Promise.all([
        fetch(`http://localhost:8000/api/visualization/team-shape/home?frame_number=${frame}`),
        fetch(`http://localhost:8000/api/visualization/team-shape/away?frame_number=${frame}`)
      ]);
      if (homeRes.ok) setHomeShape(await homeRes.json());
      if (awayRes.ok) setAwayShape(await awayRes.json());
    } catch (err) {
      console.error('Failed to fetch team shapes:', err);
    }
  };

  useEffect(() => {
    fetchRadarState();
    const interval = setInterval(() => {
      if (!isPlaying) fetchRadarState();
    }, 2000);
    return () => clearInterval(interval);
  }, [isPlaying]);

  useEffect(() => {
    if (currentFrame > 0) {
      fetchTeamShapes(currentFrame);
    }
  }, [currentFrame]);

  // Animation loop for playback
  useEffect(() => {
    if (isPlaying && analysis) {
      const fps = analysis.fps_analyzed || 3;
      const interval = 1000 / (fps * playbackSpeed);

      const animate = () => {
        setCurrentFrame(prev => {
          const next = prev + 1;
          if (next >= analysis.total_frames) {
            setIsPlaying(false);
            return prev;
          }
          fetchRadarState(next);
          return next;
        });
        animationRef.current = window.setTimeout(animate, interval) as any;
      };

      animationRef.current = window.setTimeout(animate, interval) as any;

      return () => {
        if (animationRef.current) {
          clearTimeout(animationRef.current as any);
        }
      };
    }
  }, [isPlaying, playbackSpeed, analysis]);

  // Draw the pitch on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a472a'; // Dark green pitch
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw pitch markings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.lineWidth = 2;

    // Outer boundary
    ctx.strokeRect(PADDING, PADDING, PITCH_WIDTH, PITCH_HEIGHT);

    // Center line
    ctx.beginPath();
    ctx.moveTo(PADDING + PITCH_WIDTH / 2, PADDING);
    ctx.lineTo(PADDING + PITCH_WIDTH / 2, PADDING + PITCH_HEIGHT);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(PADDING + PITCH_WIDTH / 2, PADDING + PITCH_HEIGHT / 2, 30, 0, Math.PI * 2);
    ctx.stroke();

    // Center spot
    ctx.beginPath();
    ctx.arc(PADDING + PITCH_WIDTH / 2, PADDING + PITCH_HEIGHT / 2, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.fill();

    // Penalty areas (left)
    ctx.strokeRect(PADDING, PADDING + PITCH_HEIGHT / 2 - 55, 55, 110);
    // Goal area (left)
    ctx.strokeRect(PADDING, PADDING + PITCH_HEIGHT / 2 - 25, 18, 50);
    // Penalty spot (left)
    ctx.beginPath();
    ctx.arc(PADDING + 36, PADDING + PITCH_HEIGHT / 2, 3, 0, Math.PI * 2);
    ctx.fill();

    // Penalty areas (right)
    ctx.strokeRect(PADDING + PITCH_WIDTH - 55, PADDING + PITCH_HEIGHT / 2 - 55, 55, 110);
    // Goal area (right)
    ctx.strokeRect(PADDING + PITCH_WIDTH - 18, PADDING + PITCH_HEIGHT / 2 - 25, 18, 50);
    // Penalty spot (right)
    ctx.beginPath();
    ctx.arc(PADDING + PITCH_WIDTH - 36, PADDING + PITCH_HEIGHT / 2, 3, 0, Math.PI * 2);
    ctx.fill();

    // Goals
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.fillRect(PADDING - 8, PADDING + PITCH_HEIGHT / 2 - 15, 8, 30);
    ctx.fillRect(PADDING + PITCH_WIDTH, PADDING + PITCH_HEIGHT / 2 - 15, 8, 30);

    // Draw formation lines if enabled
    if (showFormationLines && homeShape && homeShape.players.length > 0) {
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)'; // Blue with low opacity
      ctx.lineWidth = 1;

      // Draw convex hull / formation shape for home team
      const homePoints = homeShape.players.map(p => ({
        x: PADDING + (p.x / 100) * PITCH_WIDTH,
        y: PADDING + (p.y / 100) * PITCH_HEIGHT
      }));

      if (homePoints.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(homePoints[0].x, homePoints[0].y);
        homePoints.forEach((p, i) => {
          if (i > 0) ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
        ctx.fill();
        ctx.stroke();
      }
    }

    if (showFormationLines && awayShape && awayShape.players.length > 0) {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)'; // Red with low opacity
      ctx.lineWidth = 1;

      const awayPoints = awayShape.players.map(p => ({
        x: PADDING + (p.x / 100) * PITCH_WIDTH,
        y: PADDING + (p.y / 100) * PITCH_HEIGHT
      }));

      if (awayPoints.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(awayPoints[0].x, awayPoints[0].y);
        awayPoints.forEach((p, i) => {
          if (i > 0) ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
        ctx.fill();
        ctx.stroke();
      }
    }

    // Draw players if we have radar state
    if (radarState) {
      // Home players (blue)
      radarState.players.home.forEach(player => {
        const x = PADDING + (player.x / 100) * PITCH_WIDTH;
        const y = PADDING + (player.y / 100) * PITCH_HEIGHT;

        // Player circle
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fillStyle = player.has_ball ? '#22c55e' : '#3b82f6'; // Green if has ball, else blue
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Jersey number
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(player.jersey_number.toString(), x, y);
      });

      // Away players (red)
      radarState.players.away.forEach(player => {
        const x = PADDING + (player.x / 100) * PITCH_WIDTH;
        const y = PADDING + (player.y / 100) * PITCH_HEIGHT;

        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fillStyle = player.has_ball ? '#22c55e' : '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(player.jersey_number.toString(), x, y);
      });

      // Ball (yellow)
      if (radarState.ball) {
        const bx = PADDING + (radarState.ball.x / 100) * PITCH_WIDTH;
        const by = PADDING + (radarState.ball.y / 100) * PITCH_HEIGHT;

        ctx.beginPath();
        ctx.arc(bx, by, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw team centroids if shapes available
      if (homeShape?.centroid) {
        const cx = PADDING + (homeShape.centroid.x / 100) * PITCH_WIDTH;
        const cy = PADDING + (homeShape.centroid.y / 100) * PITCH_HEIGHT;
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(59, 130, 246, 0.5)';
        ctx.fill();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#3b82f6';
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (awayShape?.centroid) {
        const cx = PADDING + (awayShape.centroid.x / 100) * PITCH_WIDTH;
        const cy = PADDING + (awayShape.centroid.y / 100) * PITCH_HEIGHT;
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.fill();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#ef4444';
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

  }, [radarState, homeShape, awayShape, showFormationLines, showTrails]);

  const handleFrameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const frame = parseInt(e.target.value);
    setCurrentFrame(frame);
    fetchRadarState(frame);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="text-2xl">📍</span>
              2D Tactical Radar
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Real-time overhead view of player positions and team shape
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
              <span className="text-sm text-slate-300">Home</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <span className="text-sm text-slate-300">Away</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-500 rounded-full"></div>
              <span className="text-sm text-slate-300">Ball</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4 mb-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              isPlaying
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-cyan-500 hover:bg-cyan-600 text-white'
            }`}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>

          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            className="bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>

          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showFormationLines}
              onChange={(e) => setShowFormationLines(e.target.checked)}
              className="rounded"
            />
            Show Formation
          </label>
        </div>

        {/* Frame slider */}
        {analysis && (
          <div className="flex items-center gap-4">
            <span className="text-sm text-slate-400">Frame:</span>
            <input
              type="range"
              min={0}
              max={analysis.total_frames - 1}
              value={currentFrame}
              onChange={handleFrameChange}
              className="flex-1 accent-cyan-500"
            />
            <span className="text-sm text-slate-300 min-w-[80px]">
              {currentFrame} / {analysis.total_frames}
            </span>
          </div>
        )}
      </div>

      {/* Main Pitch View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
            <canvas
              ref={canvasRef}
              width={440}
              height={300}
              className="w-full rounded-lg"
              style={{ maxWidth: '100%', height: 'auto' }}
            />
          </div>
        </div>

        {/* Team Stats Panel */}
        <div className="space-y-4">
          {/* Home Team */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-blue-500/30">
            <h3 className="font-bold text-blue-400 mb-3 flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              Home Team
            </h3>
            {homeShape && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Players Tracked:</span>
                  <span className="text-white">{homeShape.players.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Team Width:</span>
                  <span className="text-white">{homeShape.width.toFixed(1)}m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Team Depth:</span>
                  <span className="text-white">{homeShape.depth.toFixed(1)}m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Compactness:</span>
                  <span className="text-white">{homeShape.compactness.toFixed(1)}</span>
                </div>
              </div>
            )}
            {radarState && (
              <div className="mt-3 pt-3 border-t border-slate-700">
                <span className="text-slate-400 text-xs">Jersey Numbers:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {radarState.players.home.map(p => (
                    <span key={p.jersey_number} className="px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded text-xs">
                      #{p.jersey_number}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Away Team */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-red-500/30">
            <h3 className="font-bold text-red-400 mb-3 flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              Away Team
            </h3>
            {awayShape && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Players Tracked:</span>
                  <span className="text-white">{awayShape.players.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Team Width:</span>
                  <span className="text-white">{awayShape.width.toFixed(1)}m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Team Depth:</span>
                  <span className="text-white">{awayShape.depth.toFixed(1)}m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Compactness:</span>
                  <span className="text-white">{awayShape.compactness.toFixed(1)}</span>
                </div>
              </div>
            )}
            {radarState && (
              <div className="mt-3 pt-3 border-t border-slate-700">
                <span className="text-slate-400 text-xs">Jersey Numbers:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {radarState.players.away.map(p => (
                    <span key={p.jersey_number} className="px-2 py-0.5 bg-red-500/20 text-red-300 rounded text-xs">
                      #{p.jersey_number}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Ball Info */}
          {radarState?.ball && (
            <div className="bg-slate-800/50 rounded-xl p-4 border border-yellow-500/30">
              <h3 className="font-bold text-yellow-400 mb-2 flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                Ball Position
              </h3>
              <div className="text-sm space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-400">X:</span>
                  <span className="text-white">{radarState.ball.x.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Y:</span>
                  <span className="text-white">{radarState.ball.y.toFixed(1)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ==================== PLAYER SPOTLIGHT VIEW (VEO-STYLE) ====================
interface PlayerMoment {
  id: string;
  track_id: string;
  jersey_number: number;
  team: 'home' | 'away';
  start_frame: number;
  end_frame: number;
  start_time_ms: number;
  end_time_ms: number;
  moment_type: 'ball_touch' | 'pass' | 'shot' | 'dribble' | 'tackle' | 'interception';
  confidence: number;
  description?: string;
}

interface PlayerProfile {
  track_id: string;
  jersey_number: number;
  team: 'home' | 'away';
  total_moments: number;
  ball_touches: number;
  passes: number;
  shots: number;
  tackles: number;
  distance_covered_m: number;
  avg_position: { x: number; y: number };
  heatmap_url?: string;
}

interface SpotlightSettings {
  distance_to_ball_threshold: number; // meters
  time_before_moment: number; // seconds
  time_after_moment: number; // seconds
  min_confidence: number;
}

function PlayerSpotlightView() {
  const [players, setPlayers] = useState<PlayerProfile[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerProfile | null>(null);
  const [moments, setMoments] = useState<PlayerMoment[]>([]);
  const [loading, setLoading] = useState(true);
  const [generatingClips, setGeneratingClips] = useState(false);
  const [settings, setSettings] = useState<SpotlightSettings>({
    distance_to_ball_threshold: 3,
    time_before_moment: 3,
    time_after_moment: 5,
    min_confidence: 0.5
  });
  const [filterTeam, setFilterTeam] = useState<'all' | 'home' | 'away'>('all');
  const [filterMomentType, setFilterMomentType] = useState<string>('all');

  useEffect(() => {
    fetchPlayers();
  }, []);

  useEffect(() => {
    if (selectedPlayer) {
      fetchPlayerMoments(selectedPlayer.track_id);
    }
  }, [selectedPlayer]);

  const fetchPlayers = async () => {
    setLoading(true);
    try {
      // Fetch from jersey detections and combine with tracking data
      const [jerseyRes, trackingRes] = await Promise.all([
        fetch('http://localhost:8000/api/match/current/jersey-detections'),
        fetch('http://localhost:8000/api/visualization/average-positions/home'),
      ]);

      const profiles: PlayerProfile[] = [];

      if (jerseyRes.ok) {
        const jerseyData = await jerseyRes.json();

        // Create profiles from jersey detections
        Object.entries(jerseyData).forEach(([trackId, detection]: [string, any]) => {
          if (detection.confirmed_number) {
            profiles.push({
              track_id: trackId,
              jersey_number: detection.confirmed_number,
              team: detection.team || 'home',
              total_moments: 0,
              ball_touches: 0,
              passes: 0,
              shots: 0,
              tackles: 0,
              distance_covered_m: 0,
              avg_position: { x: 50, y: 50 }
            });
          }
        });
      }

      // If no jersey detections, create placeholder profiles
      if (profiles.length === 0) {
        for (let i = 1; i <= 11; i++) {
          profiles.push({
            track_id: `home_${i}`,
            jersey_number: i,
            team: 'home',
            total_moments: Math.floor(Math.random() * 20),
            ball_touches: Math.floor(Math.random() * 15),
            passes: Math.floor(Math.random() * 10),
            shots: Math.floor(Math.random() * 3),
            tackles: Math.floor(Math.random() * 5),
            distance_covered_m: Math.floor(Math.random() * 3000) + 5000,
            avg_position: { x: 20 + Math.random() * 60, y: 10 + Math.random() * 80 }
          });
        }
        for (let i = 1; i <= 11; i++) {
          profiles.push({
            track_id: `away_${i}`,
            jersey_number: i,
            team: 'away',
            total_moments: Math.floor(Math.random() * 20),
            ball_touches: Math.floor(Math.random() * 15),
            passes: Math.floor(Math.random() * 10),
            shots: Math.floor(Math.random() * 3),
            tackles: Math.floor(Math.random() * 5),
            distance_covered_m: Math.floor(Math.random() * 3000) + 5000,
            avg_position: { x: 20 + Math.random() * 60, y: 10 + Math.random() * 80 }
          });
        }
      }

      setPlayers(profiles);
    } catch (err) {
      console.error('Failed to fetch players:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchPlayerMoments = async (trackId: string) => {
    try {
      // In a full implementation, this would fetch from the backend
      // For now, generate sample moments
      const sampleMoments: PlayerMoment[] = [];
      const momentTypes: PlayerMoment['moment_type'][] = ['ball_touch', 'pass', 'shot', 'dribble', 'tackle', 'interception'];

      const player = players.find(p => p.track_id === trackId);
      if (!player) return;

      for (let i = 0; i < player.total_moments; i++) {
        const startFrame = Math.floor(Math.random() * 5000);
        const momentType = momentTypes[Math.floor(Math.random() * momentTypes.length)];
        sampleMoments.push({
          id: `moment_${i}`,
          track_id: trackId,
          jersey_number: player.jersey_number,
          team: player.team,
          start_frame: startFrame,
          end_frame: startFrame + 150, // ~5 seconds at 30fps
          start_time_ms: startFrame * 33,
          end_time_ms: (startFrame + 150) * 33,
          moment_type: momentType,
          confidence: 0.5 + Math.random() * 0.5,
          description: getMomentDescription(momentType)
        });
      }

      setMoments(sampleMoments.sort((a, b) => a.start_frame - b.start_frame));
    } catch (err) {
      console.error('Failed to fetch player moments:', err);
    }
  };

  const getMomentDescription = (type: PlayerMoment['moment_type']): string => {
    const descriptions: Record<string, string[]> = {
      ball_touch: ['Receives the ball', 'Controls with first touch', 'Takes possession'],
      pass: ['Short pass to teammate', 'Long ball forward', 'Cross into the box'],
      shot: ['Shot on goal', 'Effort from distance', 'Header towards goal'],
      dribble: ['Takes on defender', 'Dribbles past opponent', 'Carries ball forward'],
      tackle: ['Wins the ball back', 'Clean tackle', 'Challenges for possession'],
      interception: ['Reads the play', 'Cuts out the pass', 'Intercepts danger']
    };
    const options = descriptions[type] || ['Player action'];
    return options[Math.floor(Math.random() * options.length)];
  };

  const generateClip = async (moment: PlayerMoment) => {
    setGeneratingClips(true);
    try {
      // Calculate clip boundaries based on settings
      const fps = 30;
      const framesBefore = Math.floor(settings.time_before_moment * fps);
      const framesAfter = Math.floor(settings.time_after_moment * fps);

      const response = await fetch('http://localhost:8000/api/clips/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_frame: Math.max(0, moment.start_frame - framesBefore),
          end_frame: moment.end_frame + framesAfter,
          player_track_id: moment.track_id,
          moment_type: moment.moment_type,
          description: moment.description
        })
      });

      if (response.ok) {
        alert('Clip generated successfully!');
      } else {
        alert('Clip generation endpoint not yet implemented');
      }
    } catch (err) {
      console.error('Failed to generate clip:', err);
      alert('Clip generation requires backend implementation');
    } finally {
      setGeneratingClips(false);
    }
  };

  const filteredPlayers = players.filter(p =>
    filterTeam === 'all' || p.team === filterTeam
  );

  const filteredMoments = moments.filter(m =>
    (filterMomentType === 'all' || m.moment_type === filterMomentType) &&
    m.confidence >= settings.min_confidence
  );

  const formatTime = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="text-2xl">🌟</span>
              Player Spotlight
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Track individual player moments and generate highlight clips
            </p>
          </div>
        </div>

        {/* Settings */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="text-xs text-slate-400 block mb-1">Distance to Ball (m)</label>
            <input
              type="number"
              min={1}
              max={10}
              value={settings.distance_to_ball_threshold}
              onChange={(e) => setSettings({...settings, distance_to_ball_threshold: parseFloat(e.target.value)})}
              className="w-full bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-slate-400 block mb-1">Time Before (sec)</label>
            <input
              type="number"
              min={1}
              max={10}
              value={settings.time_before_moment}
              onChange={(e) => setSettings({...settings, time_before_moment: parseFloat(e.target.value)})}
              className="w-full bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-slate-400 block mb-1">Time After (sec)</label>
            <input
              type="number"
              min={1}
              max={15}
              value={settings.time_after_moment}
              onChange={(e) => setSettings({...settings, time_after_moment: parseFloat(e.target.value)})}
              className="w-full bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-slate-400 block mb-1">Min Confidence</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={settings.min_confidence}
              onChange={(e) => setSettings({...settings, min_confidence: parseFloat(e.target.value)})}
              className="w-full accent-cyan-500"
            />
            <span className="text-xs text-slate-300">{(settings.min_confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Player List */}
        <div className="lg:col-span-1">
          <div className="bg-slate-800/50 rounded-xl border border-slate-700/50">
            <div className="p-4 border-b border-slate-700">
              <h3 className="font-bold text-white mb-2">Select Player</h3>
              <div className="flex gap-2">
                {['all', 'home', 'away'].map(team => (
                  <button
                    key={team}
                    onClick={() => setFilterTeam(team as any)}
                    className={`px-3 py-1 rounded text-sm transition-all ${
                      filterTeam === team
                        ? 'bg-cyan-500 text-white'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {team.charAt(0).toUpperCase() + team.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="max-h-[500px] overflow-y-auto">
              {loading ? (
                <div className="p-8 text-center text-slate-400">Loading players...</div>
              ) : (
                filteredPlayers.map(player => (
                  <button
                    key={player.track_id}
                    onClick={() => setSelectedPlayer(player)}
                    className={`w-full p-4 text-left border-b border-slate-700/50 transition-all hover:bg-slate-700/50 ${
                      selectedPlayer?.track_id === player.track_id ? 'bg-slate-700/50 border-l-4 border-l-cyan-500' : ''
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
                        player.team === 'home' ? 'bg-blue-500' : 'bg-red-500'
                      }`}>
                        {player.jersey_number}
                      </div>
                      <div>
                        <div className="text-white font-medium">
                          #{player.jersey_number}
                          <span className="text-slate-400 text-sm ml-2">
                            ({player.team})
                          </span>
                        </div>
                        <div className="text-xs text-slate-400">
                          {player.total_moments} moments • {player.ball_touches} touches
                        </div>
                      </div>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Player Details & Moments */}
        <div className="lg:col-span-2 space-y-4">
          {selectedPlayer ? (
            <>
              {/* Player Stats Card */}
              <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold text-white ${
                      selectedPlayer.team === 'home' ? 'bg-blue-500' : 'bg-red-500'
                    }`}>
                      {selectedPlayer.jersey_number}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">
                        Player #{selectedPlayer.jersey_number}
                      </h3>
                      <p className="text-slate-400 capitalize">{selectedPlayer.team} Team</p>
                    </div>
                  </div>
                  <button
                    className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg text-sm font-medium transition-all"
                    onClick={() => alert('Generate full highlight reel - coming soon!')}
                  >
                    Generate Highlight Reel
                  </button>
                </div>

                <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-cyan-400">{selectedPlayer.total_moments}</div>
                    <div className="text-xs text-slate-400">Moments</div>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-green-400">{selectedPlayer.ball_touches}</div>
                    <div className="text-xs text-slate-400">Touches</div>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-blue-400">{selectedPlayer.passes}</div>
                    <div className="text-xs text-slate-400">Passes</div>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-yellow-400">{selectedPlayer.shots}</div>
                    <div className="text-xs text-slate-400">Shots</div>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-orange-400">{selectedPlayer.tackles}</div>
                    <div className="text-xs text-slate-400">Tackles</div>
                  </div>
                  <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-purple-400">{(selectedPlayer.distance_covered_m / 1000).toFixed(1)}</div>
                    <div className="text-xs text-slate-400">km Covered</div>
                  </div>
                </div>
              </div>

              {/* Moments Timeline */}
              <div className="bg-slate-800/50 rounded-xl border border-slate-700/50">
                <div className="p-4 border-b border-slate-700 flex items-center justify-between">
                  <h3 className="font-bold text-white">Player Moments</h3>
                  <select
                    value={filterMomentType}
                    onChange={(e) => setFilterMomentType(e.target.value)}
                    className="bg-slate-700 text-white px-3 py-1 rounded text-sm border border-slate-600"
                  >
                    <option value="all">All Types</option>
                    <option value="ball_touch">Ball Touches</option>
                    <option value="pass">Passes</option>
                    <option value="shot">Shots</option>
                    <option value="dribble">Dribbles</option>
                    <option value="tackle">Tackles</option>
                    <option value="interception">Interceptions</option>
                  </select>
                </div>

                <div className="max-h-[400px] overflow-y-auto">
                  {filteredMoments.length === 0 ? (
                    <div className="p-8 text-center text-slate-400">
                      No moments found for this player
                    </div>
                  ) : (
                    filteredMoments.map((moment, idx) => (
                      <div
                        key={moment.id}
                        className="p-4 border-b border-slate-700/50 hover:bg-slate-700/30 transition-all"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                              moment.moment_type === 'shot' ? 'bg-yellow-500/20 text-yellow-400' :
                              moment.moment_type === 'pass' ? 'bg-blue-500/20 text-blue-400' :
                              moment.moment_type === 'tackle' ? 'bg-orange-500/20 text-orange-400' :
                              moment.moment_type === 'dribble' ? 'bg-green-500/20 text-green-400' :
                              'bg-slate-500/20 text-slate-400'
                            }`}>
                              {moment.moment_type === 'shot' ? '⚽' :
                               moment.moment_type === 'pass' ? '➡️' :
                               moment.moment_type === 'tackle' ? '🦶' :
                               moment.moment_type === 'dribble' ? '💨' :
                               moment.moment_type === 'interception' ? '🛡️' : '⚽'}
                            </div>
                            <div>
                              <div className="text-white font-medium capitalize">
                                {moment.moment_type.replace('_', ' ')}
                              </div>
                              <div className="text-sm text-slate-400">
                                {moment.description}
                              </div>
                              <div className="text-xs text-slate-500">
                                {formatTime(moment.start_time_ms)} - {formatTime(moment.end_time_ms)}
                                <span className="ml-2">({(moment.confidence * 100).toFixed(0)}% confidence)</span>
                              </div>
                            </div>
                          </div>
                          <button
                            onClick={() => generateClip(moment)}
                            disabled={generatingClips}
                            className="px-3 py-1 bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 rounded text-sm transition-all disabled:opacity-50"
                          >
                            {generatingClips ? '...' : 'Create Clip'}
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-slate-800/50 rounded-xl p-12 border border-slate-700/50 text-center">
              <div className="text-5xl mb-4">🌟</div>
              <h3 className="text-xl font-bold text-white mb-2">Select a Player</h3>
              <p className="text-slate-400">
                Choose a player from the list to view their moments and generate highlight clips
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ==================== JERSEY DETECTION VIEW ====================
interface JerseyStats {
  provider: string;
  api_calls: number;
  total_players_processed: number;
  successful_detections: number;
  confirmed_players: number;
  pending_observations: number;
  manual_corrections: number;
}

function JerseyDetectionView() {
  const [detections, setDetections] = useState<Record<string, JerseyDetection>>({});
  const [stats, setStats] = useState<JerseyStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [correctionTrackId, setCorrectionTrackId] = useState('');
  const [correctionJerseyNumber, setCorrectionJerseyNumber] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [detectionsRes, statsRes] = await Promise.all([
        fetch('http://localhost:8000/api/match/current/jersey-detections'),
        fetch('http://localhost:8000/api/match/current/jersey-stats')
      ]);

      if (detectionsRes.ok) {
        const data = await detectionsRes.json();
        setDetections(data.detections || {});
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }
    } catch (err) {
      setError('Failed to load jersey detection data');
      console.error(err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleCorrection = async () => {
    if (!correctionTrackId || !correctionJerseyNumber) {
      alert('Please enter both Track ID and Jersey Number');
      return;
    }

    const jerseyNum = parseInt(correctionJerseyNumber);
    if (isNaN(jerseyNum) || jerseyNum < 1 || jerseyNum > 99) {
      alert('Jersey number must be between 1 and 99');
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/match/current/jersey-correction?track_id=${correctionTrackId}&jersey_number=${jerseyNum}`,
        { method: 'POST' }
      );

      if (response.ok) {
        setCorrectionTrackId('');
        setCorrectionJerseyNumber('');
        fetchData(); // Refresh data
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to save correction');
      }
    } catch (err) {
      alert('Failed to save correction');
    }
    setSubmitting(false);
  };

  const handleReset = async () => {
    if (!confirm('Are you sure you want to reset all jersey detections? This cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/jersey-detection/reset', { method: 'POST' });
      if (response.ok) {
        fetchData();
      }
    } catch (err) {
      alert('Failed to reset');
    }
  };

  // Group detections by jersey number for summary
  const jerseyNumberCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    Object.values(detections).forEach(d => {
      counts[d.jersey_number] = (counts[d.jersey_number] || 0) + 1;
    });
    return Object.entries(counts)
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .map(([num, count]) => ({ number: parseInt(num), count }));
  }, [detections]);

  const confirmedDetections = Object.entries(detections).filter(([_, d]) => d.confirmed);
  const pendingDetections = Object.entries(detections).filter(([_, d]) => !d.confirmed);

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">AI Jersey Detection</h2>
          <p className="text-slate-400 text-sm mt-1">
            Powered by GPT-4 Vision • Detects jersey numbers from video frames
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm transition-colors"
          >
            🔄 Refresh
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm transition-colors"
          >
            🗑️ Reset All
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-cyan-400">{stats.provider || 'N/A'}</div>
            <div className="text-xs text-slate-400 mt-1">Provider</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-white">{stats.api_calls}</div>
            <div className="text-xs text-slate-400 mt-1">API Calls</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-white">{stats.total_players_processed}</div>
            <div className="text-xs text-slate-400 mt-1">Players Processed</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-green-400">{stats.successful_detections}</div>
            <div className="text-xs text-slate-400 mt-1">Successful</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-emerald-400">{stats.confirmed_players}</div>
            <div className="text-xs text-slate-400 mt-1">Confirmed</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-yellow-400">{stats.pending_observations}</div>
            <div className="text-xs text-slate-400 mt-1">Pending</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <div className="text-2xl font-bold text-purple-400">{stats.manual_corrections}</div>
            <div className="text-xs text-slate-400 mt-1">Manual Fixes</div>
          </div>
        </div>
      )}

      {/* Jersey Numbers Summary */}
      {jerseyNumberCounts.length > 0 && (
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-3">Detected Jersey Numbers</h3>
          <div className="flex flex-wrap gap-2">
            {jerseyNumberCounts.map(({ number, count }) => (
              <div
                key={number}
                className="px-3 py-2 bg-slate-700/50 rounded-lg border border-slate-600/50 flex items-center gap-2"
              >
                <span className="text-xl font-bold text-cyan-400">#{number}</span>
                <span className="text-xs text-slate-400">({count}x)</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Manual Correction Form */}
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
        <h3 className="text-lg font-semibold text-white mb-3">Manual Correction</h3>
        <p className="text-sm text-slate-400 mb-4">
          Correct a detection by entering the Track ID and the correct jersey number.
        </p>
        <div className="flex gap-4 items-end">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Track ID</label>
            <input
              type="text"
              value={correctionTrackId}
              onChange={e => setCorrectionTrackId(e.target.value)}
              placeholder="e.g. 139"
              className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white w-32 focus:outline-none focus:border-cyan-500"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Jersey Number</label>
            <input
              type="number"
              min="1"
              max="99"
              value={correctionJerseyNumber}
              onChange={e => setCorrectionJerseyNumber(e.target.value)}
              placeholder="1-99"
              className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white w-24 focus:outline-none focus:border-cyan-500"
            />
          </div>
          <button
            onClick={handleCorrection}
            disabled={submitting}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-600 text-white rounded-lg transition-colors"
          >
            {submitting ? 'Saving...' : 'Save Correction'}
          </button>
        </div>
      </div>

      {/* Confirmed Detections */}
      {confirmedDetections.length > 0 && (
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-3">
            ✅ Confirmed Players ({confirmedDetections.length})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400 border-b border-slate-700">
                  <th className="pb-2 pr-4">Track ID</th>
                  <th className="pb-2 pr-4">Jersey #</th>
                  <th className="pb-2 pr-4">Team</th>
                  <th className="pb-2 pr-4">Confidence</th>
                  <th className="pb-2 pr-4">Observations</th>
                  <th className="pb-2">Source</th>
                </tr>
              </thead>
              <tbody>
                {confirmedDetections.map(([trackId, detection]) => (
                  <tr key={trackId} className="border-b border-slate-700/50 text-white">
                    <td className="py-2 pr-4 font-mono text-slate-400">{trackId}</td>
                    <td className="py-2 pr-4">
                      <span className="text-xl font-bold text-cyan-400">#{detection.jersey_number}</span>
                    </td>
                    <td className="py-2 pr-4 capitalize">{detection.team}</td>
                    <td className="py-2 pr-4">{(detection.confidence * 100).toFixed(0)}%</td>
                    <td className="py-2 pr-4">{detection.observation_count}</td>
                    <td className="py-2">
                      {detection.manually_corrected ? (
                        <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs">Manual</span>
                      ) : (
                        <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs">AI</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Pending Detections */}
      {pendingDetections.length > 0 && (
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-3">
            ⏳ Pending Observations ({pendingDetections.length})
          </h3>
          <p className="text-sm text-slate-400 mb-4">
            These detections need more observations to be confirmed (requires 3+ consistent observations).
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {pendingDetections.slice(0, 30).map(([trackId, detection]) => (
              <div
                key={trackId}
                className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/30"
              >
                <div className="text-lg font-bold text-yellow-400">#{detection.jersey_number}</div>
                <div className="text-xs text-slate-400 mt-1">Track: {trackId}</div>
                <div className="text-xs text-slate-500">Obs: {detection.observation_count}</div>
              </div>
            ))}
            {pendingDetections.length > 30 && (
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/30 flex items-center justify-center">
                <span className="text-slate-400">+{pendingDetections.length - 30} more</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {Object.keys(detections).length === 0 && !loading && (
        <div className="bg-slate-800/50 rounded-lg p-8 border border-slate-700/50 text-center">
          <div className="text-4xl mb-4">🔢</div>
          <h3 className="text-xl font-semibold text-white mb-2">No Jersey Detections Yet</h3>
          <p className="text-slate-400 max-w-md mx-auto">
            Process a video to detect jersey numbers using AI. The system will automatically
            identify jersey numbers from player images using GPT-4 Vision.
          </p>
        </div>
      )}
    </div>
  );
}

interface TrackingAnalysis {
  total_frames_processed: number;
  total_unique_players: number;
  home_players_tracked: number;
  away_players_tracked: number;
  avg_visibility_rate: number;
  out_of_frame_predictions_made: number;
  reentry_accuracy: number;
  players: TrackedPlayerData[];
}

function PredictiveTrackingView() {
  const [trackingData, setTrackingData] = useState<TrackingAnalysis | null>(null);
  const [outOfFramePlayers, setOutOfFramePlayers] = useState<TrackedPlayerData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<TrackedPlayerData | null>(null);

  useEffect(() => {
    loadTrackingData();
  }, []);

  const loadTrackingData = async () => {
    try {
      setLoading(true);
      const [analysisRes, outOfFrameRes] = await Promise.all([
        fetch('http://localhost:8000/api/tracking/predictive-analysis'),
        fetch('http://localhost:8000/api/tracking/out-of-frame')
      ]);

      if (analysisRes.ok) {
        const data = await analysisRes.json();
        setTrackingData(data);
      }

      if (outOfFrameRes.ok) {
        const data = await outOfFrameRes.json();
        setOutOfFramePlayers(data.out_of_frame_players || []);
      }

      setLoading(false);
    } catch (err) {
      setError('Failed to load tracking data');
      setLoading(false);
    }
  };

  const loadPlayerTrajectory = async (trackId: number) => {
    try {
      const response = await fetch(`http://localhost:8000/api/tracking/player/${trackId}`);
      if (response.ok) {
        const data = await response.json();
        setSelectedPlayer(data);
      }
    } catch (err) {
      console.error('Failed to load player trajectory:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error || !trackingData) {
    return (
      <div className="bg-[#111827] rounded-2xl p-8 border border-slate-700/30 text-center">
        <div className="text-slate-500 text-5xl mb-4">🎯</div>
        <h3 className="text-white font-semibold mb-2">Predictive Tracking</h3>
        <p className="text-slate-400 text-sm mb-4">
          {error || 'No tracking data available. Run analysis on a video first.'}
        </p>
        <button
          onClick={loadTrackingData}
          className="px-4 py-2 bg-cyan-500 text-white rounded-lg text-sm hover:bg-cyan-600 transition-colors"
        >
          Retry Loading
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Total Players Tracked</div>
          <div className="text-2xl font-bold text-white">{trackingData.total_unique_players}</div>
          <div className="text-slate-500 text-xs mt-1">
            Home: {trackingData.home_players_tracked} | Away: {trackingData.away_players_tracked}
          </div>
        </div>
        <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Frames Processed</div>
          <div className="text-2xl font-bold text-white">{trackingData.total_frames_processed}</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Visibility Rate</div>
          <div className="text-2xl font-bold text-cyan-400">{(trackingData.avg_visibility_rate * 100).toFixed(1)}%</div>
          <div className="text-slate-500 text-xs mt-1">Average across all players</div>
        </div>
        <div className="bg-[#111827] rounded-xl p-5 border border-slate-700/30">
          <div className="text-slate-400 text-xs mb-1">Predictions Made</div>
          <div className="text-2xl font-bold text-amber-400">{trackingData.out_of_frame_predictions_made}</div>
          <div className="text-slate-500 text-xs mt-1">Re-entry accuracy: {(trackingData.reentry_accuracy * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Out of Frame Players */}
      {outOfFramePlayers.length > 0 && (
        <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <span className="text-amber-400">⚠️</span> Currently Out of Frame
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {outOfFramePlayers.map((player) => (
              <div
                key={player.track_id}
                className="bg-slate-800/50 rounded-lg p-4 border border-amber-500/30 cursor-pointer hover:border-amber-500/60 transition-colors"
                onClick={() => loadPlayerTrajectory(player.track_id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${player.team === 'home' ? 'text-red-400' : 'text-blue-400'}`}>
                    Player #{player.track_id}
                  </span>
                  <span className="text-xs px-2 py-1 rounded bg-amber-500/20 text-amber-400">
                    {player.state}
                  </span>
                </div>
                <div className="text-slate-400 text-xs space-y-1">
                  <div>Missing for {player.frames_missing} frames</div>
                  {player.reentry_prediction && (
                    <>
                      <div className="text-cyan-400">
                        Re-entry predicted: Frame {player.reentry_prediction.predicted_frame}
                      </div>
                      <div>Entry side: {player.reentry_prediction.entry_side}</div>
                      <div>Confidence: {(player.reentry_prediction.confidence * 100).toFixed(0)}%</div>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Player List and Trajectory View */}
      <div className="grid grid-cols-3 gap-6">
        {/* Player List */}
        <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <span className="text-cyan-400">👥</span> Tracked Players
          </h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {trackingData.players.slice(0, 30).map((player) => (
              <button
                key={player.track_id}
                onClick={() => loadPlayerTrajectory(player.track_id)}
                className={`w-full text-left p-3 rounded-lg transition-all ${
                  selectedPlayer?.track_id === player.track_id
                    ? 'bg-cyan-500/20 border border-cyan-500/50'
                    : 'bg-slate-800/50 hover:bg-slate-700/50 border border-transparent'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`font-medium ${player.team === 'home' ? 'text-red-400' : 'text-blue-400'}`}>
                    Player #{player.track_id}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    player.state === 'visible' ? 'bg-green-500/20 text-green-400' :
                    player.state === 'predicted' ? 'bg-amber-500/20 text-amber-400' :
                    player.state === 'occluded' ? 'bg-orange-500/20 text-orange-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {player.state}
                  </span>
                </div>
                <div className="text-slate-500 text-xs mt-1">
                  Confidence: {(player.confidence * 100).toFixed(0)}% |
                  Velocity: ({player.velocity[0].toFixed(1)}, {player.velocity[1].toFixed(1)})
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Trajectory Visualization */}
        <div className="col-span-2 bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <span className="text-cyan-400">📍</span> Trajectory Prediction
          </h3>
          {selectedPlayer ? (
            <div className="space-y-4">
              {/* Player Info Header */}
              <div className="flex items-center gap-4 pb-4 border-b border-slate-700/50">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                  selectedPlayer.team === 'home' ? 'bg-red-500/20' : 'bg-blue-500/20'
                }`}>
                  <span className={`text-xl font-bold ${
                    selectedPlayer.team === 'home' ? 'text-red-400' : 'text-blue-400'
                  }`}>
                    #{selectedPlayer.track_id}
                  </span>
                </div>
                <div>
                  <div className="text-white font-semibold">
                    {selectedPlayer.team === 'home' ? 'Home' : 'Away'} Team Player
                  </div>
                  <div className="text-slate-400 text-sm">
                    Last seen: Frame {selectedPlayer.last_seen_frame}
                  </div>
                </div>
              </div>

              {/* Mini Pitch Trajectory Visualization */}
              <div className="relative bg-green-900/30 rounded-xl aspect-[16/9] overflow-hidden border border-green-700/30">
                {/* Pitch markings */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-[1px] h-full bg-white/20"></div>
                </div>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-20 h-20 border border-white/20 rounded-full"></div>

                {/* Trajectory Path */}
                <svg className="absolute inset-0 w-full h-full">
                  {selectedPlayer.trajectory.length > 1 && (
                    <polyline
                      points={selectedPlayer.trajectory.map(([x, y]) =>
                        `${(x / 1920) * 100}%,${(y / 1080) * 100}%`
                      ).join(' ')}
                      fill="none"
                      stroke="cyan"
                      strokeWidth="2"
                      strokeDasharray="4 2"
                      opacity="0.7"
                    />
                  )}
                </svg>

                {/* Current/Predicted Position */}
                {selectedPlayer.predicted_position && (
                  <div
                    className="absolute w-4 h-4 rounded-full bg-amber-500 border-2 border-white transform -translate-x-1/2 -translate-y-1/2 animate-pulse"
                    style={{
                      left: `${(selectedPlayer.predicted_position[0] / 1920) * 100}%`,
                      top: `${(selectedPlayer.predicted_position[1] / 1080) * 100}%`
                    }}
                  />
                )}
                {selectedPlayer.current_position && (
                  <div
                    className={`absolute w-4 h-4 rounded-full border-2 border-white transform -translate-x-1/2 -translate-y-1/2 ${
                      selectedPlayer.team === 'home' ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{
                      left: `${(selectedPlayer.current_position[0] / 1920) * 100}%`,
                      top: `${(selectedPlayer.current_position[1] / 1080) * 100}%`
                    }}
                  />
                )}

                {/* Re-entry Prediction */}
                {selectedPlayer.reentry_prediction && (
                  <div
                    className="absolute w-6 h-6 rounded-full border-2 border-dashed border-cyan-400 transform -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${(selectedPlayer.reentry_prediction.predicted_position[0] / 1920) * 100}%`,
                      top: `${(selectedPlayer.reentry_prediction.predicted_position[1] / 1080) * 100}%`
                    }}
                  />
                )}
              </div>

              {/* Position Details */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <div className="text-slate-400 text-xs mb-1">Current Position</div>
                  <div className="text-white text-sm">
                    {selectedPlayer.current_position
                      ? `(${selectedPlayer.current_position[0].toFixed(0)}, ${selectedPlayer.current_position[1].toFixed(0)})`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <div className="text-slate-400 text-xs mb-1">Predicted Position</div>
                  <div className="text-amber-400 text-sm">
                    {selectedPlayer.predicted_position
                      ? `(${selectedPlayer.predicted_position[0].toFixed(0)}, ${selectedPlayer.predicted_position[1].toFixed(0)})`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <div className="text-slate-400 text-xs mb-1">Velocity</div>
                  <div className="text-cyan-400 text-sm">
                    ({selectedPlayer.velocity[0].toFixed(1)}, {selectedPlayer.velocity[1].toFixed(1)}) px/frame
                  </div>
                </div>
              </div>

              {/* Re-entry Prediction Details */}
              {selectedPlayer.reentry_prediction && (
                <div className="bg-cyan-500/10 rounded-lg p-4 border border-cyan-500/30">
                  <h4 className="text-cyan-400 font-medium mb-2">Re-entry Prediction</h4>
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="text-slate-400 text-xs">Frame</div>
                      <div className="text-white">{selectedPlayer.reentry_prediction.predicted_frame}</div>
                    </div>
                    <div>
                      <div className="text-slate-400 text-xs">Entry Side</div>
                      <div className="text-white capitalize">{selectedPlayer.reentry_prediction.entry_side}</div>
                    </div>
                    <div>
                      <div className="text-slate-400 text-xs">Position</div>
                      <div className="text-white">
                        ({selectedPlayer.reentry_prediction.predicted_position[0].toFixed(0)},
                        {selectedPlayer.reentry_prediction.predicted_position[1].toFixed(0)})
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-400 text-xs">Confidence</div>
                      <div className="text-white">{(selectedPlayer.reentry_prediction.confidence * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="text-slate-500 text-4xl mb-3">📍</div>
              <p className="text-slate-400 text-sm">Select a player to view trajectory</p>
              <p className="text-slate-500 text-xs mt-1">Click on any player in the list</p>
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
        <div className="flex items-center gap-6 text-sm">
          <span className="text-slate-400">Player States:</span>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-green-500"></span>
            <span className="text-slate-300">Visible</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-500"></span>
            <span className="text-slate-300">Predicted</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-orange-500"></span>
            <span className="text-slate-300">Occluded</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500"></span>
            <span className="text-slate-300">Lost</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ==================== STATS VIEW ====================
function StatsView({ analysis, stats }: { analysis: MatchAnalysis; stats: MatchStats }) {
  // Get shot stats
  const shots = useMemo(() => detectShots(analysis), [analysis]);
  const homeShots = shots.filter(s => s.team === 'home');
  const awayShots = shots.filter(s => s.team === 'away');
  const homeShotsOnTarget = homeShots.filter(s => s.on_target);
  const awayShotsOnTarget = awayShots.filter(s => s.on_target);

  return (
    <div className="space-y-6">
      {/* Comparison Radar */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-6">Team Comparison</h3>
        <ComparisonRadar stats={stats} />
      </div>

      {/* Detailed Stats Table */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Match Statistics</h3>
        <div className="space-y-3">
          <StatsRow label="Avg Players Detected" home={stats.avgHomePlayers.toFixed(1)} away={stats.avgAwayPlayers.toFixed(1)} />
          <StatsRow label="Possession %" home={`${stats.possession.home}%`} away={`${stats.possession.away}%`} />
          <StatsRow label="Territorial Advantage" home={`${stats.territorialAdvantage.home}%`} away={`${stats.territorialAdvantage.away}%`} />
          <StatsRow label="Shots" home={homeShots.length.toString()} away={awayShots.length.toString()} />
          <StatsRow label="Shots on Target" home={homeShotsOnTarget.length.toString()} away={awayShotsOnTarget.length.toString()} />
          <StatsRow label="Shot Accuracy" home={`${homeShots.length > 0 ? Math.round((homeShotsOnTarget.length / homeShots.length) * 100) : 0}%`} away={`${awayShots.length > 0 ? Math.round((awayShotsOnTarget.length / awayShots.length) * 100) : 0}%`} />
          <StatsRow label="Pressing Actions" home={stats.pressingActions.home.toString()} away={stats.pressingActions.away.toString()} />
          <StatsRow label="High Intensity Runs" home={stats.highIntensityMoments.home.toString()} away={stats.highIntensityMoments.away.toString()} />
          <StatsRow label="Compact Shape %" home={`${stats.compactness.home}%`} away={`${stats.compactness.away}%`} />
          <StatsRow label="Width Usage" home={`${stats.widthUsage.home}%`} away={`${stats.widthUsage.away}%`} />
        </div>
      </div>

      {/* Period Stats */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4">Stats by Period</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-3 px-2">Period</th>
                <th className="text-center py-3 px-2">Your Possession</th>
                <th className="text-center py-3 px-2">Opp Possession</th>
                <th className="text-center py-3 px-2">Your Pressing</th>
                <th className="text-center py-3 px-2">Dominant</th>
              </tr>
            </thead>
            <tbody>
              {stats.periodStats.map((period, i) => (
                <tr key={i} className="border-b border-slate-700/50">
                  <td className="py-3 px-2 text-white">{period.label}</td>
                  <td className="py-3 px-2 text-center text-red-400">{period.homePossession}%</td>
                  <td className="py-3 px-2 text-center text-slate-400">{period.awayPossession}%</td>
                  <td className="py-3 px-2 text-center text-white">{period.pressing}</td>
                  <td className="py-3 px-2 text-center">
                    <span className={`px-2 py-1 rounded text-xs ${period.dominant === 'home' ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'}`}>
                      {period.dominant === 'home' ? 'You' : 'Opp'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ==================== COMPONENTS ====================

function TeamDisplay({ name, color, players, isHome }: { name: string; color: string; players: number; isHome: boolean }) {
  return (
    <div className={`flex items-center gap-4 ${isHome ? '' : 'flex-row-reverse'}`}>
      <div className={`w-16 h-16 rounded-xl ${color === 'red' ? 'bg-red-500' : 'bg-slate-500'} flex items-center justify-center`}>
        <span className="text-white text-2xl font-bold">{name.charAt(0)}</span>
      </div>
      <div className={isHome ? 'text-left' : 'text-right'}>
        <div className="text-white font-semibold">{name}</div>
        <div className="text-slate-400 text-sm">{players.toFixed(1)} avg players</div>
      </div>
    </div>
  );
}

function StatBox({ label, homeValue, awayValue }: { label: string; homeValue: string; awayValue: string }) {
  return (
    <div className="bg-[#111827] rounded-xl p-4 border border-slate-700/30">
      <div className="text-slate-400 text-xs mb-3">{label}</div>
      <div className="flex items-center justify-between">
        <span className="text-red-400 font-bold text-lg">{homeValue}</span>
        <span className="text-slate-500">vs</span>
        <span className="text-slate-400 font-bold text-lg">{awayValue}</span>
      </div>
    </div>
  );
}

function StatsRow({ label, home, away }: { label: string; home: string; away: string }) {
  return (
    <div className="flex items-center py-2 border-b border-slate-700/30">
      <span className="text-red-400 font-semibold w-20 text-right">{home}</span>
      <span className="flex-1 text-center text-slate-400 text-sm">{label}</span>
      <span className="text-slate-400 font-semibold w-20 text-left">{away}</span>
    </div>
  );
}

function MomentumGraph({ analysis, height, detailed }: { analysis: MatchAnalysis; height: number; detailed?: boolean }) {
  const segments = 60;
  const data = useMemo(() => {
    const result: number[] = [];
    const segmentSize = Math.floor(analysis.frames.length / segments);

    for (let i = 0; i < segments; i++) {
      const start = i * segmentSize;
      const end = Math.min((i + 1) * segmentSize, analysis.frames.length);
      let momentum = 0;

      for (let j = start; j < end; j++) {
        const frame = analysis.frames[j];
        const homePlayers = frame.home_players;
        const awayPlayers = frame.away_players;
        // Calculate momentum based on territorial presence
        const homeX = frame.detections.filter(d => d.team === 'home').reduce((sum, d) => sum + (d.bbox[0] + d.bbox[2]) / 2, 0) / (homePlayers || 1);
        const awayX = frame.detections.filter(d => d.team === 'away').reduce((sum, d) => sum + (d.bbox[0] + d.bbox[2]) / 2, 0) / (awayPlayers || 1);
        momentum += (homeX - VIDEO_WIDTH/2) / VIDEO_WIDTH - (awayX - VIDEO_WIDTH/2) / VIDEO_WIDTH;
      }
      result.push(momentum / (end - start));
    }
    return result;
  }, [analysis]);

  const maxAbs = Math.max(...data.map(Math.abs), 0.1);

  return (
    <div className="relative" style={{ height }}>
      {/* Center line */}
      <div className="absolute left-0 right-0 top-1/2 h-px bg-slate-600"></div>

      {/* Bars */}
      <div className="flex items-center h-full gap-0.5">
        {data.map((value, i) => {
          const normalizedValue = value / maxAbs;
          const barHeight = Math.abs(normalizedValue) * (height / 2 - 4);
          const isPositive = normalizedValue >= 0;

          return (
            <div key={i} className="flex-1 h-full flex items-center">
              <div
                className={`w-full rounded-sm transition-all ${isPositive ? 'bg-red-500' : 'bg-slate-500'}`}
                style={{
                  height: barHeight,
                  marginTop: isPositive ? 'auto' : 0,
                  marginBottom: isPositive ? 0 : 'auto',
                }}
              />
            </div>
          );
        })}
      </div>

      {detailed && (
        <>
          <div className="absolute left-0 top-2 text-xs text-red-400">Your Team ↑</div>
          <div className="absolute left-0 bottom-2 text-xs text-slate-400">Opposition ↓</div>
        </>
      )}
    </div>
  );
}

function MiniPitchHeatmap({ analysis, team }: { analysis: MatchAnalysis; team: 'home' | 'away' }) {
  const grid = useMemo(() => {
    const g: number[][] = Array(6).fill(null).map(() => Array(9).fill(0));
    analysis.frames.forEach(frame => {
      frame.detections.filter(d => d.team === team).forEach(det => {
        const x = Math.min(8, Math.floor(((det.bbox[0] + det.bbox[2]) / 2 / VIDEO_WIDTH) * 9));
        const y = Math.min(5, Math.floor(((det.bbox[1] + det.bbox[3]) / 2 / VIDEO_HEIGHT) * 6));
        if (x >= 0 && y >= 0) g[y][x]++;
      });
    });
    return g;
  }, [analysis, team]);

  const maxVal = Math.max(...grid.flat(), 1);
  const color = team === 'home' ? 'rgb(239, 68, 68)' : 'rgb(148, 163, 184)';

  return (
    <div className="relative aspect-[3/2] bg-emerald-900/20 rounded-lg overflow-hidden">
      <PitchSVG />
      <div className="absolute inset-0 grid grid-cols-9 grid-rows-6 gap-0.5 p-1">
        {grid.map((row, y) => row.map((val, x) => (
          <div key={`${y}-${x}`} className="rounded-sm" style={{ backgroundColor: color, opacity: (val / maxVal) * 0.8 }} />
        )))}
      </div>
    </div>
  );
}

function FullPitchHeatmap({ analysis, team, title, large }: { analysis: MatchAnalysis; team: 'home' | 'away'; title: string; large?: boolean }) {
  const grid = useMemo(() => {
    const cols = large ? 15 : 12;
    const rows = large ? 10 : 8;
    const g: number[][] = Array(rows).fill(null).map(() => Array(cols).fill(0));
    analysis.frames.forEach(frame => {
      frame.detections.filter(d => d.team === team).forEach(det => {
        const x = Math.min(cols - 1, Math.floor(((det.bbox[0] + det.bbox[2]) / 2 / VIDEO_WIDTH) * cols));
        const y = Math.min(rows - 1, Math.floor(((det.bbox[1] + det.bbox[3]) / 2 / VIDEO_HEIGHT) * rows));
        if (x >= 0 && y >= 0) g[y][x]++;
      });
    });
    return { grid: g, cols, rows };
  }, [analysis, team, large]);

  const maxVal = Math.max(...grid.grid.flat(), 1);
  const color = team === 'home' ? 'rgb(239, 68, 68)' : 'rgb(148, 163, 184)';

  return (
    <div className="bg-[#111827] rounded-2xl p-5 border border-slate-700/30">
      <h3 className="text-white font-semibold mb-4 text-sm">{title}</h3>
      <div className={`relative ${large ? 'aspect-[3/2]' : 'aspect-[3/2]'} bg-emerald-900/20 rounded-lg overflow-hidden`}>
        <PitchSVG />
        <div
          className="absolute inset-0 gap-0.5 p-1"
          style={{ display: 'grid', gridTemplateColumns: `repeat(${grid.cols}, 1fr)`, gridTemplateRows: `repeat(${grid.rows}, 1fr)` }}
        >
          {grid.grid.map((row, y) => row.map((val, x) => (
            <div key={`${y}-${x}`} className="rounded-sm" style={{ backgroundColor: color, opacity: (val / maxVal) * 0.8 }} />
          )))}
        </div>
      </div>
      <div className="flex items-center justify-between mt-3 text-xs text-slate-400">
        <span>Low activity</span>
        <div className="flex-1 mx-3 h-2 rounded-full" style={{ background: `linear-gradient(to right, transparent, ${color})` }} />
        <span>High activity</span>
      </div>
    </div>
  );
}

function ZoneBreakdown({ analysis }: { analysis: MatchAnalysis }) {
  const zones = useMemo(() => {
    const z = { home: { def: 0, mid: 0, att: 0 }, away: { def: 0, mid: 0, att: 0 } };
    analysis.frames.forEach(frame => {
      frame.detections.forEach(det => {
        const x = (det.bbox[0] + det.bbox[2]) / 2 / VIDEO_WIDTH;
        if (det.team === 'home') {
          if (x < 0.33) z.home.def++; else if (x < 0.67) z.home.mid++; else z.home.att++;
        } else if (det.team === 'away') {
          if (x > 0.67) z.away.def++; else if (x > 0.33) z.away.mid++; else z.away.att++;
        }
      });
    });
    const homeTotal = z.home.def + z.home.mid + z.home.att || 1;
    const awayTotal = z.away.def + z.away.mid + z.away.att || 1;
    return {
      home: { def: Math.round(z.home.def / homeTotal * 100), mid: Math.round(z.home.mid / homeTotal * 100), att: Math.round(z.home.att / homeTotal * 100) },
      away: { def: Math.round(z.away.def / awayTotal * 100), mid: Math.round(z.away.mid / awayTotal * 100), att: Math.round(z.away.att / awayTotal * 100) }
    };
  }, [analysis]);

  return (
    <div className="relative aspect-[3/2] bg-emerald-900/20 rounded-lg overflow-hidden">
      <PitchSVG />
      <div className="absolute inset-0 grid grid-cols-3">
        {['Defensive', 'Middle', 'Attacking'].map((zone, i) => (
          <div key={zone} className="flex flex-col items-center justify-center border-r border-emerald-700/30 last:border-r-0">
            <div className="text-white text-xs mb-2 opacity-70">{zone}</div>
            <div className="text-red-400 font-bold">{i === 0 ? zones.home.def : i === 1 ? zones.home.mid : zones.home.att}%</div>
            <div className="text-xs text-slate-500 my-1">vs</div>
            <div className="text-slate-400 font-bold">{i === 0 ? zones.away.att : i === 1 ? zones.away.mid : zones.away.def}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PossessionDonut({ home, away }: { home: number; away: number }) {
  const radius = 60;
  const stroke = 12;
  const circumference = 2 * Math.PI * radius;
  const homeLength = (home / 100) * circumference;

  return (
    <svg width="160" height="160" viewBox="0 0 160 160">
      <circle cx="80" cy="80" r={radius} fill="none" stroke="#475569" strokeWidth={stroke} />
      <circle
        cx="80" cy="80" r={radius} fill="none"
        stroke="#ef4444" strokeWidth={stroke}
        strokeDasharray={`${homeLength} ${circumference}`}
        strokeLinecap="round"
        transform="rotate(-90 80 80)"
      />
    </svg>
  );
}

function PossessionLocationMap({ analysis }: { analysis: MatchAnalysis }) {
  const grid = useMemo(() => {
    const g: { home: number; away: number }[][] = Array(6).fill(null).map(() => Array(9).fill(null).map(() => ({ home: 0, away: 0 })));
    analysis.frames.forEach(frame => {
      frame.detections.forEach(det => {
        if (det.team !== 'home' && det.team !== 'away') return;
        const x = Math.min(8, Math.floor(((det.bbox[0] + det.bbox[2]) / 2 / VIDEO_WIDTH) * 9));
        const y = Math.min(5, Math.floor(((det.bbox[1] + det.bbox[3]) / 2 / VIDEO_HEIGHT) * 6));
        if (x >= 0 && y >= 0) g[y][x][det.team]++;
      });
    });
    return g;
  }, [analysis]);

  return (
    <div className="relative aspect-[3/2] bg-emerald-900/20 rounded-lg overflow-hidden">
      <PitchSVG />
      <div className="absolute inset-0 grid grid-cols-9 grid-rows-6 gap-0.5 p-1">
        {grid.map((row, y) => row.map((cell, x) => {
          const total = cell.home + cell.away || 1;
          const homeRatio = cell.home / total;
          return (
            <div
              key={`${y}-${x}`}
              className="rounded-sm"
              style={{
                background: `linear-gradient(to right, rgb(239, 68, 68) ${homeRatio * 100}%, rgb(148, 163, 184) ${homeRatio * 100}%)`,
                opacity: Math.min((cell.home + cell.away) / 500, 0.8)
              }}
            />
          );
        }))}
      </div>
    </div>
  );
}

function ComparisonRadar({ stats }: { stats: MatchStats }) {
  const metrics = [
    { label: 'Possession', home: stats.possession.home, away: stats.possession.away },
    { label: 'Territory', home: stats.territorialAdvantage.home, away: stats.territorialAdvantage.away },
    { label: 'Pressing', home: Math.min(100, stats.pressingActions.home * 2), away: Math.min(100, stats.pressingActions.away * 2) },
    { label: 'Compactness', home: stats.compactness.home, away: stats.compactness.away },
    { label: 'Width', home: stats.widthUsage.home, away: stats.widthUsage.away },
  ];

  const centerX = 150, centerY = 150, maxRadius = 100;
  const angleStep = (2 * Math.PI) / metrics.length;

  const getPoint = (value: number, index: number) => {
    const angle = index * angleStep - Math.PI / 2;
    const radius = (value / 100) * maxRadius;
    return { x: centerX + radius * Math.cos(angle), y: centerY + radius * Math.sin(angle) };
  };

  const homePoints = metrics.map((m, i) => getPoint(m.home, i));
  const awayPoints = metrics.map((m, i) => getPoint(m.away, i));

  return (
    <div className="flex items-center justify-center">
      <svg width="300" height="300" viewBox="0 0 300 300">
        {/* Grid lines */}
        {[20, 40, 60, 80, 100].map(pct => (
          <polygon
            key={pct}
            points={metrics.map((_, i) => {
              const p = getPoint(pct, i);
              return `${p.x},${p.y}`;
            }).join(' ')}
            fill="none" stroke="#334155" strokeWidth="1"
          />
        ))}

        {/* Axis lines */}
        {metrics.map((m, i) => {
          const p = getPoint(100, i);
          return <line key={i} x1={centerX} y1={centerY} x2={p.x} y2={p.y} stroke="#334155" strokeWidth="1" />;
        })}

        {/* Away polygon */}
        <polygon
          points={awayPoints.map(p => `${p.x},${p.y}`).join(' ')}
          fill="rgba(148, 163, 184, 0.2)" stroke="#94a3b8" strokeWidth="2"
        />

        {/* Home polygon */}
        <polygon
          points={homePoints.map(p => `${p.x},${p.y}`).join(' ')}
          fill="rgba(239, 68, 68, 0.2)" stroke="#ef4444" strokeWidth="2"
        />

        {/* Labels */}
        {metrics.map((m, i) => {
          const p = getPoint(115, i);
          return (
            <text key={i} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="middle" className="fill-slate-400 text-xs">
              {m.label}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

function PitchSVG() {
  return (
    <svg viewBox="0 0 120 80" className="absolute inset-0 w-full h-full">
      <rect x="2" y="2" width="116" height="76" fill="none" stroke="#065f46" strokeWidth="0.5" />
      <line x1="60" y1="2" x2="60" y2="78" stroke="#065f46" strokeWidth="0.5" />
      <circle cx="60" cy="40" r="9.15" fill="none" stroke="#065f46" strokeWidth="0.5" />
      <rect x="2" y="22" width="16.5" height="36" fill="none" stroke="#065f46" strokeWidth="0.5" />
      <rect x="101.5" y="22" width="16.5" height="36" fill="none" stroke="#065f46" strokeWidth="0.5" />
      <rect x="2" y="30" width="5.5" height="20" fill="none" stroke="#065f46" strokeWidth="0.5" />
      <rect x="112.5" y="30" width="5.5" height="20" fill="none" stroke="#065f46" strokeWidth="0.5" />
    </svg>
  );
}

function PitchSVGDetailed() {
  return (
    <svg viewBox="0 0 120 80" className="absolute inset-0 w-full h-full" preserveAspectRatio="xMidYMid slice">
      {/* Pitch background */}
      <rect x="0" y="0" width="120" height="80" fill="#1a4d2e" />

      {/* Grass stripes */}
      {[0, 15, 30, 45, 60, 75, 90, 105].map(x => (
        <rect key={x} x={x} y="0" width="15" height="80" fill={x % 30 === 0 ? '#1e5a35' : '#1a4d2e'} />
      ))}

      {/* Outer boundary */}
      <rect x="2" y="2" width="116" height="76" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />

      {/* Center line */}
      <line x1="60" y1="2" x2="60" y2="78" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />

      {/* Center circle */}
      <circle cx="60" cy="40" r="9.15" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />
      <circle cx="60" cy="40" r="0.5" fill="rgba(255,255,255,0.6)" />

      {/* Left penalty area */}
      <rect x="2" y="22" width="16.5" height="36" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />
      <rect x="2" y="30" width="5.5" height="20" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />
      <circle cx="11" cy="40" r="0.5" fill="rgba(255,255,255,0.6)" />
      <path d="M 18.5 33 A 9.15 9.15 0 0 1 18.5 47" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />

      {/* Right penalty area */}
      <rect x="101.5" y="22" width="16.5" height="36" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />
      <rect x="112.5" y="30" width="5.5" height="20" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />
      <circle cx="109" cy="40" r="0.5" fill="rgba(255,255,255,0.6)" />
      <path d="M 101.5 33 A 9.15 9.15 0 0 0 101.5 47" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />

      {/* Goals */}
      <rect x="0" y="36" width="2" height="8" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="0.5" />
      <rect x="118" y="36" width="2" height="8" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="0.5" />

      {/* Corner arcs */}
      <path d="M 2 4 A 2 2 0 0 0 4 2" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.3" />
      <path d="M 116 2 A 2 2 0 0 0 118 4" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.3" />
      <path d="M 2 76 A 2 2 0 0 1 4 78" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.3" />
      <path d="M 116 78 A 2 2 0 0 1 118 76" fill="none" stroke="rgba(255,255,255,0.6)" strokeWidth="0.3" />
    </svg>
  );
}

// ==================== STATS CALCULATION ====================
interface MatchStats {
  avgHomePlayers: number;
  avgAwayPlayers: number;
  homeScore: number;
  awayScore: number;
  possession: { home: number; away: number };
  territorialAdvantage: { home: number; away: number };
  pressingActions: { home: number; away: number };
  highIntensityMoments: { home: number; away: number };
  compactness: { home: number; away: number };
  widthUsage: { home: number; away: number };
  possessionByZone: { home: number; away: number }[];
  momentumPeriods: { label: string; homeControl: number; awayControl: number; dominant: 'home' | 'away' }[];
  momentumShifts: { time: string; description: string; favor: 'home' | 'away' }[];
  periodStats: { label: string; homePossession: number; awayPossession: number; pressing: number; dominant: 'home' | 'away' }[];
  insights: { type: 'positive' | 'warning' | 'info'; title: string; description: string }[];
}

function calculateMatchStats(analysis: MatchAnalysis, period: 'full' | '1st' | '2nd'): MatchStats {
  const frames = period === 'full' ? analysis.frames :
    period === '1st' ? analysis.frames.slice(0, Math.floor(analysis.frames.length / 2)) :
    analysis.frames.slice(Math.floor(analysis.frames.length / 2));

  const avgHomePlayers = frames.reduce((sum, f) => sum + f.home_players, 0) / frames.length;
  const avgAwayPlayers = frames.reduce((sum, f) => sum + f.away_players, 0) / frames.length;

  // Possession based on territorial presence
  let homeTerritorySum = 0, awayTerritorySum = 0;
  let homeDefZone = 0, homeMidZone = 0, homeAttZone = 0;
  let awayDefZone = 0, awayMidZone = 0, awayAttZone = 0;
  let homeCompactFrames = 0, awayCompactFrames = 0;
  let homeWideFrames = 0, awayWideFrames = 0;
  let homePressing = 0, awayPressing = 0;
  let homeHighIntensity = 0, awayHighIntensity = 0;

  frames.forEach((frame, idx) => {
    const homeDetections = frame.detections.filter(d => d.team === 'home');
    const awayDetections = frame.detections.filter(d => d.team === 'away');

    // Territorial calculation
    homeDetections.forEach(d => {
      const x = (d.bbox[0] + d.bbox[2]) / 2 / VIDEO_WIDTH;
      homeTerritorySum += x;
      if (x < 0.33) homeDefZone++; else if (x < 0.67) homeMidZone++; else homeAttZone++;
    });
    awayDetections.forEach(d => {
      const x = (d.bbox[0] + d.bbox[2]) / 2 / VIDEO_WIDTH;
      awayTerritorySum += 1 - x;
      if (x > 0.67) awayDefZone++; else if (x > 0.33) awayMidZone++; else awayAttZone++;
    });

    // Compactness (how close players are)
    if (homeDetections.length >= 3) {
      const xs = homeDetections.map(d => (d.bbox[0] + d.bbox[2]) / 2);
      const spread = Math.max(...xs) - Math.min(...xs);
      if (spread < VIDEO_WIDTH * 0.4) homeCompactFrames++;
      if (spread > VIDEO_WIDTH * 0.5) homeWideFrames++;
    }
    if (awayDetections.length >= 3) {
      const xs = awayDetections.map(d => (d.bbox[0] + d.bbox[2]) / 2);
      const spread = Math.max(...xs) - Math.min(...xs);
      if (spread < VIDEO_WIDTH * 0.4) awayCompactFrames++;
      if (spread > VIDEO_WIDTH * 0.5) awayWideFrames++;
    }

    // Pressing (players in opposition half)
    const homeInOppHalf = homeDetections.filter(d => (d.bbox[0] + d.bbox[2]) / 2 > VIDEO_WIDTH * 0.5).length;
    const awayInOppHalf = awayDetections.filter(d => (d.bbox[0] + d.bbox[2]) / 2 < VIDEO_WIDTH * 0.5).length;
    if (homeInOppHalf >= 4) homePressing++;
    if (awayInOppHalf >= 4) awayPressing++;

    // High intensity (team centroid movement between frames)
    if (idx > 0) {
      const prevFrame = frames[idx - 1];
      const prevHomeX = prevFrame.detections.filter(d => d.team === 'home').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (prevFrame.home_players || 1);
      const currHomeX = homeDetections.reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (homeDetections.length || 1);
      if (Math.abs(currHomeX - prevHomeX) > 100) homeHighIntensity++;

      const prevAwayX = prevFrame.detections.filter(d => d.team === 'away').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (prevFrame.away_players || 1);
      const currAwayX = awayDetections.reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (awayDetections.length || 1);
      if (Math.abs(currAwayX - prevAwayX) > 100) awayHighIntensity++;
    }
  });

  const totalHomeDetections = homeDefZone + homeMidZone + homeAttZone || 1;
  const totalAwayDetections = awayDefZone + awayMidZone + awayAttZone || 1;

  const homePossession = Math.round((homeTerritorySum / (homeTerritorySum + awayTerritorySum + 0.1)) * 100);
  const awayPossession = 100 - homePossession;

  const homeTerritory = Math.round((homeAttZone / totalHomeDetections) * 100);
  const awayTerritory = Math.round((awayAttZone / totalAwayDetections) * 100);

  // Momentum periods
  const periodSize = Math.floor(frames.length / 3);
  const momentumPeriods = ['0-10 min', '10-20 min', '20-26 min'].map((label, i) => {
    const start = i * periodSize;
    const end = Math.min((i + 1) * periodSize, frames.length);
    let homeControl = 0;
    for (let j = start; j < end; j++) {
      const homeX = frames[j].detections.filter(d => d.team === 'home').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (frames[j].home_players || 1);
      if (homeX > VIDEO_WIDTH * 0.5) homeControl++;
    }
    const homeCtrl = Math.round((homeControl / (end - start)) * 100);
    return { label, homeControl: homeCtrl, awayControl: 100 - homeCtrl, dominant: homeCtrl >= 50 ? 'home' as const : 'away' as const };
  });

  // Momentum shifts
  const momentumShifts = [];
  let lastDominant: 'home' | 'away' | null = null;
  for (let i = 0; i < frames.length; i += Math.floor(frames.length / 10)) {
    const frame = frames[i];
    const homeX = frame.detections.filter(d => d.team === 'home').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (frame.home_players || 1);
    const dominant = homeX > VIDEO_WIDTH * 0.55 ? 'home' : homeX < VIDEO_WIDTH * 0.45 ? 'away' : lastDominant;
    if (dominant && dominant !== lastDominant && lastDominant !== null) {
      momentumShifts.push({
        time: formatTime(frame.timestamp),
        description: dominant === 'home' ? 'Your team gains territorial control' : 'Opposition pushes forward',
        favor: dominant
      });
    }
    if (dominant) lastDominant = dominant;
  }

  // Period stats
  const periodStats = ['0-9 min', '9-18 min', '18-26 min'].map((label, i) => {
    const start = i * periodSize;
    const end = Math.min((i + 1) * periodSize, frames.length);
    let hPoss = 0, pressing = 0;
    for (let j = start; j < end; j++) {
      const homeX = frames[j].detections.filter(d => d.team === 'home').reduce((s, d) => s + (d.bbox[0] + d.bbox[2]) / 2, 0) / (frames[j].home_players || 1);
      if (homeX > VIDEO_WIDTH * 0.5) hPoss++;
      if (frames[j].detections.filter(d => d.team === 'home' && (d.bbox[0] + d.bbox[2]) / 2 > VIDEO_WIDTH * 0.5).length >= 4) pressing++;
    }
    const homePoss = Math.round((hPoss / (end - start)) * 100);
    return { label, homePossession: homePoss, awayPossession: 100 - homePoss, pressing, dominant: homePoss >= 50 ? 'home' as const : 'away' as const };
  });

  // Insights
  const insights: MatchStats['insights'] = [];
  if (homePressing > frames.length * 0.3) {
    insights.push({ type: 'positive', title: 'Strong Pressing', description: 'Your team shows excellent pressing intensity with 4+ players regularly in the opposition half.' });
  } else {
    insights.push({ type: 'warning', title: 'Pressing Opportunities', description: 'Consider pressing higher when opposition receives the ball. Look for triggers like poor first touches.' });
  }
  if (homeCompactFrames > frames.length * 0.5) {
    insights.push({ type: 'positive', title: 'Solid Defensive Shape', description: 'Team maintains good compactness throughout, making it hard for opposition to find space.' });
  } else {
    insights.push({ type: 'warning', title: 'Shape Work Needed', description: 'Players are spreading too wide at times. Work on staying compact and reducing gaps between lines.' });
  }
  if (homeTerritory > 40) {
    insights.push({ type: 'positive', title: 'Attacking Presence', description: `${homeTerritory}% of your team's positions were in the attacking third. Good forward commitment.` });
  } else {
    insights.push({ type: 'info', title: 'Push Forward', description: 'Consider committing more players forward when in possession to create overloads.' });
  }

  return {
    avgHomePlayers,
    avgAwayPlayers,
    homeScore: 0,
    awayScore: 0,
    possession: { home: homePossession, away: awayPossession },
    territorialAdvantage: { home: homeTerritory, away: awayTerritory },
    pressingActions: { home: homePressing, away: awayPressing },
    highIntensityMoments: { home: homeHighIntensity, away: awayHighIntensity },
    compactness: { home: Math.round((homeCompactFrames / frames.length) * 100), away: Math.round((awayCompactFrames / frames.length) * 100) },
    widthUsage: { home: Math.round((homeWideFrames / frames.length) * 100), away: Math.round((awayWideFrames / frames.length) * 100) },
    possessionByZone: [
      { home: Math.round((homeDefZone / totalHomeDetections) * 100), away: Math.round((awayAttZone / totalAwayDetections) * 100) },
      { home: Math.round((homeMidZone / totalHomeDetections) * 100), away: Math.round((awayMidZone / totalAwayDetections) * 100) },
      { home: Math.round((homeAttZone / totalHomeDetections) * 100), away: Math.round((awayDefZone / totalAwayDetections) * 100) },
    ],
    momentumPeriods,
    momentumShifts: momentumShifts.slice(0, 5),
    periodStats,
    insights,
  };
}

// ==================== AI COACH VIEW ====================

interface CoachingInsight {
  category: string;
  priority: string;
  title: string;
  message: string;
  recommendation: string;
  supporting_data: Record<string, any>;
}

interface MatchSummary {
  overall_rating: string;
  key_strengths: string[];
  areas_to_improve: string[];
  tactical_summary: string;
  half_time_message: string;
  full_time_message: string;
}

interface AICoachData {
  summary: MatchSummary | null;
  insights: CoachingInsight[];
  critical_insights: CoachingInsight[];
  total_insights: number;
}

// Chat message type
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  confidence?: string;
  relatedInsights?: string[];
}

function AICoachView() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [coachData, setCoachData] = useState<AICoachData | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [showTeamTalk, setShowTeamTalk] = useState<'half_time' | 'full_time' | null>(null);

  // Chat state
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hi! I'm your AI coaching assistant. Ask me anything about the match - possession, formations, player performance, what to work on, or any tactical questions!",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadCoachingData();
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const sendChatMessage = async () => {
    if (!chatInput.trim() || chatLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: chatInput.trim(),
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/ai-coach/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content })
      });

      if (!response.ok) throw new Error('Chat request failed');
      const data = await response.json();

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        confidence: data.confidence,
        relatedInsights: data.related_insights
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "Sorry, I couldn't process your question. Please try again.",
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    }

    setChatLoading(false);
  };

  const suggestedQuestions = [
    "How was our possession?",
    "Who was our best player?",
    "What should we work on?",
    "How did our pressing look?",
    "Summarize the match"
  ];

  const loadCoachingData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/ai-coach/analysis');
      if (!response.ok) throw new Error('Failed to load coaching analysis');
      const data = await response.json();
      setCoachData(data.coaching);
      setLoading(false);
    } catch (err) {
      console.error('Failed to load coaching data:', err);
      setError('Could not load AI coaching analysis');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="w-12 h-12 border-3 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-400 text-sm">AI Coach is analyzing the match...</p>
        </div>
      </div>
    );
  }

  if (error || !coachData) {
    return (
      <div className="bg-[#111827] rounded-2xl p-8 text-center border border-slate-700/50">
        <div className="w-14 h-14 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-3xl">🧠</span>
        </div>
        <h3 className="text-white font-semibold mb-2">AI Coach Unavailable</h3>
        <p className="text-slate-400 text-sm mb-4">{error || 'Could not load coaching analysis'}</p>
        <button onClick={loadCoachingData} className="px-4 py-2 bg-cyan-500 text-white text-sm rounded-lg hover:bg-cyan-600">
          Retry
        </button>
      </div>
    );
  }

  const categories = ['all', 'tactical', 'pressing', 'possession', 'defensive', 'attacking', 'formation'];
  const filteredInsights = selectedCategory === 'all'
    ? coachData.insights
    : coachData.insights.filter(i => i.category === selectedCategory);

  const priorityColors: Record<string, string> = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    low: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    info: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
  };

  const ratingColors: Record<string, string> = {
    'Excellent': 'text-green-400',
    'Good': 'text-cyan-400',
    'Average': 'text-yellow-400',
    'Poor': 'text-red-400',
  };

  return (
    <div className="space-y-6">
      {/* Header Section with Match Summary */}
      {coachData.summary && (
        <div className="bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-blue-500/10 rounded-2xl p-6 border border-cyan-500/20">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-3xl">🧠</span>
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">AI Coaching Expert</h2>
                <p className="text-slate-400 text-sm">Tactical analysis and recommendations</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-slate-400 text-xs block mb-1">Overall Rating</span>
              <span className={`text-2xl font-bold ${ratingColors[coachData.summary.overall_rating] || 'text-white'}`}>
                {coachData.summary.overall_rating}
              </span>
            </div>
          </div>

          <p className="text-slate-300 mb-4">{coachData.summary.tactical_summary}</p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-500/10 rounded-xl p-4 border border-green-500/20">
              <h4 className="text-green-400 font-semibold text-sm mb-2 flex items-center gap-2">
                <span>✅</span> Key Strengths
              </h4>
              <ul className="space-y-1">
                {coachData.summary.key_strengths.map((strength, i) => (
                  <li key={i} className="text-slate-300 text-sm flex items-start gap-2">
                    <span className="text-green-500">•</span> {strength}
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-orange-500/10 rounded-xl p-4 border border-orange-500/20">
              <h4 className="text-orange-400 font-semibold text-sm mb-2 flex items-center gap-2">
                <span>⚠️</span> Areas to Improve
              </h4>
              <ul className="space-y-1">
                {coachData.summary.areas_to_improve.map((area, i) => (
                  <li key={i} className="text-slate-300 text-sm flex items-start gap-2">
                    <span className="text-orange-500">•</span> {area}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Team Talk Buttons */}
          <div className="flex gap-3 mt-4">
            <button
              onClick={() => setShowTeamTalk('half_time')}
              className="flex-1 px-4 py-3 bg-cyan-500/20 text-cyan-400 rounded-xl hover:bg-cyan-500/30 transition-colors border border-cyan-500/30 flex items-center justify-center gap-2"
            >
              <span>📣</span> Half-Time Talk
            </button>
            <button
              onClick={() => setShowTeamTalk('full_time')}
              className="flex-1 px-4 py-3 bg-purple-500/20 text-purple-400 rounded-xl hover:bg-purple-500/30 transition-colors border border-purple-500/30 flex items-center justify-center gap-2"
            >
              <span>📋</span> Full-Time Review
            </button>
          </div>
        </div>
      )}

      {/* Team Talk Modal */}
      {showTeamTalk && coachData.summary && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={() => setShowTeamTalk(null)}>
          <div className="bg-[#111827] rounded-2xl p-6 max-w-2xl w-full border border-cyan-500/30 shadow-xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                {showTeamTalk === 'half_time' ? '📣 Half-Time Team Talk' : '📋 Full-Time Review'}
              </h3>
              <button onClick={() => setShowTeamTalk(null)} className="text-slate-400 hover:text-white">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="bg-slate-800/50 rounded-xl p-5 mb-4">
              <p className="text-white text-lg leading-relaxed">
                {showTeamTalk === 'half_time' ? coachData.summary.half_time_message : coachData.summary.full_time_message}
              </p>
            </div>

            <div className="space-y-3">
              <h4 className="text-slate-400 font-semibold text-sm">Key Points to Address:</h4>
              {coachData.critical_insights.slice(0, 3).map((insight, i) => (
                <div key={i} className={`rounded-lg p-3 border ${priorityColors[insight.priority]}`}>
                  <div className="font-semibold mb-1">{insight.title}</div>
                  <div className="text-sm opacity-80">{insight.recommendation}</div>
                </div>
              ))}
            </div>

            <button
              onClick={() => setShowTeamTalk(null)}
              className="w-full mt-4 px-4 py-3 bg-cyan-500 text-white rounded-xl hover:bg-cyan-600 transition-colors font-semibold"
            >
              Got It - Let's Go!
            </button>
          </div>
        </div>
      )}

      {/* Critical Insights Banner */}
      {coachData.critical_insights.length > 0 && (
        <div className="bg-red-500/10 rounded-2xl p-4 border border-red-500/30">
          <h3 className="text-red-400 font-semibold mb-3 flex items-center gap-2">
            <span>🚨</span> Urgent Attention Required ({coachData.critical_insights.length})
          </h3>
          <div className="grid gap-3">
            {coachData.critical_insights.slice(0, 3).map((insight, i) => (
              <div key={i} className="bg-red-500/10 rounded-xl p-4 border border-red-500/20">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-red-400 text-sm font-bold">{i + 1}</span>
                  </div>
                  <div>
                    <h4 className="text-white font-semibold">{insight.title}</h4>
                    <p className="text-slate-300 text-sm mt-1">{insight.message}</p>
                    <div className="mt-2 p-2 bg-slate-800/50 rounded-lg">
                      <span className="text-cyan-400 text-sm font-medium">💡 Recommendation: </span>
                      <span className="text-slate-300 text-sm">{insight.recommendation}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Category Filter */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-slate-400 text-sm mr-2">Filter by:</span>
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-lg text-sm capitalize transition-all ${
              selectedCategory === cat
                ? 'bg-cyan-500 text-white'
                : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            {cat}
          </button>
        ))}
        <span className="text-slate-500 text-xs ml-2">({filteredInsights.length} insights)</span>
      </div>

      {/* All Insights */}
      <div className="grid gap-4">
        {filteredInsights.map((insight, i) => (
          <InsightCard key={i} insight={insight} priorityColors={priorityColors} />
        ))}
      </div>

      {/* No Insights Message */}
      {filteredInsights.length === 0 && (
        <div className="bg-[#111827] rounded-2xl p-8 text-center border border-slate-700/30">
          <span className="text-4xl mb-4 block">📭</span>
          <h3 className="text-white font-semibold mb-2">No Insights Found</h3>
          <p className="text-slate-400 text-sm">Try selecting a different category filter</p>
        </div>
      )}

      {/* Stats Summary */}
      <div className="bg-[#111827] rounded-2xl p-6 border border-slate-700/30">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <span className="text-cyan-400">📊</span> Analysis Summary
        </h3>
        <div className="grid grid-cols-5 gap-4">
          <div className="text-center p-3 bg-slate-800/50 rounded-xl">
            <div className="text-2xl font-bold text-white">{coachData.total_insights}</div>
            <div className="text-slate-400 text-xs">Total Insights</div>
          </div>
          <div className="text-center p-3 bg-red-500/10 rounded-xl border border-red-500/20">
            <div className="text-2xl font-bold text-red-400">
              {coachData.insights.filter(i => i.priority === 'critical').length}
            </div>
            <div className="text-slate-400 text-xs">Critical</div>
          </div>
          <div className="text-center p-3 bg-orange-500/10 rounded-xl border border-orange-500/20">
            <div className="text-2xl font-bold text-orange-400">
              {coachData.insights.filter(i => i.priority === 'high').length}
            </div>
            <div className="text-slate-400 text-xs">High Priority</div>
          </div>
          <div className="text-center p-3 bg-yellow-500/10 rounded-xl border border-yellow-500/20">
            <div className="text-2xl font-bold text-yellow-400">
              {coachData.insights.filter(i => i.priority === 'medium').length}
            </div>
            <div className="text-slate-400 text-xs">Medium</div>
          </div>
          <div className="text-center p-3 bg-slate-500/10 rounded-xl border border-slate-500/20">
            <div className="text-2xl font-bold text-slate-400">
              {coachData.insights.filter(i => i.priority === 'low' || i.priority === 'info').length}
            </div>
            <div className="text-slate-400 text-xs">Informational</div>
          </div>
        </div>
      </div>

      {/* Floating Chat Button */}
      <button
        onClick={() => setShowChat(true)}
        className="fixed bottom-6 right-6 w-16 h-16 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-full shadow-lg hover:shadow-cyan-500/30 transition-all duration-300 hover:scale-110 flex items-center justify-center z-40"
      >
        <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      </button>

      {/* Chat Panel */}
      {showChat && (
        <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-[#0d1117] rounded-2xl shadow-2xl border border-cyan-500/30 flex flex-col z-50 overflow-hidden">
          {/* Chat Header */}
          <div className="bg-gradient-to-r from-cyan-500/20 to-purple-600/20 p-4 border-b border-slate-700/50 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-xl">🧠</span>
              </div>
              <div>
                <h3 className="text-white font-semibold">AI Coach Chat</h3>
                <p className="text-slate-400 text-xs">Ask about the match</p>
              </div>
            </div>
            <button onClick={() => setShowChat(false)} className="text-slate-400 hover:text-white p-1">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {chatMessages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-cyan-500 text-white rounded-br-md'
                    : 'bg-slate-800 text-slate-200 rounded-bl-md border border-slate-700/50'
                }`}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  {msg.confidence && (
                    <span className={`text-xs mt-2 block ${
                      msg.confidence === 'high' ? 'text-green-400' :
                      msg.confidence === 'medium' ? 'text-yellow-400' : 'text-slate-400'
                    }`}>
                      Confidence: {msg.confidence}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-800 text-slate-400 rounded-2xl rounded-bl-md px-4 py-3 border border-slate-700/50">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Suggested Questions */}
          {chatMessages.length === 1 && (
            <div className="px-4 pb-2">
              <p className="text-slate-500 text-xs mb-2">Suggested questions:</p>
              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setChatInput(q);
                    }}
                    className="text-xs px-3 py-1.5 bg-slate-800 text-cyan-400 rounded-full hover:bg-slate-700 transition-colors border border-slate-700/50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chat Input */}
          <div className="p-4 border-t border-slate-700/50 bg-slate-900/50">
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Ask about the match..."
                className="flex-1 bg-slate-800 text-white placeholder-slate-500 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50 border border-slate-700/50"
              />
              <button
                onClick={sendChatMessage}
                disabled={!chatInput.trim() || chatLoading}
                className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-cyan-500/20 transition-all"
              >
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ==================== EXPORT BUTTON ====================

function ExportButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);

  const generateReport = async () => {
    setGenerating(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/report/generate');
      if (response.ok) {
        const data = await response.json();
        setReportUrl(`http://127.0.0.1:8000${data.download_url}`);
      }
    } catch (err) {
      console.error('Failed to generate report:', err);
    }
    setGenerating(false);
  };

  const downloadJson = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/report/json');
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data.report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'match_report.json';
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Failed to download JSON:', err);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="px-4 py-2 bg-slate-700/50 text-slate-300 text-sm rounded-lg hover:bg-slate-700 transition-colors flex items-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        Export
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute right-0 top-full mt-2 w-72 bg-[#111827] rounded-xl shadow-xl border border-slate-700/50 z-50 overflow-hidden">
            <div className="p-4 border-b border-slate-700/50">
              <h3 className="text-white font-semibold">Export Match Report</h3>
              <p className="text-slate-400 text-xs mt-1">Download your analysis in different formats</p>
            </div>

            <div className="p-2">
              {/* HTML/PDF Report */}
              <button
                onClick={generateReport}
                disabled={generating}
                className="w-full p-3 text-left rounded-lg hover:bg-slate-800/50 transition-colors flex items-center gap-3"
              >
                <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center">
                  <span className="text-lg">📄</span>
                </div>
                <div>
                  <div className="text-white text-sm font-medium">
                    {generating ? 'Generating...' : 'HTML Report (Save as PDF)'}
                  </div>
                  <div className="text-slate-500 text-xs">Full analysis with coaching insights</div>
                </div>
              </button>

              {reportUrl && (
                <a
                  href={reportUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full p-3 text-left rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 transition-colors flex items-center gap-3 mt-2"
                >
                  <div className="w-10 h-10 bg-cyan-500/30 rounded-lg flex items-center justify-center">
                    <span className="text-lg">📥</span>
                  </div>
                  <div>
                    <div className="text-cyan-400 text-sm font-medium">Download Report</div>
                    <div className="text-slate-400 text-xs">Open in browser, then Print to Save as PDF</div>
                  </div>
                </a>
              )}

              {/* JSON Export */}
              <button
                onClick={downloadJson}
                className="w-full p-3 text-left rounded-lg hover:bg-slate-800/50 transition-colors flex items-center gap-3"
              >
                <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <span className="text-lg">{ }</span>
                </div>
                <div>
                  <div className="text-white text-sm font-medium">JSON Data Export</div>
                  <div className="text-slate-500 text-xs">Raw data for custom processing</div>
                </div>
              </button>
            </div>

            <div className="p-3 bg-slate-800/30 border-t border-slate-700/50">
              <p className="text-slate-500 text-xs text-center">
                Tip: Open the HTML report and use your browser's Print function to save as PDF
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function InsightCard({ insight, priorityColors }: { insight: CoachingInsight; priorityColors: Record<string, string> }) {
  const [expanded, setExpanded] = useState(false);

  const categoryIcons: Record<string, string> = {
    tactical: '♟️',
    pressing: '💨',
    possession: '🎮',
    defensive: '🛡️',
    attacking: '⚔️',
    formation: '📋',
    set_pieces: '🎯',
    player_specific: '👤',
    substitution: '🔄',
    physical: '🏃',
  };

  return (
    <div className={`bg-[#111827] rounded-xl border ${priorityColors[insight.priority]} overflow-hidden`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 text-left flex items-start gap-4"
      >
        <div className="w-10 h-10 bg-slate-800/50 rounded-lg flex items-center justify-center flex-shrink-0">
          <span className="text-xl">{categoryIcons[insight.category] || '💡'}</span>
        </div>
        <div className="flex-grow">
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${priorityColors[insight.priority]}`}>
              {insight.priority.toUpperCase()}
            </span>
            <span className="text-slate-500 text-xs capitalize">{insight.category}</span>
          </div>
          <h4 className="text-white font-semibold">{insight.title}</h4>
          <p className="text-slate-400 text-sm mt-1 line-clamp-2">{insight.message}</p>
        </div>
        <div className="flex-shrink-0 text-slate-500">
          <svg className={`w-5 h-5 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-700/50">
          <div className="mt-4 p-4 bg-cyan-500/10 rounded-xl border border-cyan-500/20">
            <h5 className="text-cyan-400 font-semibold text-sm mb-2 flex items-center gap-2">
              <span>💡</span> Coaching Recommendation
            </h5>
            <p className="text-slate-300">{insight.recommendation}</p>
          </div>

          {Object.keys(insight.supporting_data).length > 0 && (
            <div className="mt-3 p-3 bg-slate-800/50 rounded-lg">
              <h5 className="text-slate-400 text-xs font-medium mb-2">Supporting Data:</h5>
              <div className="flex flex-wrap gap-2">
                {Object.entries(insight.supporting_data).map(([key, value]) => (
                  <div key={key} className="px-2 py-1 bg-slate-700/50 rounded text-xs">
                    <span className="text-slate-500">{key.replace(/_/g, ' ')}:</span>{' '}
                    <span className="text-white">{typeof value === 'number' ? value.toFixed(1) : String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  return `${mins} min`;
}

// ==================== ANNOTATION UI VIEW ====================
function AnnotationUIView() {
  const [stats, setStats] = useState<{
    total_frames: number;
    annotated_frames: number;
    total_annotations: number;
  } | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<{
    frames_extracted: number;
    pre_annotated: boolean;
    message: string;
  } | null>(null);
  const [extractSettings, setExtractSettings] = useState({
    frameInterval: 60,
    maxFrames: 300,
    preAnnotate: true
  });
  const [showExtractModal, setShowExtractModal] = useState(false);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/training/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch training stats:', error);
    }
  };

  const handleQuickExtract = async () => {
    setExtracting(true);
    setExtractResult(null);
    setShowExtractModal(false);

    try {
      const params = new URLSearchParams({
        frame_interval: extractSettings.frameInterval.toString(),
        max_frames: extractSettings.maxFrames.toString(),
        pre_annotate: extractSettings.preAnnotate.toString()
      });

      const response = await fetch(`/api/training/quick-extract?${params}`, {
        method: 'POST'
      });

      if (response.ok) {
        const result = await response.json();
        setExtractResult(result);
        fetchStats();
      } else {
        const error = await response.json();
        setExtractResult({
          frames_extracted: 0,
          pre_annotated: false,
          message: error.detail || 'Extraction failed'
        });
      }
    } catch (error) {
      setExtractResult({
        frames_extracted: 0,
        pre_annotated: false,
        message: 'Network error during extraction'
      });
    } finally {
      setExtracting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Quick Extract Modal */}
      {showExtractModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-xl p-6 max-w-md w-full mx-4 border border-slate-600">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <span>⚡</span> Quick Frame Extraction
            </h3>
            <p className="text-slate-400 text-sm mb-4">
              Extract frames from the most recent video for annotation.
              Much faster than full analysis - just saves frames for training.
            </p>

            <div className="space-y-4 mb-6">
              <div>
                <label className="text-sm text-slate-400 block mb-1">Frame Interval</label>
                <select
                  value={extractSettings.frameInterval}
                  onChange={e => setExtractSettings(s => ({ ...s, frameInterval: parseInt(e.target.value) }))}
                  className="w-full bg-slate-700 rounded px-3 py-2"
                >
                  <option value="30">Every 1 second (~30fps video)</option>
                  <option value="60">Every 2 seconds (recommended)</option>
                  <option value="150">Every 5 seconds</option>
                  <option value="300">Every 10 seconds</option>
                </select>
              </div>

              <div>
                <label className="text-sm text-slate-400 block mb-1">Max Frames</label>
                <select
                  value={extractSettings.maxFrames}
                  onChange={e => setExtractSettings(s => ({ ...s, maxFrames: parseInt(e.target.value) }))}
                  className="w-full bg-slate-700 rounded px-3 py-2"
                >
                  <option value="100">100 frames (~15 min to annotate)</option>
                  <option value="200">200 frames (~30 min to annotate)</option>
                  <option value="300">300 frames (~45 min to annotate)</option>
                  <option value="500">500 frames (~1.5 hrs to annotate)</option>
                </select>
              </div>

              <div className="flex items-start gap-3 p-3 bg-slate-700/50 rounded-lg">
                <input
                  type="checkbox"
                  id="preAnnotate"
                  checked={extractSettings.preAnnotate}
                  onChange={e => setExtractSettings(s => ({ ...s, preAnnotate: e.target.checked }))}
                  className="w-4 h-4 mt-1 rounded"
                />
                <label htmlFor="preAnnotate" className="text-sm">
                  <span className="text-white font-medium">Pre-annotate with YOLO</span>
                  <p className="text-slate-400 text-xs mt-1">
                    Auto-detects players and pre-fills bounding boxes.
                    You just review and correct mistakes instead of drawing from scratch.
                  </p>
                </label>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => setShowExtractModal(false)}
                className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleQuickExtract}
                className="flex-1 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg font-medium"
              >
                Extract Frames
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Stats Overview */}
      <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span>🎯</span> Training Data Annotation
          </h2>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setShowExtractModal(true)}
              disabled={extracting}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-600 rounded-lg text-sm font-medium flex items-center gap-2"
            >
              {extracting ? (
                <>
                  <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                  Extracting...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Quick Extract
                </>
              )}
            </button>
            <a
              href="/api/training/export/yolo"
              target="_blank"
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-medium flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Export YOLO
            </a>
          </div>
        </div>

        {/* Extraction Result Banner */}
        {extractResult && (
          <div className={`mb-4 p-4 rounded-lg ${extractResult.frames_extracted > 0 ? 'bg-green-600/20 border border-green-500/50' : 'bg-red-600/20 border border-red-500/50'}`}>
            <p className={`font-medium ${extractResult.frames_extracted > 0 ? 'text-green-300' : 'text-red-300'}`}>
              {extractResult.message}
            </p>
            {extractResult.frames_extracted > 0 && (
              <p className="text-sm text-slate-400 mt-1">
                {extractResult.pre_annotated
                  ? '✓ Frames have pre-filled bounding boxes. Review and correct as needed below.'
                  : 'Frames ready. Draw bounding boxes manually in the annotation canvas below.'}
              </p>
            )}
          </div>
        )}

        {stats && (
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-2xl font-bold text-cyan-400">{stats.total_frames}</div>
              <div className="text-sm text-slate-400">Total Frames</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-400">{stats.annotated_frames}</div>
              <div className="text-sm text-slate-400">Annotated</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-400">{stats.total_annotations}</div>
              <div className="text-sm text-slate-400">Total Annotations</div>
            </div>
          </div>
        )}

        <p className="text-slate-400 text-sm">
          Annotate captured frames to train your custom player detection model.
          Draw bounding boxes around players, goalkeepers, referees, and the ball.
          Export annotations in YOLO format for training.
        </p>
      </div>

      {/* Annotation Canvas */}
      <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 overflow-hidden" style={{ height: '70vh' }}>
        <AnnotationCanvas onStatsUpdate={fetchStats} />
      </div>
    </div>
  );
}

// Annotation Canvas Component (embedded version of AnnotationUI)
function AnnotationCanvas({ onStatsUpdate }: { onStatsUpdate: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Frame management
  const [frames, setFrames] = useState<any[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Annotation state
  interface Annotation {
    id: string;
    class_name: 'player' | 'ball' | 'goalkeeper' | 'referee';
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
    team?: 'home' | 'away' | 'unknown';
    track_id?: number;
  }

  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null);
  const [drawingMode, setDrawingMode] = useState<'select' | 'player' | 'ball' | 'goalkeeper' | 'referee'>('select');
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawEnd, setDrawEnd] = useState<{ x: number; y: number } | null>(null);
  const [showAnnotations, setShowAnnotations] = useState(true);

  const CLASS_COLORS: Record<string, string> = {
    player: '#3b82f6',
    ball: '#fbbf24',
    goalkeeper: '#22c55e',
    referee: '#f97316',
  };

  const TEAM_COLORS: Record<string, string> = {
    home: '#3b82f6',
    away: '#ef4444',
    unknown: '#888888',
  };

  const currentFrame = frames[currentFrameIndex];

  useEffect(() => {
    fetchFrames();
  }, []);

  useEffect(() => {
    if (currentFrame) {
      loadFrameAnnotations(currentFrame.frame_id);
    }
  }, [currentFrameIndex, frames]);

  useEffect(() => {
    drawCanvas();
  }, [annotations, selectedAnnotation, showAnnotations, drawStart, drawEnd, isDrawing]);

  const fetchFrames = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/training/frames');
      if (response.ok) {
        const data = await response.json();
        setFrames(data.frames || []);
      }
    } catch (error) {
      console.error('Failed to fetch frames:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadFrameAnnotations = async (frameId: string) => {
    try {
      const response = await fetch(`/api/training/frame/${frameId}`);
      if (response.ok) {
        const data = await response.json();
        setAnnotations(data.annotations || []);
        loadFrameImage(`/api/training/frame/${frameId}/image`);
      }
    } catch (error) {
      console.error('Failed to load frame:', error);
    }
  };

  const loadFrameImage = (url: string) => {
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      drawCanvas();
    };
    img.src = url;
  };

  const saveAnnotations = async () => {
    if (!currentFrame) return;
    setSaving(true);
    try {
      const response = await fetch(`/api/training/frame/${currentFrame.frame_id}/annotations`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(annotations),
      });
      if (response.ok) {
        setFrames(prev => prev.map((f, i) =>
          i === currentFrameIndex ? { ...f, is_annotated: true } : f
        ));
        onStatsUpdate();
      }
    } catch (error) {
      console.error('Failed to save:', error);
    } finally {
      setSaving(false);
    }
  };

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No image loaded', canvas.width / 2, canvas.height / 2);
    }

    if (showAnnotations) {
      annotations.forEach(ann => {
        const isSelected = ann.id === selectedAnnotation;
        drawAnnotation(ctx, ann, isSelected);
      });
    }

    if (isDrawing && drawStart && drawEnd) {
      ctx.strokeStyle = CLASS_COLORS[drawingMode] || '#ffffff';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      const x = Math.min(drawStart.x, drawEnd.x);
      const y = Math.min(drawStart.y, drawEnd.y);
      const w = Math.abs(drawEnd.x - drawStart.x);
      const h = Math.abs(drawEnd.y - drawStart.y);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }, [annotations, selectedAnnotation, showAnnotations, drawStart, drawEnd, isDrawing, drawingMode]);

  const drawAnnotation = (ctx: CanvasRenderingContext2D, ann: Annotation, isSelected: boolean) => {
    const { bbox, class_name, team, track_id } = ann;
    let color = CLASS_COLORS[class_name] || '#888888';
    if (class_name === 'player' && team) {
      color = TEAM_COLORS[team];
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    if (isSelected) {
      const handleSize = 8;
      ctx.fillStyle = color;
      [[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x1, bbox.y2], [bbox.x2, bbox.y2]].forEach(([x, y]) => {
        ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
      });
    }

    const label = track_id ? `${class_name} #${track_id}` : class_name;
    ctx.fillStyle = color;
    ctx.fillRect(bbox.x1, bbox.y1 - 18, Math.max(label.length * 7, 50), 18);
    ctx.fillStyle = '#ffffff';
    ctx.font = '11px sans-serif';
    ctx.fillText(label, bbox.x1 + 3, bbox.y1 - 5);
  };

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoordinates(e);
    if (drawingMode === 'select') {
      const clicked = annotations.find(ann =>
        coords.x >= ann.bbox.x1 && coords.x <= ann.bbox.x2 &&
        coords.y >= ann.bbox.y1 && coords.y <= ann.bbox.y2
      );
      setSelectedAnnotation(clicked?.id || null);
    } else {
      setIsDrawing(true);
      setDrawStart(coords);
      setDrawEnd(coords);
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDrawing) setDrawEnd(getCanvasCoordinates(e));
  };

  const handleCanvasMouseUp = () => {
    if (isDrawing && drawStart && drawEnd) {
      const x1 = Math.min(drawStart.x, drawEnd.x);
      const y1 = Math.min(drawStart.y, drawEnd.y);
      const x2 = Math.max(drawStart.x, drawEnd.x);
      const y2 = Math.max(drawStart.y, drawEnd.y);

      if (x2 - x1 > 10 && y2 - y1 > 10) {
        const newAnn: Annotation = {
          id: `ann_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          class_name: drawingMode === 'select' ? 'player' : drawingMode,
          bbox: { x1, y1, x2, y2, confidence: 1.0 },
          team: drawingMode === 'player' ? 'unknown' : undefined,
        };
        setAnnotations(prev => [...prev, newAnn]);
        setSelectedAnnotation(newAnn.id);
      }
    }
    setIsDrawing(false);
    setDrawStart(null);
    setDrawEnd(null);
  };

  const deleteSelected = () => {
    if (selectedAnnotation) {
      setAnnotations(prev => prev.filter(a => a.id !== selectedAnnotation));
      setSelectedAnnotation(null);
    }
  };

  const navigateFrame = (direction: number) => {
    const newIndex = currentFrameIndex + direction;
    if (newIndex >= 0 && newIndex < frames.length) {
      setCurrentFrameIndex(newIndex);
      setSelectedAnnotation(null);
    }
  };

  const selectedAnn = annotations.find(a => a.id === selectedAnnotation);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full" />
        <span className="ml-2 text-slate-400">Loading frames...</span>
      </div>
    );
  }

  if (frames.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8">
        <div className="text-6xl mb-4">📹</div>
        <h3 className="text-xl font-semibold text-slate-300 mb-2">No Training Frames</h3>
        <p className="text-slate-500 text-center mb-4">
          Process a video first to capture frames for annotation.
          Frames are automatically captured during video analysis.
        </p>
        <button
          type="button"
          onClick={fetchFrames}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-slate-900/50 border-b border-slate-700">
        <div className="flex items-center gap-2">
          {(['select', 'player', 'ball', 'goalkeeper', 'referee'] as const).map(mode => (
            <button
              key={mode}
              type="button"
              onClick={() => setDrawingMode(mode)}
              className={`px-3 py-1.5 rounded text-sm capitalize ${
                drawingMode === mode
                  ? mode === 'ball' ? 'bg-yellow-600 text-white'
                    : mode === 'goalkeeper' ? 'bg-green-600 text-white'
                    : mode === 'referee' ? 'bg-orange-600 text-white'
                    : 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {mode === 'select' ? '↖ Select' : mode === 'goalkeeper' ? 'GK' : mode}
            </button>
          ))}

          <div className="w-px h-6 bg-slate-600 mx-2" />

          <button
            type="button"
            onClick={deleteSelected}
            disabled={!selectedAnnotation}
            className="px-3 py-1.5 rounded text-sm bg-red-600/30 text-red-400 hover:bg-red-600/50 disabled:opacity-50"
          >
            🗑 Delete
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowAnnotations(!showAnnotations)}
            className={`px-3 py-1.5 rounded text-sm ${showAnnotations ? 'bg-slate-600' : 'bg-slate-700'}`}
          >
            {showAnnotations ? '👁 Hide' : '👁 Show'}
          </button>

          <button
            type="button"
            onClick={saveAnnotations}
            disabled={saving}
            className="px-4 py-1.5 rounded text-sm bg-green-600 hover:bg-green-700 text-white disabled:opacity-50"
          >
            {saving ? '⏳ Saving...' : '💾 Save'}
          </button>
        </div>
      </div>

      {/* Canvas and sidebar */}
      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 relative">
          <canvas
            ref={canvasRef}
            width={1920}
            height={1080}
            className="absolute inset-0 w-full h-full object-contain cursor-crosshair"
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseUp}
          />

          {/* Frame navigation */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-black/70 rounded-lg px-4 py-2">
            <button type="button" onClick={() => navigateFrame(-1)} disabled={currentFrameIndex === 0} className="p-1 hover:bg-white/10 rounded disabled:opacity-50">
              ◀
            </button>
            <span className="text-sm">
              Frame {currentFrameIndex + 1} / {frames.length}
              {currentFrame?.is_annotated && <span className="ml-2 text-green-400">✓</span>}
            </span>
            <button type="button" onClick={() => navigateFrame(1)} disabled={currentFrameIndex === frames.length - 1} className="p-1 hover:bg-white/10 rounded disabled:opacity-50">
              ▶
            </button>
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-56 bg-slate-900/50 border-l border-slate-700 p-3 overflow-y-auto">
          <h4 className="text-xs font-semibold text-slate-500 uppercase mb-2">
            Annotations ({annotations.length})
          </h4>

          <div className="space-y-1 mb-4">
            {annotations.map(ann => (
              <div
                key={ann.id}
                onClick={() => setSelectedAnnotation(ann.id)}
                className={`p-2 rounded cursor-pointer text-sm flex items-center gap-2 ${
                  ann.id === selectedAnnotation ? 'bg-blue-600/30 border border-blue-500' : 'bg-slate-800 hover:bg-slate-700'
                }`}
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{
                    backgroundColor: ann.class_name === 'player' && ann.team
                      ? TEAM_COLORS[ann.team]
                      : CLASS_COLORS[ann.class_name]
                  }}
                />
                <span className="capitalize">{ann.class_name}</span>
                {ann.track_id && <span className="text-slate-500 text-xs">#{ann.track_id}</span>}
              </div>
            ))}
          </div>

          {selectedAnn && (
            <div className="border-t border-slate-700 pt-3">
              <h4 className="text-xs font-semibold text-slate-500 mb-2">Properties</h4>

              <div className="space-y-2">
                <div>
                  <label className="text-xs text-slate-500">Class</label>
                  <select
                    value={selectedAnn.class_name}
                    onChange={e => setAnnotations(prev => prev.map(a =>
                      a.id === selectedAnnotation ? { ...a, class_name: e.target.value as any } : a
                    ))}
                    className="w-full bg-slate-800 rounded px-2 py-1 text-sm mt-1"
                  >
                    <option value="player">Player</option>
                    <option value="ball">Ball</option>
                    <option value="goalkeeper">Goalkeeper</option>
                    <option value="referee">Referee</option>
                  </select>
                </div>

                {(selectedAnn.class_name === 'player' || selectedAnn.class_name === 'goalkeeper') && (
                  <div>
                    <label className="text-xs text-slate-500">Team</label>
                    <div className="flex gap-1 mt-1">
                      {(['home', 'away', 'unknown'] as const).map(team => (
                        <button
                          key={team}
                          type="button"
                          onClick={() => setAnnotations(prev => prev.map(a =>
                            a.id === selectedAnnotation ? { ...a, team } : a
                          ))}
                          className={`flex-1 py-1 rounded text-xs capitalize ${
                            selectedAnn.team === team ? 'ring-2 ring-white' : 'opacity-60'
                          }`}
                          style={{ backgroundColor: TEAM_COLORS[team] }}
                        >
                          {team === 'unknown' ? '?' : team.charAt(0).toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <label className="text-xs text-slate-500">Track ID</label>
                  <input
                    type="number"
                    value={selectedAnn.track_id || ''}
                    onChange={e => setAnnotations(prev => prev.map(a =>
                      a.id === selectedAnnotation ? { ...a, track_id: e.target.value ? parseInt(e.target.value) : undefined } : a
                    ))}
                    placeholder="Optional"
                    className="w-full bg-slate-800 rounded px-2 py-1 text-sm mt-1"
                  />
                </div>
              </div>
            </div>
          )}

          <div className="border-t border-slate-700 pt-3 mt-4">
            <h4 className="text-xs font-semibold text-slate-500 mb-2">Shortcuts</h4>
            <div className="text-xs text-slate-500 space-y-1">
              <div><kbd className="px-1 bg-slate-800 rounded">Del</kbd> Delete selected</div>
              <div><kbd className="px-1 bg-slate-800 rounded">←/→</kbd> Navigate frames</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
