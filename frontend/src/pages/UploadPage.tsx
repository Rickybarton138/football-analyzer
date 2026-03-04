import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { useMatchStore } from '../stores/matchStore';
import { FORMATIONS, JERSEY_COLORS } from '../lib/constants';
import { Button } from '../components/ui/Button';
import { ChevronLeft, Upload, Monitor } from 'lucide-react';
import { cn } from '../lib/utils';
import type { MatchMetadata, SystemStatus, ProcessingModeType } from '../types/match';

export default function UploadPage() {
  const navigate = useNavigate();
  const { loadMatches, loadAnalysis } = useMatchStore();
  const [step, setStep] = useState<'metadata' | 'upload' | 'processing'>('metadata');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  const [, setVideoId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingModeType>('quick_preview');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [metadata, setMetadata] = useState<MatchMetadata>({
    homeTeam: '',
    awayTeam: '',
    isHomeTeam: true,
    homeJerseyColor: '#dc2626',
    awayJerseyColor: '#3b82f6',
    homeFormation: '4-4-2',
    awayFormation: '4-3-3',
    matchDate: new Date().toISOString().split('T')[0],
    competition: '',
    venue: '',
  });

  useEffect(() => {
    api.get<SystemStatus>('/api/status').then(data => {
      setSystemStatus(data);
      if (data.recommended_mode === 'full' && data.gpu?.available) {
        setProcessingMode('full');
      }
    }).catch(() => {});
  }, []);

  const handleMetadataSubmit = () => {
    if (!metadata.homeTeam || !metadata.awayTeam) return alert('Please enter both team names');
    setStep('upload');
  };

  const handleUpload = async () => {
    if (!selectedFile) return alert('Please select a video file');
    setUploading(true);
    setUploadProgress(0);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('metadata', JSON.stringify(metadata));
      const result = await api.upload('/api/video/upload', formData, setUploadProgress);
      setVideoId(result.video_id);
      setStep('processing');
      setUploading(false);

      await api.post(`/api/video/${result.video_id}/process?analysis_mode=${processingMode}`, { metadata });
      pollStatus(result.video_id);
    } catch {
      alert('Upload failed. Please try again.');
      setUploading(false);
    }
  };

  const pollStatus = async (id: string) => {
    const poll = async () => {
      try {
        const status = await api.get<{ progress_pct?: number; progress?: number; status: string; error_message?: string }>(`/api/video/${id}/status`);
        setProcessingProgress(status.progress_pct || status.progress || 0);
        setProcessingStatus(status.status || 'Processing...');
        if (status.status === 'completed' || status.status === 'complete') {
          await loadMatches();
          await loadAnalysis();
          navigate('/match/current/overview');
        } else if (status.status === 'failed' || status.status === 'error') {
          alert('Processing failed: ' + (status.error_message || 'Unknown error'));
          setStep('upload');
        } else {
          setTimeout(poll, 2000);
        }
      } catch {
        setTimeout(poll, 3000);
      }
    };
    poll();
  };

  // Step: Metadata
  if (step === 'metadata') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-surface rounded-2xl p-8 max-w-2xl w-full border border-slate-700/50">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">Match Setup</h1>
            <p className="text-slate-400">Enter match details for accurate analysis</p>
          </div>

          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Home Team</label>
                <input type="text" value={metadata.homeTeam}
                  onChange={e => setMetadata({ ...metadata, homeTeam: e.target.value })}
                  placeholder="e.g. Manchester United"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Away Team</label>
                <input type="text" value={metadata.awayTeam}
                  onChange={e => setMetadata({ ...metadata, awayTeam: e.target.value })}
                  placeholder="e.g. Liverpool"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Which team are you filming?</label>
              <div className="flex gap-4">
                {[true, false].map(isHome => (
                  <button key={String(isHome)}
                    onClick={() => setMetadata({ ...metadata, isHomeTeam: isHome })}
                    className={cn('flex-1 py-3 px-4 rounded-lg border-2 transition-all',
                      metadata.isHomeTeam === isHome
                        ? 'border-emerald-500 bg-emerald-500/10 text-emerald-400'
                        : 'border-slate-600 text-slate-400 hover:border-slate-500'
                    )}>
                    {isHome ? 'Home Team (at home)' : 'Away Team (away)'}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Home Jersey</label>
                <div className="flex flex-wrap gap-2">
                  {JERSEY_COLORS.map(c => (
                    <button key={c.value} onClick={() => setMetadata({ ...metadata, homeJerseyColor: c.value })}
                      className={cn('w-8 h-8 rounded-full border-2 transition-all',
                        metadata.homeJerseyColor === c.value ? 'border-white scale-110' : 'border-transparent'
                      )}
                      style={{ backgroundColor: c.value }} title={c.name} />
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Away Jersey</label>
                <div className="flex flex-wrap gap-2">
                  {JERSEY_COLORS.map(c => (
                    <button key={c.value} onClick={() => setMetadata({ ...metadata, awayJerseyColor: c.value })}
                      className={cn('w-8 h-8 rounded-full border-2 transition-all',
                        metadata.awayJerseyColor === c.value ? 'border-white scale-110' : 'border-transparent'
                      )}
                      style={{ backgroundColor: c.value }} title={c.name} />
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {(['home', 'away'] as const).map(side => (
                <div key={side}>
                  <label className="block text-sm font-medium text-slate-300 mb-2">{side === 'home' ? 'Home' : 'Away'} Formation</label>
                  <select
                    value={side === 'home' ? metadata.homeFormation : metadata.awayFormation}
                    onChange={e => setMetadata({ ...metadata, [side === 'home' ? 'homeFormation' : 'awayFormation']: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-emerald-500 outline-none">
                    {FORMATIONS.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Match Date</label>
                <input type="date" value={metadata.matchDate}
                  onChange={e => setMetadata({ ...metadata, matchDate: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-emerald-500 outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Competition</label>
                <input type="text" value={metadata.competition}
                  onChange={e => setMetadata({ ...metadata, competition: e.target.value })}
                  placeholder="e.g. League Cup"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500 outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Venue</label>
                <input type="text" value={metadata.venue}
                  onChange={e => setMetadata({ ...metadata, venue: e.target.value })}
                  placeholder="e.g. Old Trafford"
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500 outline-none" />
              </div>
            </div>

            <Button className="w-full py-4" onClick={handleMetadataSubmit}>
              Continue to Upload
            </Button>
            <button onClick={() => navigate('/')} className="w-full py-3 text-slate-400 hover:text-white transition-colors text-sm">
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Step: Upload
  if (step === 'upload') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-surface rounded-2xl p-8 max-w-xl w-full border border-slate-700/50">
          <button onClick={() => setStep('metadata')} className="text-slate-400 hover:text-white mb-4 flex items-center gap-2 text-sm">
            <ChevronLeft className="w-4 h-4" /> Back to Match Setup
          </button>

          <div className="text-center mb-6">
            <h2 className="text-xl font-bold text-white mb-1">Upload Match Video</h2>
            <p className="text-slate-400 text-sm">{metadata.homeTeam} vs {metadata.awayTeam}</p>
          </div>

          <div
            onClick={() => fileInputRef.current?.click()}
            className={cn('border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all mb-6',
              selectedFile ? 'border-emerald-500 bg-emerald-500/5' : 'border-slate-600 hover:border-slate-500'
            )}
          >
            <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
            {selectedFile ? (
              <>
                <p className="text-white font-medium">{selectedFile.name}</p>
                <p className="text-slate-400 text-sm">{(selectedFile.size / 1024 / 1024).toFixed(1)} MB</p>
              </>
            ) : (
              <>
                <p className="text-slate-300">Click to select video file</p>
                <p className="text-slate-500 text-xs mt-1">MP4, AVI, MOV supported</p>
              </>
            )}
            <input ref={fileInputRef} type="file" accept="video/*" className="hidden"
              onChange={e => { if (e.target.files?.[0]) setSelectedFile(e.target.files[0]); }} />
          </div>

          {/* Processing Mode */}
          {systemStatus && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-300 mb-2">Processing Mode</label>
              <div className="grid grid-cols-3 gap-2">
                {(['quick_preview', 'standard', 'full'] as const).map(mode => (
                  <button key={mode} onClick={() => setProcessingMode(mode)}
                    className={cn('p-3 rounded-lg border text-center transition-all text-sm',
                      processingMode === mode
                        ? 'border-emerald-500 bg-emerald-500/10 text-emerald-400'
                        : 'border-slate-600 text-slate-400 hover:border-slate-500'
                    )}>
                    <div className="font-medium capitalize">{mode.replace('_', ' ')}</div>
                    {systemStatus.processing_estimates?.[mode] && (
                      <div className="text-xs text-slate-500 mt-1">
                        ~{systemStatus.gpu?.available ? systemStatus.processing_estimates[mode].gpu_minutes : systemStatus.processing_estimates[mode].cpu_minutes} min
                      </div>
                    )}
                  </button>
                ))}
              </div>
              {systemStatus.gpu?.available && (
                <p className="text-emerald-400 text-xs mt-2 flex items-center gap-1">
                  <Monitor className="w-3 h-3" /> GPU: {systemStatus.gpu.name}
                </p>
              )}
            </div>
          )}

          {uploading ? (
            <div>
              <div className="h-2 bg-slate-700 rounded-full overflow-hidden mb-2">
                <div className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all" style={{ width: `${uploadProgress}%` }} />
              </div>
              <p className="text-slate-400 text-sm text-center">Uploading... {uploadProgress}%</p>
            </div>
          ) : (
            <Button className="w-full py-4" onClick={handleUpload} disabled={!selectedFile}>
              <Upload className="w-4 h-4" /> Upload & Analyze
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Step: Processing
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="bg-surface rounded-2xl p-8 max-w-md w-full border border-slate-700/50 text-center">
        <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-6" />
        <h2 className="text-xl font-bold text-white mb-2">Analyzing Match</h2>
        <p className="text-slate-400 text-sm mb-6">{processingStatus || 'Processing your video...'}</p>
        <div className="h-3 bg-slate-700 rounded-full overflow-hidden mb-2">
          <div className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all" style={{ width: `${processingProgress}%` }} />
        </div>
        <p className="text-emerald-400 font-medium">{processingProgress}%</p>
      </div>
    </div>
  );
}
