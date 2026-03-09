import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import type { VeoRecording } from '../lib/api';
import { useMatchStore } from '../stores/matchStore';
import { FORMATIONS, JERSEY_COLORS } from '../lib/constants';
import { Button } from '../components/ui/Button';
import { ChevronLeft, Upload, Monitor, Pause, Play, RotateCcw, Link2, Video, Download } from 'lucide-react';
import { cn } from '../lib/utils';
import type { MatchMetadata, SystemStatus, ProcessingModeType } from '../types/match';

type UploadType = 'full' | 'halves';
type UploadStep = 'metadata' | 'upload' | 'processing';
type SourceTab = 'file' | 'url' | 'veo';

interface ChunkProgress {
  pct: number;
  chunksDone: number;
  totalChunks: number;
  speedMBps: number;
  etaSeconds: number;
}

interface HalfState {
  file: File | null;
  uploading: boolean;
  progress: ChunkProgress | null;
  videoId: string | null;
  uploadId: string | null;
  done: boolean;
  error: string | null;
}

function formatEta(seconds: number): string {
  if (seconds <= 0) return '---';
  if (seconds < 60) return `~${seconds}s remaining`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `~${mins}m ${secs}s remaining`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const { loadMatches, loadAnalysis } = useMatchStore();
  const [step, setStep] = useState<UploadStep>('metadata');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [processingMode, setProcessingMode] = useState<ProcessingModeType>('quick_preview');

  // Source tab (file upload vs URL import vs VEO)
  const [sourceTab, setSourceTab] = useState<SourceTab>('file');

  // Upload type
  const [uploadType, setUploadType] = useState<UploadType>('full');
  const [matchId] = useState(() => crypto.randomUUID());

  // Full match upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<ChunkProgress | null>(null);
  const [paused, setPaused] = useState(false);
  const abortRef = useRef({ aborted: false });

  // Halves upload state
  const [firstHalf, setFirstHalf] = useState<HalfState>({
    file: null, uploading: false, progress: null, videoId: null, uploadId: null, done: false, error: null,
  });
  const [secondHalf, setSecondHalf] = useState<HalfState>({
    file: null, uploading: false, progress: null, videoId: null, uploadId: null, done: false, error: null,
  });

  // Processing state
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  const [halvesProcessing, setHalvesProcessing] = useState<{
    first?: { status: string; pct: number };
    second?: { status: string; pct: number };
  }>({});

  // Resume detection
  const [resumableUploadId, setResumableUploadId] = useState<string | null>(null);

  // URL import state
  const [importUrl, setImportUrl] = useState('');
  const [importing, setImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [importStatus, setImportStatus] = useState('');
  const [importError, setImportError] = useState('');

  // VEO state
  const [veoConnected, setVeoConnected] = useState(false);
  const [veoToken, setVeoToken] = useState('');
  const [veoConnecting, setVeoConnecting] = useState(false);
  const [veoRecordings, setVeoRecordings] = useState<VeoRecording[]>([]);
  const [veoLoading, setVeoLoading] = useState(false);
  const [veoError, setVeoError] = useState('');
  const [veoImporting, setVeoImporting] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const firstHalfInputRef = useRef<HTMLInputElement>(null);
  const secondHalfInputRef = useRef<HTMLInputElement>(null);

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

    // Check for resumable upload
    const rid = api.getResumableUpload();
    if (rid) setResumableUploadId(rid);

    // Check VEO connection status
    api.veoStatus().then(data => setVeoConnected(data.connected)).catch(() => {});

    // Check for shared file from PWA share target
    if (new URLSearchParams(window.location.search).get('shared') === 'true') {
      loadSharedFile();
    }
  }, []);

  // Load shared file from IndexedDB (PWA share target)
  const loadSharedFile = async () => {
    try {
      const db = await new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open('share-target-db', 1);
        req.onerror = () => reject(req.error);
        req.onsuccess = () => resolve(req.result);
      });
      const tx = db.transaction('shared-files', 'readwrite');
      const store = tx.objectStore('shared-files');
      const getReq = store.get('shared-video');
      getReq.onsuccess = () => {
        if (getReq.result?.file) {
          setSelectedFile(getReq.result.file);
          setSourceTab('file');
        }
        store.delete('shared-video');
      };
      db.close();
      // Clean up URL
      window.history.replaceState({}, '', '/upload');
    } catch {
      // IndexedDB not available or no shared file
    }
  };

  const handleMetadataSubmit = () => {
    if (!metadata.homeTeam || !metadata.awayTeam) return alert('Please enter both team names');
    setStep('upload');
  };

  // ── Full match chunked upload ──
  const handleFullUpload = useCallback(async () => {
    if (!selectedFile) return alert('Please select a video file');
    setUploading(true);
    setUploadProgress(null);
    abortRef.current = { aborted: false };
    setPaused(false);

    try {
      const result = await api.uploadChunked(
        selectedFile,
        metadata,
        'full',
        null,
        (info) => setUploadProgress(info),
        abortRef.current,
      );

      if (abortRef.current.aborted) {
        setUploading(false);
        return;
      }

      setStep('processing');
      setUploading(false);
      await api.post(`/api/video/${result.video_id}/process?analysis_mode=${processingMode}`, { metadata });
      pollStatus(result.video_id);
    } catch (err) {
      alert('Upload failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
      setUploading(false);
    }
  }, [selectedFile, metadata, processingMode]);

  // ── URL Import ──
  const handleUrlImport = useCallback(async () => {
    if (!importUrl.trim()) return;
    setImporting(true);
    setImportError('');
    setImportProgress(0);
    setImportStatus('Starting download...');

    try {
      const { import_id } = await api.importFromUrl(importUrl.trim(), metadata);

      // Poll import status
      const poll = async () => {
        try {
          const status = await api.getImportStatus(import_id);
          setImportProgress(status.progress_pct || 0);

          if (status.status === 'downloading') {
            setImportStatus(`Downloading... ${status.progress_pct || 0}%`);
            setTimeout(poll, 2000);
          } else if (status.status === 'processing_metadata') {
            setImportStatus('Extracting video metadata...');
            setTimeout(poll, 1000);
          } else if (status.status === 'ready' && status.video_id) {
            setImporting(false);
            setStep('processing');
            await api.post(`/api/video/${status.video_id}/process?analysis_mode=${processingMode}`, { metadata });
            pollStatus(status.video_id);
          } else if (status.status === 'failed') {
            setImporting(false);
            setImportError(status.error || 'Import failed');
          } else {
            setTimeout(poll, 2000);
          }
        } catch {
          setTimeout(poll, 3000);
        }
      };
      poll();
    } catch (err) {
      setImporting(false);
      setImportError(err instanceof Error ? err.message : 'Import failed');
    }
  }, [importUrl, metadata, processingMode]);

  // ── VEO Connect ──
  const handleVeoConnect = useCallback(async () => {
    if (!veoToken.trim()) return;
    setVeoConnecting(true);
    setVeoError('');
    try {
      await api.veoConnect(veoToken.trim());
      setVeoConnected(true);
      setVeoConnecting(false);
      // Load recordings
      handleLoadVeoRecordings();
    } catch (err) {
      setVeoConnecting(false);
      setVeoError(err instanceof Error ? err.message : 'Connection failed');
    }
  }, [veoToken]);

  const handleLoadVeoRecordings = useCallback(async () => {
    setVeoLoading(true);
    setVeoError('');
    try {
      const data = await api.veoRecordings();
      setVeoRecordings(data.recordings);
    } catch (err) {
      setVeoError(err instanceof Error ? err.message : 'Failed to load recordings');
    } finally {
      setVeoLoading(false);
    }
  }, []);

  useEffect(() => {
    if (veoConnected && sourceTab === 'veo' && veoRecordings.length === 0) {
      handleLoadVeoRecordings();
    }
  }, [veoConnected, sourceTab]);

  const handleVeoImport = useCallback(async (recordingId: string) => {
    setVeoImporting(recordingId);
    setImportError('');
    setImportProgress(0);

    try {
      const { import_id } = await api.veoImport(recordingId);

      // Poll import status — same pattern as URL import
      const poll = async () => {
        try {
          const status = await api.getImportStatus(import_id);
          setImportProgress(status.progress_pct || 0);

          if (status.status === 'downloading') {
            setImportStatus(`Downloading VEO recording... ${status.progress_pct || 0}%`);
            setTimeout(poll, 2000);
          } else if (status.status === 'processing_metadata') {
            setImportStatus('Extracting video metadata...');
            setTimeout(poll, 1000);
          } else if (status.status === 'ready' && status.video_id) {
            setVeoImporting(null);
            setStep('processing');
            await api.post(`/api/video/${status.video_id}/process?analysis_mode=${processingMode}`, { metadata });
            pollStatus(status.video_id);
          } else if (status.status === 'failed') {
            setVeoImporting(null);
            setImportError(status.error || 'Import failed');
          } else {
            setTimeout(poll, 2000);
          }
        } catch {
          setTimeout(poll, 3000);
        }
      };
      poll();
    } catch (err) {
      setVeoImporting(null);
      setImportError(err instanceof Error ? err.message : 'VEO import failed');
    }
  }, [metadata, processingMode]);

  // ── Halves upload ──
  const uploadHalf = useCallback(async (
    half: 'first' | 'second',
    file: File,
    setHalf: React.Dispatch<React.SetStateAction<HalfState>>,
  ) => {
    setHalf(prev => ({ ...prev, uploading: true, error: null }));
    try {
      const result = await api.uploadChunked(
        file,
        metadata,
        half,
        matchId,
        (info) => setHalf(prev => ({ ...prev, progress: info })),
      );
      setHalf(prev => ({ ...prev, uploading: false, videoId: result.video_id, uploadId: result.upload_id, done: true }));
      return result.video_id;
    } catch (err) {
      setHalf(prev => ({
        ...prev,
        uploading: false,
        error: err instanceof Error ? err.message : 'Upload failed',
      }));
      return null;
    }
  }, [metadata, matchId]);

  const handleHalvesUpload = useCallback(async () => {
    if (!firstHalf.file || !secondHalf.file) return alert('Please select both half files');

    const firstVid = await uploadHalf('first', firstHalf.file, setFirstHalf);
    if (!firstVid) return;

    const secondVid = await uploadHalf('second', secondHalf.file, setSecondHalf);
    if (!secondVid) return;

    setStep('processing');

    await Promise.all([
      api.post(`/api/video/${firstVid}/process?analysis_mode=${processingMode}`, { metadata }),
      api.post(`/api/video/${secondVid}/process?analysis_mode=${processingMode}`, { metadata }),
    ]);

    pollHalvesStatus(firstVid, secondVid);
  }, [firstHalf.file, secondHalf.file, metadata, processingMode, uploadHalf]);

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

  const pollHalvesStatus = async (firstVid: string, secondVid: string) => {
    let firstDone = false;
    let secondDone = false;

    const poll = async () => {
      try {
        const [s1, s2] = await Promise.all([
          api.get<{ progress_pct?: number; status: string; error_message?: string }>(`/api/video/${firstVid}/status`),
          api.get<{ progress_pct?: number; status: string; error_message?: string }>(`/api/video/${secondVid}/status`),
        ]);

        setHalvesProcessing({
          first: { status: s1.status, pct: s1.progress_pct || 0 },
          second: { status: s2.status, pct: s2.progress_pct || 0 },
        });

        if (s1.status === 'completed' || s1.status === 'complete') firstDone = true;
        if (s2.status === 'completed' || s2.status === 'complete') secondDone = true;

        if ((s1.status === 'failed' || s1.status === 'error') || (s2.status === 'failed' || s2.status === 'error')) {
          alert('Processing failed: ' + (s1.error_message || s2.error_message || 'Unknown error'));
          setStep('upload');
          return;
        }

        if (firstDone && secondDone) {
          await loadMatches();
          await loadAnalysis();
          navigate('/match/current/overview');
        } else {
          setTimeout(poll, 2000);
        }
      } catch {
        setTimeout(poll, 3000);
      }
    };
    poll();
  };

  const handlePause = () => {
    abortRef.current.aborted = true;
    setPaused(true);
  };

  const handleResume = () => {
    setPaused(false);
    handleFullUpload();
  };

  const handleDismissResume = () => {
    if (resumableUploadId) {
      api.clearResumableUpload(resumableUploadId);
      setResumableUploadId(null);
    }
  };

  // ==============================
  // Step: Metadata
  // ==============================
  if (step === 'metadata') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-surface rounded-card p-8 max-w-2xl w-full border border-border shadow-card">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-pitch-deep to-pitch rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-text-primary mb-2 font-display">Match Setup</h1>
            <p className="text-text-muted">Enter match details for accurate analysis</p>
          </div>

          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Home Team</label>
                <input type="text" value={metadata.homeTeam}
                  onChange={e => setMetadata({ ...metadata, homeTeam: e.target.value })}
                  placeholder="e.g. Manchester United"
                  className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary placeholder-text-muted focus:border-pitch focus:ring-1 focus:ring-pitch outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Away Team</label>
                <input type="text" value={metadata.awayTeam}
                  onChange={e => setMetadata({ ...metadata, awayTeam: e.target.value })}
                  placeholder="e.g. Liverpool"
                  className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary placeholder-text-muted focus:border-pitch focus:ring-1 focus:ring-pitch outline-none" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Which team are you filming?</label>
              <div className="flex gap-4">
                {[true, false].map(isHome => (
                  <button key={String(isHome)}
                    onClick={() => setMetadata({ ...metadata, isHomeTeam: isHome })}
                    className={cn('flex-1 py-3 px-4 rounded-btn border-2 transition-all font-medium',
                      metadata.isHomeTeam === isHome
                        ? 'border-pitch bg-pitch-light text-pitch-deep'
                        : 'border-border text-text-muted hover:border-gray-300'
                    )}>
                    {isHome ? 'Home Team (at home)' : 'Away Team (away)'}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Home Jersey</label>
                <div className="flex flex-wrap gap-2">
                  {JERSEY_COLORS.map(c => (
                    <button key={c.value} onClick={() => setMetadata({ ...metadata, homeJerseyColor: c.value })}
                      className={cn('w-8 h-8 rounded-full border-2 transition-all',
                        metadata.homeJerseyColor === c.value ? 'border-text-primary scale-110 shadow-md' : 'border-transparent'
                      )}
                      style={{ backgroundColor: c.value }} title={c.name} />
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Away Jersey</label>
                <div className="flex flex-wrap gap-2">
                  {JERSEY_COLORS.map(c => (
                    <button key={c.value} onClick={() => setMetadata({ ...metadata, awayJerseyColor: c.value })}
                      className={cn('w-8 h-8 rounded-full border-2 transition-all',
                        metadata.awayJerseyColor === c.value ? 'border-text-primary scale-110 shadow-md' : 'border-transparent'
                      )}
                      style={{ backgroundColor: c.value }} title={c.name} />
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {(['home', 'away'] as const).map(side => (
                <div key={side}>
                  <label className="block text-sm font-medium text-text-secondary mb-2">{side === 'home' ? 'Home' : 'Away'} Formation</label>
                  <select
                    value={side === 'home' ? metadata.homeFormation : metadata.awayFormation}
                    onChange={e => setMetadata({ ...metadata, [side === 'home' ? 'homeFormation' : 'awayFormation']: e.target.value })}
                    className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary focus:border-pitch outline-none">
                    {FORMATIONS.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Match Date</label>
                <input type="date" value={metadata.matchDate}
                  onChange={e => setMetadata({ ...metadata, matchDate: e.target.value })}
                  className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary focus:border-pitch outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Competition</label>
                <input type="text" value={metadata.competition}
                  onChange={e => setMetadata({ ...metadata, competition: e.target.value })}
                  placeholder="e.g. League Cup"
                  className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary placeholder-text-muted focus:border-pitch outline-none" />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">Venue</label>
                <input type="text" value={metadata.venue}
                  onChange={e => setMetadata({ ...metadata, venue: e.target.value })}
                  placeholder="e.g. Old Trafford"
                  className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary placeholder-text-muted focus:border-pitch outline-none" />
              </div>
            </div>

            {/* Upload Type Selector */}
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-2">Upload Type</label>
              <div className="flex gap-4">
                {([
                  { value: 'full' as UploadType, label: 'Full Match', desc: 'Single video file' },
                  { value: 'halves' as UploadType, label: 'Upload by Half', desc: '1st & 2nd half separately' },
                ]).map(opt => (
                  <button key={opt.value}
                    onClick={() => setUploadType(opt.value)}
                    className={cn('flex-1 py-3 px-4 rounded-btn border-2 transition-all',
                      uploadType === opt.value
                        ? 'border-pitch bg-pitch-light text-pitch-deep'
                        : 'border-border text-text-muted hover:border-gray-300'
                    )}>
                    <div className="font-medium">{opt.label}</div>
                    <div className="text-xs mt-0.5 opacity-70">{opt.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <Button className="w-full py-4" onClick={handleMetadataSubmit}>
              Continue to Upload
            </Button>
            <button onClick={() => navigate('/')} className="w-full py-3 text-text-muted hover:text-text-primary transition-colors text-sm">
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ==============================
  // Step: Upload
  // ==============================
  if (step === 'upload') {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-surface rounded-card p-8 max-w-xl w-full border border-border shadow-card">
          <button onClick={() => setStep('metadata')} className="text-text-muted hover:text-text-primary mb-4 flex items-center gap-2 text-sm">
            <ChevronLeft className="w-4 h-4" /> Back to Match Setup
          </button>

          <div className="text-center mb-6">
            <h2 className="text-xl font-bold text-text-primary mb-1">
              Upload Match Video{uploadType === 'halves' ? 's' : ''}
            </h2>
            <p className="text-text-muted text-sm">{metadata.homeTeam} vs {metadata.awayTeam}</p>
          </div>

          {/* Source Tab Selector */}
          {uploadType === 'full' && (
            <div className="flex border border-border rounded-btn overflow-hidden mb-6">
              {([
                { id: 'file' as SourceTab, label: 'Select File', icon: Upload },
                { id: 'url' as SourceTab, label: 'Import URL', icon: Link2 },
                { id: 'veo' as SourceTab, label: 'VEO', icon: Video },
              ]).map(tab => (
                <button key={tab.id}
                  onClick={() => setSourceTab(tab.id)}
                  className={cn('flex-1 py-2.5 px-3 text-sm font-medium flex items-center justify-center gap-1.5 transition-colors',
                    sourceTab === tab.id
                      ? 'bg-pitch text-white'
                      : 'text-text-muted hover:text-text-primary hover:bg-surface-alt'
                  )}>
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>
          )}

          {/* Resume banner */}
          {resumableUploadId && !uploading && sourceTab === 'file' && (
            <div className="mb-4 p-3 bg-amber-900/20 border border-amber-700/30 rounded-btn flex items-center justify-between">
              <div>
                <p className="text-amber-300 text-sm font-medium flex items-center gap-1.5">
                  <RotateCcw className="w-4 h-4" /> Incomplete upload detected
                </p>
                <p className="text-text-muted text-xs mt-0.5">Select the same file to resume where you left off</p>
              </div>
              <button onClick={handleDismissResume} className="text-text-muted hover:text-text-primary text-xs">
                Dismiss
              </button>
            </div>
          )}

          {/* ── File Upload Tab ── */}
          {(sourceTab === 'file' || uploadType === 'halves') && uploadType === 'full' ? (
            <>
              <div
                onClick={() => fileInputRef.current?.click()}
                className={cn('border-2 border-dashed rounded-card p-12 text-center cursor-pointer transition-all mb-6',
                  selectedFile ? 'border-pitch bg-pitch-light' : 'border-border hover:border-gray-300'
                )}
              >
                <Upload className="w-10 h-10 text-text-muted mx-auto mb-3" />
                {selectedFile ? (
                  <>
                    <p className="text-text-primary font-medium">{selectedFile.name}</p>
                    <p className="text-text-muted text-sm">{formatSize(selectedFile.size)}</p>
                  </>
                ) : (
                  <>
                    <p className="text-text-secondary">Click to select video file</p>
                    <p className="text-text-muted text-xs mt-1">MP4, AVI, MOV supported — any size</p>
                  </>
                )}
                <input ref={fileInputRef} type="file" accept="video/*" className="hidden"
                  onChange={e => { if (e.target.files?.[0]) setSelectedFile(e.target.files[0]); }} />
              </div>

              {systemStatus && <ProcessingModeSelector systemStatus={systemStatus} processingMode={processingMode} setProcessingMode={setProcessingMode} />}

              {uploading && uploadProgress ? (
                <div>
                  <div className="h-3 bg-surface-alt rounded-full overflow-hidden mb-2">
                    <div className="h-full bg-gradient-to-r from-pitch-deep to-pitch transition-all" style={{ width: `${uploadProgress.pct}%` }} />
                  </div>
                  <div className="flex justify-between items-center text-sm mb-1">
                    <span className="text-text-secondary">
                      Uploading chunk {uploadProgress.chunksDone} of {uploadProgress.totalChunks}
                    </span>
                    <span className="text-pitch font-medium">{uploadProgress.pct}%</span>
                  </div>
                  <div className="flex justify-between text-xs text-text-muted">
                    <span>{uploadProgress.speedMBps} MB/s</span>
                    <span>{formatEta(uploadProgress.etaSeconds)}</span>
                  </div>
                  <button onClick={handlePause}
                    className="mt-3 w-full py-2 border border-border rounded-btn text-text-muted hover:text-text-primary flex items-center justify-center gap-2 text-sm transition-colors">
                    <Pause className="w-4 h-4" /> Pause Upload
                  </button>
                </div>
              ) : paused ? (
                <div>
                  <p className="text-amber-300 text-sm text-center mb-3">Upload paused — you can resume anytime</p>
                  <Button className="w-full py-4" onClick={handleResume}>
                    <Play className="w-4 h-4" /> Resume Upload
                  </Button>
                </div>
              ) : (
                <Button className="w-full py-4" onClick={handleFullUpload} disabled={!selectedFile}>
                  <Upload className="w-4 h-4" /> Upload & Analyse
                </Button>
              )}
            </>
          ) : sourceTab === 'url' && uploadType === 'full' ? (
            /* ── URL Import Tab ── */
            <>
              <div className="mb-6">
                <label className="block text-sm font-medium text-text-secondary mb-2">Video URL</label>
                <input type="url" value={importUrl}
                  onChange={e => setImportUrl(e.target.value)}
                  placeholder="Paste Google Drive, Dropbox, or direct video link"
                  disabled={importing}
                  className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary placeholder-text-muted focus:border-pitch focus:ring-1 focus:ring-pitch outline-none disabled:opacity-50" />
                <p className="text-text-muted text-xs mt-2">
                  Supported: Google Drive share links, Dropbox links, direct .mp4 URLs
                </p>
              </div>

              {systemStatus && <ProcessingModeSelector systemStatus={systemStatus} processingMode={processingMode} setProcessingMode={setProcessingMode} />}

              {importing ? (
                <div>
                  <div className="h-3 bg-surface-alt rounded-full overflow-hidden mb-2">
                    <div className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all" style={{ width: `${importProgress}%` }} />
                  </div>
                  <p className="text-text-secondary text-sm text-center">{importStatus}</p>
                </div>
              ) : (
                <>
                  {importError && (
                    <div className="mb-4 p-3 bg-red-900/20 border border-red-700/30 rounded-btn">
                      <p className="text-red-400 text-sm">{importError}</p>
                    </div>
                  )}
                  <Button className="w-full py-4" onClick={handleUrlImport} disabled={!importUrl.trim()}>
                    <Download className="w-4 h-4" /> Import & Analyse
                  </Button>
                </>
              )}
            </>
          ) : sourceTab === 'veo' && uploadType === 'full' ? (
            /* ── VEO Browser Tab ── */
            <>
              {!veoConnected ? (
                <div className="text-center">
                  <Video className="w-12 h-12 text-text-muted mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-text-primary mb-2">Connect VEO Account</h3>
                  <p className="text-text-muted text-sm mb-4">
                    Enter your VEO API bearer token to browse and import recordings.
                  </p>
                  <input type="password" value={veoToken}
                    onChange={e => setVeoToken(e.target.value)}
                    placeholder="Paste your VEO API token"
                    disabled={veoConnecting}
                    className="w-full px-4 py-3 bg-background border border-border rounded-btn text-text-primary placeholder-text-muted focus:border-pitch focus:ring-1 focus:ring-pitch outline-none mb-3 disabled:opacity-50" />
                  {veoError && (
                    <div className="mb-3 p-2 bg-red-900/20 border border-red-700/30 rounded-btn">
                      <p className="text-red-400 text-sm">{veoError}</p>
                    </div>
                  )}
                  <Button className="w-full py-3" onClick={handleVeoConnect} disabled={!veoToken.trim() || veoConnecting}>
                    {veoConnecting ? 'Connecting...' : 'Connect VEO'}
                  </Button>
                </div>
              ) : veoImporting ? (
                <div>
                  <div className="h-3 bg-surface-alt rounded-full overflow-hidden mb-2">
                    <div className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all" style={{ width: `${importProgress}%` }} />
                  </div>
                  <p className="text-text-secondary text-sm text-center">{importStatus}</p>
                </div>
              ) : (
                <>
                  {veoLoading ? (
                    <div className="text-center py-8">
                      <div className="w-8 h-8 border-2 border-pitch border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                      <p className="text-text-muted text-sm">Loading recordings...</p>
                    </div>
                  ) : veoRecordings.length === 0 ? (
                    <div className="text-center py-8">
                      <Video className="w-10 h-10 text-text-muted mx-auto mb-3" />
                      <p className="text-text-muted">No recordings found in your VEO account</p>
                      <button onClick={handleLoadVeoRecordings} className="text-pitch text-sm mt-2 hover:underline">
                        Refresh
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-80 overflow-y-auto mb-4">
                      {veoRecordings.map(rec => (
                        <div key={rec.id}
                          className="flex items-center gap-3 p-3 bg-background border border-border rounded-btn hover:border-pitch/50 transition-colors group">
                          {rec.thumbnail_url ? (
                            <img src={rec.thumbnail_url} alt="" className="w-20 h-12 object-cover rounded" />
                          ) : (
                            <div className="w-20 h-12 bg-surface-alt rounded flex items-center justify-center">
                              <Video className="w-5 h-5 text-text-muted" />
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="text-text-primary text-sm font-medium truncate">{rec.title}</p>
                            <p className="text-text-muted text-xs">
                              {rec.date ? new Date(rec.date).toLocaleDateString() : ''}
                              {rec.duration ? ` · ${formatDuration(rec.duration)}` : ''}
                            </p>
                          </div>
                          <Button className="py-1.5 px-3 text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={() => handleVeoImport(rec.id)}>
                            Import
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}

                  {veoError && (
                    <div className="mb-3 p-2 bg-red-900/20 border border-red-700/30 rounded-btn">
                      <p className="text-red-400 text-sm">{veoError}</p>
                    </div>
                  )}

                  {systemStatus && <ProcessingModeSelector systemStatus={systemStatus} processingMode={processingMode} setProcessingMode={setProcessingMode} />}
                </>
              )}
            </>
          ) : (
            /* ── Halves Upload ── */
            <>
              <div className="space-y-4 mb-6">
                {([
                  { label: '1st Half', half: firstHalf, setHalf: setFirstHalf, inputRef: firstHalfInputRef },
                  { label: '2nd Half', half: secondHalf, setHalf: setSecondHalf, inputRef: secondHalfInputRef },
                ] as const).map(({ label, half, setHalf, inputRef }) => (
                  <div key={label}>
                    <label className="block text-sm font-medium text-text-secondary mb-2">{label}</label>
                    <div
                      onClick={() => !half.uploading && inputRef.current?.click()}
                      className={cn('border-2 border-dashed rounded-btn p-6 text-center cursor-pointer transition-all',
                        half.file ? 'border-pitch bg-pitch-light' : 'border-border hover:border-gray-300',
                        half.done && 'border-green-500 bg-green-900/10',
                        half.uploading && 'pointer-events-none opacity-70',
                      )}
                    >
                      {half.done ? (
                        <p className="text-green-400 font-medium text-sm">Uploaded successfully</p>
                      ) : half.uploading && half.progress ? (
                        <div>
                          <div className="h-2 bg-surface-alt rounded-full overflow-hidden mb-1.5">
                            <div className="h-full bg-gradient-to-r from-pitch-deep to-pitch transition-all" style={{ width: `${half.progress.pct}%` }} />
                          </div>
                          <p className="text-text-muted text-xs">
                            Chunk {half.progress.chunksDone}/{half.progress.totalChunks} — {half.progress.pct}% — {half.progress.speedMBps} MB/s
                          </p>
                        </div>
                      ) : half.file ? (
                        <>
                          <p className="text-text-primary font-medium text-sm">{half.file.name}</p>
                          <p className="text-text-muted text-xs">{formatSize(half.file.size)}</p>
                        </>
                      ) : (
                        <p className="text-text-muted text-sm">Click to select {label.toLowerCase()} video</p>
                      )}
                      {half.error && <p className="text-red-400 text-xs mt-1">{half.error}</p>}
                      <input ref={inputRef} type="file" accept="video/*" className="hidden"
                        onChange={e => {
                          if (e.target.files?.[0]) setHalf(prev => ({ ...prev, file: e.target.files![0] }));
                        }} />
                    </div>
                  </div>
                ))}
              </div>

              {systemStatus && <ProcessingModeSelector systemStatus={systemStatus} processingMode={processingMode} setProcessingMode={setProcessingMode} />}

              <Button className="w-full py-4" onClick={handleHalvesUpload}
                disabled={!firstHalf.file || !secondHalf.file || firstHalf.uploading || secondHalf.uploading}>
                <Upload className="w-4 h-4" /> Upload Both Halves & Analyse
              </Button>
            </>
          )}
        </div>
      </div>
    );
  }

  // ==============================
  // Step: Processing
  // ==============================
  if (uploadType === 'halves' && halvesProcessing.first) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="bg-surface rounded-card p-8 max-w-md w-full border border-border shadow-card text-center">
          <div className="w-16 h-16 border-4 border-pitch border-t-transparent rounded-full animate-spin mx-auto mb-6" />
          <h2 className="text-xl font-bold text-text-primary mb-4">Analysing Match</h2>

          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-text-secondary">1st Half</span>
              <span className="text-pitch font-medium">{halvesProcessing.first?.pct || 0}%</span>
            </div>
            <div className="h-2.5 bg-surface-alt rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-pitch-deep to-pitch transition-all"
                style={{ width: `${halvesProcessing.first?.pct || 0}%` }} />
            </div>
            <p className="text-text-muted text-xs mt-1">{halvesProcessing.first?.status || 'Waiting...'}</p>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-text-secondary">2nd Half</span>
              <span className="text-pitch font-medium">{halvesProcessing.second?.pct || 0}%</span>
            </div>
            <div className="h-2.5 bg-surface-alt rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-pitch-deep to-pitch transition-all"
                style={{ width: `${halvesProcessing.second?.pct || 0}%` }} />
            </div>
            <p className="text-text-muted text-xs mt-1">{halvesProcessing.second?.status || 'Waiting...'}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="bg-surface rounded-card p-8 max-w-md w-full border border-border shadow-card text-center">
        <div className="w-16 h-16 border-4 border-pitch border-t-transparent rounded-full animate-spin mx-auto mb-6" />
        <h2 className="text-xl font-bold text-text-primary mb-2">Analysing Match</h2>
        <p className="text-text-muted text-sm mb-6">{processingStatus || 'Processing your video...'}</p>
        <div className="h-3 bg-surface-alt rounded-full overflow-hidden mb-2">
          <div className="h-full bg-gradient-to-r from-pitch-deep to-pitch transition-all" style={{ width: `${processingProgress}%` }} />
        </div>
        <p className="text-pitch font-medium">{processingProgress}%</p>
      </div>
    </div>
  );
}

// ── Processing Mode Selector (extracted to avoid duplication) ──
function ProcessingModeSelector({ systemStatus, processingMode, setProcessingMode }: {
  systemStatus: SystemStatus;
  processingMode: ProcessingModeType;
  setProcessingMode: (mode: ProcessingModeType) => void;
}) {
  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-text-secondary mb-2">Processing Mode</label>
      <div className="grid grid-cols-3 gap-2">
        {(['quick_preview', 'standard', 'full'] as const).map(mode => (
          <button key={mode} onClick={() => setProcessingMode(mode)}
            className={cn('p-3 rounded-btn border text-center transition-all text-sm',
              processingMode === mode
                ? 'border-pitch bg-pitch-light text-pitch-deep font-medium'
                : 'border-border text-text-muted hover:border-gray-300'
            )}>
            <div className="font-medium capitalize">{mode.replace('_', ' ')}</div>
            {systemStatus.processing_estimates?.[mode] && (
              <div className="text-xs text-text-muted mt-1">
                ~{systemStatus.gpu?.available ? systemStatus.processing_estimates[mode].gpu_minutes : systemStatus.processing_estimates[mode].cpu_minutes} min
              </div>
            )}
          </button>
        ))}
      </div>
      {systemStatus.gpu?.available && (
        <p className="text-pitch text-xs mt-2 flex items-center gap-1">
          <Monitor className="w-3 h-3" /> GPU: {systemStatus.gpu.name}
        </p>
      )}
    </div>
  );
}
