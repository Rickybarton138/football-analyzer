import { useState, useRef, useEffect, useCallback } from 'react';

interface CameraSettings {
  resolution: '1080p' | '4K';
  frameRate: 30 | 60;
  stabilization: boolean;
  gridOverlay: boolean;
  qualityIndicator: boolean;
}

interface QualityMetrics {
  stability: number;      // 0-100
  brightness: number;     // 0-100
  coverage: number;       // 0-100 (estimated pitch coverage)
  sharpness: number;      // 0-100
  overall: number;        // 0-100
}

interface RecordingState {
  isRecording: boolean;
  duration: number;
  fileSize: number;
}

export function CameraCapture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [settings, setSettings] = useState<CameraSettings>({
    resolution: '1080p',
    frameRate: 30,
    stabilization: true,
    gridOverlay: true,
    qualityIndicator: true
  });
  const [quality, setQuality] = useState<QualityMetrics>({
    stability: 0,
    brightness: 0,
    coverage: 0,
    sharpness: 0,
    overall: 0
  });
  const [recording, setRecording] = useState<RecordingState>({
    isRecording: false,
    duration: 0,
    fileSize: 0
  });
  const [error, setError] = useState<string | null>(null);

  // Previous frame for stability detection
  const prevFrameRef = useRef<ImageData | null>(null);
  const stabilityHistoryRef = useRef<number[]>([]);

  // Get available cameras
  useEffect(() => {
    async function getCameras() {
      try {
        // Request permission first
        await navigator.mediaDevices.getUserMedia({ video: true });
        setHasPermission(true);

        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        setAvailableCameras(cameras);
        if (cameras.length > 0 && !selectedCamera) {
          setSelectedCamera(cameras[0].deviceId);
        }
      } catch (err) {
        console.error('Camera access error:', err);
        setHasPermission(false);
        setError('Camera access denied. Please allow camera permissions.');
      }
    }
    getCameras();
  }, []);

  // Start camera stream
  const startCamera = useCallback(async () => {
    if (!selectedCamera) return;

    try {
      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedCamera,
          width: settings.resolution === '4K' ? 3840 : 1920,
          height: settings.resolution === '4K' ? 2160 : 1080,
          frameRate: settings.frameRate,
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setError(null);
    } catch (err) {
      console.error('Failed to start camera:', err);
      setError('Failed to access camera with requested settings. Try lower resolution.');
    }
  }, [selectedCamera, settings]);

  useEffect(() => {
    if (hasPermission && selectedCamera) {
      startCamera();
    }
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [hasPermission, selectedCamera, startCamera]);

  // Quality analysis loop
  useEffect(() => {
    if (!settings.qualityIndicator || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const analyzeFrame = () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      // Draw current frame to canvas (scaled down for performance)
      const scale = 0.25;
      canvas.width = video.videoWidth * scale;
      canvas.height = video.videoHeight * scale;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Calculate brightness (average luminance)
      let totalBrightness = 0;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        totalBrightness += (0.299 * r + 0.587 * g + 0.114 * b);
      }
      const avgBrightness = totalBrightness / (data.length / 4);
      const brightnessScore = Math.min(100, Math.max(0,
        avgBrightness > 127 ? 100 - (avgBrightness - 127) * 0.5 : avgBrightness * 0.8
      ));

      // Calculate stability (frame difference)
      let stabilityScore = 100;
      if (prevFrameRef.current) {
        const prevData = prevFrameRef.current.data;
        let diff = 0;
        const sampleStep = 20; // Sample every 20th pixel for performance
        for (let i = 0; i < data.length; i += 4 * sampleStep) {
          diff += Math.abs(data[i] - prevData[i]);
          diff += Math.abs(data[i + 1] - prevData[i + 1]);
          diff += Math.abs(data[i + 2] - prevData[i + 2]);
        }
        const avgDiff = diff / (data.length / (4 * sampleStep) * 3);
        stabilityScore = Math.max(0, 100 - avgDiff * 2);
      }
      prevFrameRef.current = imageData;

      // Rolling average for stability
      stabilityHistoryRef.current.push(stabilityScore);
      if (stabilityHistoryRef.current.length > 30) {
        stabilityHistoryRef.current.shift();
      }
      const avgStability = stabilityHistoryRef.current.reduce((a, b) => a + b, 0) / stabilityHistoryRef.current.length;

      // Estimate sharpness (edge detection via Sobel-like approach)
      let edgeSum = 0;
      const width = canvas.width;
      for (let y = 1; y < canvas.height - 1; y++) {
        for (let x = 1; x < width - 1; x += 4) {
          const idx = (y * width + x) * 4;
          const gx = Math.abs(data[idx - 4] - data[idx + 4]);
          const gy = Math.abs(data[idx - width * 4] - data[idx + width * 4]);
          edgeSum += gx + gy;
        }
      }
      const sharpnessScore = Math.min(100, edgeSum / (canvas.width * canvas.height) * 2);

      // Estimate pitch coverage (green detection)
      let greenPixels = 0;
      for (let i = 0; i < data.length; i += 4 * 10) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // Check if pixel is "grass green"
        if (g > r && g > b && g > 50) {
          greenPixels++;
        }
      }
      const greenRatio = greenPixels / (data.length / (4 * 10));
      const coverageScore = Math.min(100, greenRatio * 150); // Expect ~70% green for good coverage

      // Overall score (weighted average)
      const overall = (
        avgStability * 0.35 +
        brightnessScore * 0.2 +
        coverageScore * 0.25 +
        sharpnessScore * 0.2
      );

      setQuality({
        stability: Math.round(avgStability),
        brightness: Math.round(brightnessScore),
        coverage: Math.round(coverageScore),
        sharpness: Math.round(sharpnessScore),
        overall: Math.round(overall)
      });
    };

    const interval = setInterval(analyzeFrame, 200); // 5 FPS analysis
    return () => clearInterval(interval);
  }, [settings.qualityIndicator]);

  // Recording timer
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (recording.isRecording) {
      interval = setInterval(() => {
        setRecording(prev => ({
          ...prev,
          duration: prev.duration + 1
        }));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [recording.isRecording]);

  const startRecording = () => {
    if (!streamRef.current) return;

    chunksRef.current = [];
    const options = { mimeType: 'video/webm;codecs=vp9' };

    try {
      const mediaRecorder = new MediaRecorder(streamRef.current, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
          setRecording(prev => ({
            ...prev,
            fileSize: chunksRef.current.reduce((acc, chunk) => acc + chunk.size, 0)
          }));
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        downloadRecording(blob);
      };

      mediaRecorder.start(1000); // Collect data every second
      setRecording({ isRecording: true, duration: 0, fileSize: 0 });
    } catch (err) {
      console.error('Recording error:', err);
      setError('Failed to start recording');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording.isRecording) {
      mediaRecorderRef.current.stop();
      setRecording(prev => ({ ...prev, isRecording: false }));
    }
  };

  const downloadRecording = (blob: Blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `football-match-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const uploadToAnalyzer = async () => {
    if (chunksRef.current.length === 0) {
      setError('No recording to upload');
      return;
    }

    const blob = new Blob(chunksRef.current, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('file', blob, `match-${Date.now()}.webm`);

    try {
      const response = await fetch('http://localhost:8000/api/video/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Upload successful! Video ID: ${result.video_id}`);
      } else {
        throw new Error('Upload failed');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError('Failed to upload recording');
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    if (score >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getQualityBg = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    if (score >= 40) return 'bg-orange-500';
    return 'bg-red-500';
  };

  if (hasPermission === null) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-800 rounded-xl">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-slate-300">Requesting camera access...</p>
        </div>
      </div>
    );
  }

  if (hasPermission === false) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-800 rounded-xl">
        <div className="text-center p-8">
          <div className="text-6xl mb-4">📷</div>
          <h3 className="text-xl font-bold text-white mb-2">Camera Access Required</h3>
          <p className="text-slate-400 mb-4">
            Please allow camera access to use the capture feature.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="text-2xl">🎥</span>
              Match Capture
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Record optimized footage for tactical analysis
            </p>
          </div>
          {recording.isRecording && (
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-red-400 font-mono font-bold">
                {formatDuration(recording.duration)}
              </span>
              <span className="text-slate-400">
                ({formatFileSize(recording.fileSize)})
              </span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 text-red-300">
          {error}
          <button onClick={() => setError(null)} className="ml-4 underline">Dismiss</button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Video Preview */}
        <div className="lg:col-span-3">
          <div className="bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700/50 relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full aspect-video bg-black"
            />

            {/* Hidden canvas for analysis */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Grid Overlay */}
            {settings.gridOverlay && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 56.25">
                {/* Rule of thirds */}
                <line x1="33.33" y1="0" x2="33.33" y2="56.25" stroke="rgba(255,255,255,0.3)" strokeWidth="0.2" strokeDasharray="1,1" />
                <line x1="66.67" y1="0" x2="66.67" y2="56.25" stroke="rgba(255,255,255,0.3)" strokeWidth="0.2" strokeDasharray="1,1" />
                <line x1="0" y1="18.75" x2="100" y2="18.75" stroke="rgba(255,255,255,0.3)" strokeWidth="0.2" strokeDasharray="1,1" />
                <line x1="0" y1="37.5" x2="100" y2="37.5" stroke="rgba(255,255,255,0.3)" strokeWidth="0.2" strokeDasharray="1,1" />

                {/* Pitch outline guide */}
                <rect x="5" y="5" width="90" height="46.25" fill="none" stroke="rgba(0,255,0,0.4)" strokeWidth="0.3" strokeDasharray="2,1" />

                {/* Center guides */}
                <line x1="50" y1="5" x2="50" y2="51.25" stroke="rgba(0,255,0,0.3)" strokeWidth="0.2" strokeDasharray="1,2" />
                <circle cx="50" cy="28.125" r="8" fill="none" stroke="rgba(0,255,0,0.3)" strokeWidth="0.2" />

                {/* Goal areas */}
                <rect x="5" y="18" width="10" height="20" fill="none" stroke="rgba(0,255,0,0.3)" strokeWidth="0.2" />
                <rect x="85" y="18" width="10" height="20" fill="none" stroke="rgba(0,255,0,0.3)" strokeWidth="0.2" />

                {/* Corner markers */}
                <text x="8" y="10" fill="rgba(255,255,255,0.5)" fontSize="3">Align pitch edges</text>
              </svg>
            )}

            {/* Quality Indicator Overlay */}
            {settings.qualityIndicator && (
              <div className="absolute top-4 right-4 bg-black/70 rounded-lg p-3 text-sm">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-slate-400">Quality:</span>
                  <span className={`font-bold ${getQualityColor(quality.overall)}`}>
                    {quality.overall}%
                  </span>
                </div>
                <div className="w-32 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${getQualityBg(quality.overall)}`}
                    style={{ width: `${quality.overall}%` }}
                  />
                </div>
              </div>
            )}

            {/* Recording indicator */}
            {recording.isRecording && (
              <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <span className="text-white font-medium text-sm">REC</span>
              </div>
            )}
          </div>

          {/* Recording Controls */}
          <div className="mt-4 flex items-center justify-center gap-4">
            {!recording.isRecording ? (
              <button
                onClick={startRecording}
                className="px-8 py-3 bg-red-500 hover:bg-red-600 text-white rounded-full font-medium flex items-center gap-2 transition-all"
              >
                <div className="w-4 h-4 bg-white rounded-full"></div>
                Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="px-8 py-3 bg-slate-600 hover:bg-slate-700 text-white rounded-full font-medium flex items-center gap-2 transition-all"
              >
                <div className="w-4 h-4 bg-white rounded-sm"></div>
                Stop Recording
              </button>
            )}

            {chunksRef.current.length > 0 && !recording.isRecording && (
              <>
                <button
                  onClick={() => downloadRecording(new Blob(chunksRef.current, { type: 'video/webm' }))}
                  className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-full font-medium flex items-center gap-2"
                >
                  💾 Download
                </button>
                <button
                  onClick={uploadToAnalyzer}
                  className="px-6 py-3 bg-cyan-500 hover:bg-cyan-600 text-white rounded-full font-medium flex items-center gap-2"
                >
                  📤 Analyze Now
                </button>
              </>
            )}
          </div>
        </div>

        {/* Settings Panel */}
        <div className="space-y-4">
          {/* Camera Selection */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <h3 className="font-bold text-white mb-3">Camera</h3>
            <select
              value={selectedCamera}
              onChange={(e) => setSelectedCamera(e.target.value)}
              className="w-full bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600"
            >
              {availableCameras.map(cam => (
                <option key={cam.deviceId} value={cam.deviceId}>
                  {cam.label || `Camera ${cam.deviceId.slice(0, 8)}`}
                </option>
              ))}
            </select>
          </div>

          {/* Quality Metrics */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <h3 className="font-bold text-white mb-3">Quality Metrics</h3>
            <div className="space-y-3">
              {[
                { label: 'Stability', value: quality.stability, icon: '📐' },
                { label: 'Brightness', value: quality.brightness, icon: '☀️' },
                { label: 'Coverage', value: quality.coverage, icon: '🏟️' },
                { label: 'Sharpness', value: quality.sharpness, icon: '🔍' },
              ].map(metric => (
                <div key={metric.label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">{metric.icon} {metric.label}</span>
                    <span className={getQualityColor(metric.value)}>{metric.value}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${getQualityBg(metric.value)}`}
                      style={{ width: `${metric.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Settings */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
            <h3 className="font-bold text-white mb-3">Settings</h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm text-slate-400">Resolution</label>
                <select
                  value={settings.resolution}
                  onChange={(e) => setSettings(s => ({ ...s, resolution: e.target.value as any }))}
                  className="w-full mt-1 bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600"
                >
                  <option value="1080p">1080p (Recommended)</option>
                  <option value="4K">4K (Best quality)</option>
                </select>
              </div>

              <div>
                <label className="text-sm text-slate-400">Frame Rate</label>
                <select
                  value={settings.frameRate}
                  onChange={(e) => setSettings(s => ({ ...s, frameRate: parseInt(e.target.value) as any }))}
                  className="w-full mt-1 bg-slate-700 text-white px-3 py-2 rounded-lg border border-slate-600"
                >
                  <option value={30}>30 FPS (Standard)</option>
                  <option value={60}>60 FPS (Smooth)</option>
                </select>
              </div>

              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={settings.gridOverlay}
                  onChange={(e) => setSettings(s => ({ ...s, gridOverlay: e.target.checked }))}
                  className="rounded"
                />
                Show pitch alignment guide
              </label>

              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={settings.qualityIndicator}
                  onChange={(e) => setSettings(s => ({ ...s, qualityIndicator: e.target.checked }))}
                  className="rounded"
                />
                Show quality indicator
              </label>
            </div>
          </div>

          {/* Tips */}
          <div className="bg-slate-800/50 rounded-xl p-4 border border-cyan-500/30">
            <h3 className="font-bold text-cyan-400 mb-2">📋 Recording Tips</h3>
            <ul className="text-sm text-slate-300 space-y-1">
              <li>• Position camera high and central</li>
              <li>• Keep the entire pitch in frame</li>
              <li>• Use a tripod for stability</li>
              <li>• Ensure good lighting</li>
              <li>• Aim for 80%+ quality score</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CameraCapture;
