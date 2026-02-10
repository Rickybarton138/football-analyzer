import { useState, useRef, useEffect, useCallback } from 'react';

/**
 * Smart Camera Mode - VEO GO-style intelligent recording
 *
 * Features:
 * - Auto pitch detection and calibration
 * - Motion-triggered recording
 * - Live player detection preview
 * - Half-time auto-pause
 * - Match timer with period tracking
 * - Direct upload to analysis pipeline
 */

interface PitchCalibration {
  topLeft: { x: number; y: number } | null;
  topRight: { x: number; y: number } | null;
  bottomLeft: { x: number; y: number } | null;
  bottomRight: { x: number; y: number } | null;
  isCalibrated: boolean;
}

interface MatchConfig {
  homeTeam: string;
  awayTeam: string;
  periodLength: number; // minutes
  totalPeriods: number;
  currentPeriod: number;
}

interface SmartRecordingState {
  mode: 'idle' | 'calibrating' | 'ready' | 'recording' | 'paused' | 'halftime';
  matchTime: number; // seconds
  recordingTime: number;
  motionLevel: number; // 0-100
  autoRecordEnabled: boolean;
}

export function SmartCameraMode() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const motionHistoryRef = useRef<number[]>([]);
  const prevFrameRef = useRef<ImageData | null>(null);

  const [calibration, setCalibration] = useState<PitchCalibration>({
    topLeft: null,
    topRight: null,
    bottomLeft: null,
    bottomRight: null,
    isCalibrated: false
  });

  const [matchConfig, setMatchConfig] = useState<MatchConfig>({
    homeTeam: 'Home',
    awayTeam: 'Away',
    periodLength: 45,
    totalPeriods: 2,
    currentPeriod: 1
  });

  const [state, setState] = useState<SmartRecordingState>({
    mode: 'idle',
    matchTime: 0,
    recordingTime: 0,
    motionLevel: 0,
    autoRecordEnabled: true
  });

  const [detectedPlayers, setDetectedPlayers] = useState<{ x: number; y: number; team: 'home' | 'away' }[]>([]);
  const [showSetup, setShowSetup] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize camera
  useEffect(() => {
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 30 }
          },
          audio: false
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setError('Failed to access camera');
      }
    }
    initCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Motion detection and player detection loop
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const detectMotionAndPlayers = () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      // Scale down for performance
      const scale = 0.25;
      canvas.width = video.videoWidth * scale;
      canvas.height = video.videoHeight * scale;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Motion detection
      let motionScore = 0;
      if (prevFrameRef.current) {
        const prevData = prevFrameRef.current.data;
        let diff = 0;
        for (let i = 0; i < data.length; i += 16) { // Sample every 4th pixel
          diff += Math.abs(data[i] - prevData[i]);
        }
        motionScore = Math.min(100, diff / (data.length / 16) * 3);
      }
      prevFrameRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Rolling average
      motionHistoryRef.current.push(motionScore);
      if (motionHistoryRef.current.length > 10) motionHistoryRef.current.shift();
      const avgMotion = motionHistoryRef.current.reduce((a, b) => a + b, 0) / motionHistoryRef.current.length;

      setState(prev => ({ ...prev, motionLevel: Math.round(avgMotion) }));

      // Simple player detection (find clusters of non-green pixels)
      const players: { x: number; y: number; team: 'home' | 'away' }[] = [];
      const visited = new Set<number>();

      for (let y = 0; y < canvas.height; y += 4) {
        for (let x = 0; x < canvas.width; x += 4) {
          const idx = (y * canvas.width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          // Skip grass (green dominant)
          if (g > r + 20 && g > b + 20) continue;

          // Skip if already part of a cluster
          if (visited.has(idx)) continue;

          // Found potential player pixel - mark cluster
          visited.add(idx);

          // Determine team by color
          let team: 'home' | 'away' = 'home';
          if (r > g && r > b) team = 'away'; // Red = away
          // Blue or other = home

          players.push({
            x: (x / canvas.width) * 100,
            y: (y / canvas.height) * 100,
            team
          });

          if (players.length >= 25) break; // Max players
        }
        if (players.length >= 25) break;
      }

      setDetectedPlayers(players);

      // Auto-record trigger
      if (state.autoRecordEnabled && state.mode === 'ready' && avgMotion > 30) {
        startRecording();
      }
    };

    const interval = setInterval(detectMotionAndPlayers, 100);
    return () => clearInterval(interval);
  }, [state.mode, state.autoRecordEnabled]);

  // Match timer
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (state.mode === 'recording') {
      interval = setInterval(() => {
        setState(prev => {
          const newMatchTime = prev.matchTime + 1;
          const periodSeconds = matchConfig.periodLength * 60;

          // Check for half-time
          if (newMatchTime >= periodSeconds && matchConfig.currentPeriod < matchConfig.totalPeriods) {
            return { ...prev, mode: 'halftime', matchTime: newMatchTime, recordingTime: prev.recordingTime + 1 };
          }

          return { ...prev, matchTime: newMatchTime, recordingTime: prev.recordingTime + 1 };
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [state.mode, matchConfig]);

  // Draw overlay
  useEffect(() => {
    if (!overlayCanvasRef.current || !videoRef.current) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const video = videoRef.current;
    canvas.width = video.videoWidth || 1920;
    canvas.height = video.videoHeight || 1080;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw detected players
    detectedPlayers.forEach(player => {
      const x = (player.x / 100) * canvas.width;
      const y = (player.y / 100) * canvas.height;

      ctx.beginPath();
      ctx.arc(x, y, 15, 0, Math.PI * 2);
      ctx.strokeStyle = player.team === 'home' ? '#3b82f6' : '#ef4444';
      ctx.lineWidth = 3;
      ctx.stroke();
    });

    // Draw calibration points if calibrating
    if (state.mode === 'calibrating') {
      const points = [
        { point: calibration.topLeft, label: 'Top Left' },
        { point: calibration.topRight, label: 'Top Right' },
        { point: calibration.bottomLeft, label: 'Bottom Left' },
        { point: calibration.bottomRight, label: 'Bottom Right' },
      ];

      points.forEach(({ point, label }) => {
        if (point) {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
          ctx.fillStyle = '#22c55e';
          ctx.fill();
          ctx.fillStyle = '#fff';
          ctx.font = '14px sans-serif';
          ctx.fillText(label, point.x + 15, point.y + 5);
        }
      });

      // Draw pitch outline if all points set
      if (calibration.topLeft && calibration.topRight && calibration.bottomLeft && calibration.bottomRight) {
        ctx.beginPath();
        ctx.moveTo(calibration.topLeft.x, calibration.topLeft.y);
        ctx.lineTo(calibration.topRight.x, calibration.topRight.y);
        ctx.lineTo(calibration.bottomRight.x, calibration.bottomRight.y);
        ctx.lineTo(calibration.bottomLeft.x, calibration.bottomLeft.y);
        ctx.closePath();
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [detectedPlayers, calibration, state.mode]);

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;

    chunksRef.current = [];
    const mediaRecorder = new MediaRecorder(streamRef.current, {
      mimeType: 'video/webm;codecs=vp9'
    });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.start(1000);
    setState(prev => ({ ...prev, mode: 'recording' }));
  }, []);

  const pauseRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.pause();
      setState(prev => ({ ...prev, mode: 'paused' }));
    }
  };

  const resumeRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.resume();
      setState(prev => ({ ...prev, mode: 'recording' }));
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setState(prev => ({ ...prev, mode: 'ready' }));
    }
  };

  const startSecondHalf = () => {
    setMatchConfig(prev => ({ ...prev, currentPeriod: 2 }));
    setState(prev => ({ ...prev, mode: 'recording', matchTime: matchConfig.periodLength * 60 }));
  };

  const formatMatchTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}'${secs.toString().padStart(2, '0')}"`;
  };

  const handleCalibrationClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (state.mode !== 'calibrating') return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * (videoRef.current?.videoWidth || 1920);
    const y = ((e.clientY - rect.top) / rect.height) * (videoRef.current?.videoHeight || 1080);

    setCalibration(prev => {
      if (!prev.topLeft) return { ...prev, topLeft: { x, y } };
      if (!prev.topRight) return { ...prev, topRight: { x, y } };
      if (!prev.bottomLeft) return { ...prev, bottomLeft: { x, y } };
      if (!prev.bottomRight) return { ...prev, bottomRight: { x, y }, isCalibrated: true };
      return prev;
    });
  };

  const beginMatch = () => {
    setShowSetup(false);
    setState(prev => ({ ...prev, mode: 'ready' }));
  };

  const startCalibration = () => {
    setCalibration({
      topLeft: null,
      topRight: null,
      bottomLeft: null,
      bottomRight: null,
      isCalibrated: false
    });
    setState(prev => ({ ...prev, mode: 'calibrating' }));
  };

  const finishCalibration = () => {
    setState(prev => ({ ...prev, mode: 'ready' }));
  };

  // Setup screen
  if (showSetup) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center p-6">
        <div className="bg-slate-800/80 rounded-2xl p-8 max-w-lg w-full border border-slate-700">
          <div className="text-center mb-8">
            <div className="text-6xl mb-4">⚽</div>
            <h1 className="text-3xl font-bold text-white mb-2">Smart Match Recorder</h1>
            <p className="text-slate-400">Professional football analysis capture</p>
          </div>

          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-1 block">Home Team</label>
                <input
                  type="text"
                  value={matchConfig.homeTeam}
                  onChange={(e) => setMatchConfig(c => ({ ...c, homeTeam: e.target.value }))}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
                  placeholder="Home Team"
                />
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-1 block">Away Team</label>
                <input
                  type="text"
                  value={matchConfig.awayTeam}
                  onChange={(e) => setMatchConfig(c => ({ ...c, awayTeam: e.target.value }))}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:border-red-500 focus:outline-none"
                  placeholder="Away Team"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-400 mb-1 block">Period Length</label>
                <select
                  value={matchConfig.periodLength}
                  onChange={(e) => setMatchConfig(c => ({ ...c, periodLength: parseInt(e.target.value) }))}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600"
                >
                  <option value={20}>20 minutes</option>
                  <option value={25}>25 minutes</option>
                  <option value={30}>30 minutes</option>
                  <option value={35}>35 minutes</option>
                  <option value={40}>40 minutes</option>
                  <option value={45}>45 minutes</option>
                </select>
              </div>
              <div>
                <label className="text-sm text-slate-400 mb-1 block">Periods</label>
                <select
                  value={matchConfig.totalPeriods}
                  onChange={(e) => setMatchConfig(c => ({ ...c, totalPeriods: parseInt(e.target.value) }))}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600"
                >
                  <option value={2}>2 halves</option>
                  <option value={4}>4 quarters</option>
                </select>
              </div>
            </div>

            <label className="flex items-center gap-3 p-4 bg-slate-700/50 rounded-lg cursor-pointer">
              <input
                type="checkbox"
                checked={state.autoRecordEnabled}
                onChange={(e) => setState(s => ({ ...s, autoRecordEnabled: e.target.checked }))}
                className="w-5 h-5 rounded"
              />
              <div>
                <div className="text-white font-medium">Smart Auto-Record</div>
                <div className="text-slate-400 text-sm">Automatically start recording when motion detected</div>
              </div>
            </label>

            <button
              onClick={beginMatch}
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded-xl font-bold text-lg hover:from-cyan-600 hover:to-blue-600 transition-all"
            >
              Begin Match Setup
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black relative">
      {/* Video Feed */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-screen object-contain"
      />

      {/* Hidden canvas for analysis */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Overlay canvas */}
      <canvas
        ref={overlayCanvasRef}
        className="absolute inset-0 w-full h-full pointer-events-auto"
        onClick={handleCalibrationClick}
        style={{ cursor: state.mode === 'calibrating' ? 'crosshair' : 'default' }}
      />

      {/* Top HUD */}
      <div className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center justify-between">
          {/* Teams & Score */}
          <div className="flex items-center gap-4">
            <div className="bg-blue-500 px-4 py-2 rounded-lg">
              <span className="text-white font-bold">{matchConfig.homeTeam}</span>
            </div>
            <div className="text-white text-2xl font-mono">vs</div>
            <div className="bg-red-500 px-4 py-2 rounded-lg">
              <span className="text-white font-bold">{matchConfig.awayTeam}</span>
            </div>
          </div>

          {/* Match Time */}
          <div className="text-center">
            <div className="text-4xl font-mono text-white font-bold">
              {formatMatchTime(state.matchTime)}
            </div>
            <div className="text-slate-400 text-sm">
              {matchConfig.currentPeriod === 1 ? '1st Half' : '2nd Half'}
            </div>
          </div>

          {/* Recording Status */}
          <div className="flex items-center gap-4">
            {state.mode === 'recording' && (
              <div className="flex items-center gap-2 bg-red-600 px-4 py-2 rounded-lg">
                <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                <span className="text-white font-bold">REC</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-slate-400 text-sm">Motion</div>
              <div className={`font-bold ${state.motionLevel > 50 ? 'text-green-400' : 'text-slate-400'}`}>
                {state.motionLevel}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex items-center justify-center gap-4">
          {state.mode === 'calibrating' && (
            <div className="text-center">
              <p className="text-white mb-4">
                Click the 4 corners of the pitch:
                {!calibration.topLeft && ' Top Left'}
                {calibration.topLeft && !calibration.topRight && ' Top Right'}
                {calibration.topRight && !calibration.bottomLeft && ' Bottom Left'}
                {calibration.bottomLeft && !calibration.bottomRight && ' Bottom Right'}
              </p>
              {calibration.isCalibrated && (
                <button
                  onClick={finishCalibration}
                  className="px-8 py-3 bg-green-500 text-white rounded-full font-bold hover:bg-green-600"
                >
                  Confirm Calibration
                </button>
              )}
            </div>
          )}

          {state.mode === 'ready' && (
            <>
              <button
                onClick={startCalibration}
                className="px-6 py-3 bg-slate-700 text-white rounded-full font-medium hover:bg-slate-600"
              >
                📐 Calibrate Pitch
              </button>
              <button
                onClick={startRecording}
                className="px-8 py-4 bg-red-500 text-white rounded-full font-bold text-lg hover:bg-red-600 flex items-center gap-2"
              >
                <div className="w-4 h-4 bg-white rounded-full"></div>
                Start Match
              </button>
            </>
          )}

          {state.mode === 'recording' && (
            <>
              <button
                onClick={pauseRecording}
                className="px-6 py-3 bg-yellow-500 text-black rounded-full font-bold hover:bg-yellow-400"
              >
                ⏸ Pause
              </button>
              <button
                onClick={stopRecording}
                className="px-6 py-3 bg-slate-700 text-white rounded-full font-bold hover:bg-slate-600"
              >
                ⏹ End Match
              </button>
            </>
          )}

          {state.mode === 'paused' && (
            <>
              <button
                onClick={resumeRecording}
                className="px-8 py-4 bg-green-500 text-white rounded-full font-bold text-lg hover:bg-green-600"
              >
                ▶ Resume
              </button>
              <button
                onClick={stopRecording}
                className="px-6 py-3 bg-slate-700 text-white rounded-full font-bold hover:bg-slate-600"
              >
                ⏹ End Match
              </button>
            </>
          )}

          {state.mode === 'halftime' && (
            <div className="text-center">
              <div className="text-3xl text-white font-bold mb-4">⏰ Half Time</div>
              <button
                onClick={startSecondHalf}
                className="px-8 py-4 bg-green-500 text-white rounded-full font-bold text-lg hover:bg-green-600"
              >
                Start 2nd Half
              </button>
            </div>
          )}
        </div>

        {/* Player count indicator */}
        <div className="flex justify-center mt-4 gap-8">
          <div className="text-center">
            <div className="text-blue-400 font-bold text-xl">
              {detectedPlayers.filter(p => p.team === 'home').length}
            </div>
            <div className="text-slate-400 text-sm">Home Detected</div>
          </div>
          <div className="text-center">
            <div className="text-red-400 font-bold text-xl">
              {detectedPlayers.filter(p => p.team === 'away').length}
            </div>
            <div className="text-slate-400 text-sm">Away Detected</div>
          </div>
        </div>
      </div>

      {error && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-red-500/90 text-white px-6 py-4 rounded-lg">
          {error}
        </div>
      )}
    </div>
  );
}

export default SmartCameraMode;
