import { useRef, useEffect, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, Maximize2 } from 'lucide-react';
import type { FrameDetection, DetectedPlayer, DetectedBall } from '../types';

interface VideoPlayerProps {
  videoUrl: string | null;
  currentFrame: FrameDetection | null;
  isLive: boolean;
}

export function VideoPlayer({ videoUrl, currentFrame, isLive }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Draw overlay on canvas
  useEffect(() => {
    if (!canvasRef.current || !currentFrame) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw player bounding boxes and IDs
    currentFrame.players.forEach((player) => {
      drawPlayer(ctx, player);
    });

    // Draw ball
    if (currentFrame.ball) {
      drawBall(ctx, currentFrame.ball);
    }
  }, [currentFrame]);

  const drawPlayer = (ctx: CanvasRenderingContext2D, player: DetectedPlayer) => {
    const { bbox, track_id, team, is_goalkeeper } = player;

    // Color based on team
    let color = '#888888'; // Unknown
    if (team === 'home') color = '#3b82f6'; // Blue
    else if (team === 'away') color = '#ef4444'; // Red

    // Goalkeeper gets different styling
    if (is_goalkeeper) {
      color = team === 'home' ? '#22c55e' : '#f97316';
    }

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    // Draw track ID label
    ctx.fillStyle = color;
    ctx.fillRect(bbox.x1, bbox.y1 - 20, 30, 20);
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    ctx.fillText(`#${track_id}`, bbox.x1 + 4, bbox.y1 - 6);

    // Draw confidence indicator
    const confidenceWidth = (bbox.x2 - bbox.x1) * bbox.confidence;
    ctx.fillStyle = color;
    ctx.fillRect(bbox.x1, bbox.y2, confidenceWidth, 3);
  };

  const drawBall = (ctx: CanvasRenderingContext2D, ball: DetectedBall) => {
    const { pixel_position, velocity } = ball;

    // Draw ball circle
    ctx.beginPath();
    ctx.arc(pixel_position.x, pixel_position.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#fbbf24';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw velocity vector if available
    if (velocity && velocity.speed_kmh > 5) {
      const scale = 3;
      ctx.beginPath();
      ctx.moveTo(pixel_position.x, pixel_position.y);
      ctx.lineTo(
        pixel_position.x + velocity.vx * scale,
        pixel_position.y + velocity.vy * scale
      );
      ctx.strokeStyle = '#f472b6';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  };

  const handlePlayPause = () => {
    if (!videoRef.current) return;

    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!videoUrl && !isLive) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900">
        <p className="text-slate-400">No video loaded</p>
      </div>
    );
  }

  if (isLive) {
    return (
      <div className="relative h-full bg-black">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          width={1920}
          height={1080}
        />
        <div className="absolute top-2 left-2 flex items-center gap-2 bg-red-500/80 px-2 py-1 rounded text-sm">
          <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
          LIVE
        </div>
        {currentFrame && (
          <div className="absolute bottom-2 left-2 bg-black/60 px-2 py-1 rounded text-xs">
            Frame: {currentFrame.frame_number} |
            Players: {currentFrame.players.length} |
            Ball: {currentFrame.ball ? 'Detected' : 'Not found'}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="relative h-full bg-black">
      {/* Video element */}
      <video
        ref={videoRef}
        src={videoUrl || undefined}
        className="w-full h-full object-contain"
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        onDurationChange={(e) => setDuration(e.currentTarget.duration)}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />

      {/* Overlay canvas for annotations */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        width={1920}
        height={1080}
      />

      {/* Controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
        {/* Progress bar */}
        <div className="mb-2">
          <input
            type="range"
            min={0}
            max={duration || 100}
            value={currentTime}
            onChange={(e) => {
              if (videoRef.current) {
                videoRef.current.currentTime = parseFloat(e.target.value);
              }
            }}
            className="w-full h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleSeek(-10)}
              className="p-2 hover:bg-white/10 rounded"
            >
              <SkipBack className="w-4 h-4" />
            </button>

            <button
              onClick={handlePlayPause}
              className="p-2 hover:bg-white/10 rounded"
            >
              {isPlaying ? (
                <Pause className="w-5 h-5" />
              ) : (
                <Play className="w-5 h-5" />
              )}
            </button>

            <button
              onClick={() => handleSeek(10)}
              className="p-2 hover:bg-white/10 rounded"
            >
              <SkipForward className="w-4 h-4" />
            </button>

            <span className="text-sm text-slate-300 ml-2">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <button className="p-2 hover:bg-white/10 rounded">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Frame info overlay */}
      {currentFrame && (
        <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-xs">
          Frame: {currentFrame.frame_number} |
          Home: {currentFrame.home_players} |
          Away: {currentFrame.away_players}
        </div>
      )}
    </div>
  );
}
