import { useRef, useEffect, useState, useCallback } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Save,
  Trash2,
  Users,
  CircleDot,
  Download,
  Upload,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Check,
  X,
  Eye,
  EyeOff,
  Layers,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import clsx from 'clsx';
import type { TeamSide, BoundingBox } from '../types';

// Annotation types
interface Annotation {
  id: string;
  class_name: 'player' | 'ball' | 'goalkeeper' | 'referee';
  bbox: BoundingBox;
  team?: TeamSide;
  track_id?: number;
  is_selected?: boolean;
}

interface FrameData {
  frame_id: string;
  video_id: string;
  frame_number: number;
  timestamp_ms: number;
  image_url: string;
  annotations: Annotation[];
  is_annotated: boolean;
}

interface AnnotationUIProps {
  videoId?: string;
  onExport?: () => void;
}

type DrawingMode = 'select' | 'player' | 'ball' | 'goalkeeper' | 'referee';

const CLASS_COLORS: Record<string, string> = {
  player: '#3b82f6',      // Blue
  ball: '#fbbf24',        // Yellow
  goalkeeper: '#22c55e',  // Green
  referee: '#f97316',     // Orange
};

const TEAM_COLORS: Record<TeamSide, string> = {
  home: '#3b82f6',   // Blue
  away: '#ef4444',   // Red
  unknown: '#888888', // Gray
};

export function AnnotationUI({ videoId, onExport }: AnnotationUIProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Frame management
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Annotation state
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null);
  const [drawingMode, setDrawingMode] = useState<DrawingMode>('select');
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawEnd, setDrawEnd] = useState<{ x: number; y: number } | null>(null);

  // View controls
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Stats
  const [stats, setStats] = useState<{
    total_frames: number;
    annotated_frames: number;
    total_annotations: number;
  } | null>(null);

  const currentFrame = frames[currentFrameIndex];

  // Fetch frames list
  useEffect(() => {
    fetchFrames();
    fetchStats();
  }, [videoId]);

  // Load annotations when frame changes
  useEffect(() => {
    if (currentFrame) {
      loadFrameAnnotations(currentFrame.frame_id);
    }
  }, [currentFrameIndex, frames]);

  // Draw canvas whenever relevant state changes
  useEffect(() => {
    drawCanvas();
  }, [annotations, selectedAnnotation, showAnnotations, zoom, pan, drawStart, drawEnd, isDrawing]);

  const fetchFrames = async () => {
    setLoading(true);
    try {
      const url = videoId
        ? `/api/training/frames?video_id=${videoId}`
        : '/api/training/frames';
      const response = await fetch(url);
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

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/training/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const loadFrameAnnotations = async (frameId: string) => {
    try {
      const response = await fetch(`/api/training/frame/${frameId}`);
      if (response.ok) {
        const data = await response.json();
        setAnnotations(data.annotations || []);
        loadFrameImage(data.image_url || `/api/training/frame/${frameId}/image`);
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
      const response = await fetch(
        `/api/training/frame/${currentFrame.frame_id}/annotations`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(annotations),
        }
      );

      if (response.ok) {
        // Update frame as annotated
        setFrames((prev) =>
          prev.map((f, i) =>
            i === currentFrameIndex ? { ...f, is_annotated: true } : f
          )
        );
        fetchStats();
      }
    } catch (error) {
      console.error('Failed to save annotations:', error);
    } finally {
      setSaving(false);
    }
  };

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply zoom and pan transformations
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // Draw image
    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0, canvas.width / zoom, canvas.height / zoom);
    } else {
      // Draw placeholder
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width / zoom, canvas.height / zoom);
      ctx.fillStyle = '#64748b';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No image loaded', canvas.width / zoom / 2, canvas.height / zoom / 2);
    }

    // Draw annotations
    if (showAnnotations) {
      annotations.forEach((ann) => {
        const isSelected = ann.id === selectedAnnotation;
        drawAnnotation(ctx, ann, isSelected);
      });
    }

    // Draw current drawing box
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

    ctx.restore();
  }, [annotations, selectedAnnotation, showAnnotations, zoom, pan, drawStart, drawEnd, isDrawing, drawingMode]);

  const drawAnnotation = (
    ctx: CanvasRenderingContext2D,
    ann: Annotation,
    isSelected: boolean
  ) => {
    const { bbox, class_name, team, track_id } = ann;

    // Determine color
    let color = CLASS_COLORS[class_name] || '#888888';
    if (class_name === 'player' && team) {
      color = TEAM_COLORS[team];
    }

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    // Draw selection handles if selected
    if (isSelected) {
      const handleSize = 8;
      ctx.fillStyle = color;

      // Corner handles
      const corners = [
        { x: bbox.x1, y: bbox.y1 },
        { x: bbox.x2, y: bbox.y1 },
        { x: bbox.x1, y: bbox.y2 },
        { x: bbox.x2, y: bbox.y2 },
      ];

      corners.forEach(({ x, y }) => {
        ctx.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize);
      });
    }

    // Draw label
    const label = track_id ? `${class_name} #${track_id}` : class_name;
    ctx.fillStyle = color;
    ctx.fillRect(bbox.x1, bbox.y1 - 20, Math.max(label.length * 7, 60), 20);
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    ctx.fillText(label, bbox.x1 + 4, bbox.y1 - 6);

    // Draw team indicator for players
    if (team && team !== 'unknown') {
      ctx.fillStyle = TEAM_COLORS[team];
      ctx.beginPath();
      ctx.arc(bbox.x2 - 8, bbox.y1 + 8, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  };

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: ((e.clientX - rect.left) * scaleX - pan.x) / zoom,
      y: ((e.clientY - rect.top) * scaleY - pan.y) / zoom,
    };
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoordinates(e);

    if (drawingMode === 'select') {
      // Check if clicking on an annotation
      const clicked = annotations.find((ann) => {
        const { bbox } = ann;
        return (
          coords.x >= bbox.x1 &&
          coords.x <= bbox.x2 &&
          coords.y >= bbox.y1 &&
          coords.y <= bbox.y2
        );
      });

      setSelectedAnnotation(clicked?.id || null);
    } else {
      // Start drawing new annotation
      setIsDrawing(true);
      setDrawStart(coords);
      setDrawEnd(coords);
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDrawing) {
      setDrawEnd(getCanvasCoordinates(e));
    }
  };

  const handleCanvasMouseUp = () => {
    if (isDrawing && drawStart && drawEnd) {
      const x1 = Math.min(drawStart.x, drawEnd.x);
      const y1 = Math.min(drawStart.y, drawEnd.y);
      const x2 = Math.max(drawStart.x, drawEnd.x);
      const y2 = Math.max(drawStart.y, drawEnd.y);

      // Only create if box is big enough
      if (x2 - x1 > 10 && y2 - y1 > 10) {
        const newAnnotation: Annotation = {
          id: `ann_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          class_name: drawingMode === 'select' ? 'player' : drawingMode,
          bbox: { x1, y1, x2, y2, confidence: 1.0 },
          team: drawingMode === 'player' ? 'unknown' : undefined,
        };

        setAnnotations((prev) => [...prev, newAnnotation]);
        setSelectedAnnotation(newAnnotation.id);
      }
    }

    setIsDrawing(false);
    setDrawStart(null);
    setDrawEnd(null);
  };

  const deleteSelectedAnnotation = () => {
    if (selectedAnnotation) {
      setAnnotations((prev) => prev.filter((a) => a.id !== selectedAnnotation));
      setSelectedAnnotation(null);
    }
  };

  const updateSelectedAnnotation = (updates: Partial<Annotation>) => {
    if (!selectedAnnotation) return;

    setAnnotations((prev) =>
      prev.map((a) => (a.id === selectedAnnotation ? { ...a, ...updates } : a))
    );
  };

  const navigateFrame = (direction: number) => {
    const newIndex = currentFrameIndex + direction;
    if (newIndex >= 0 && newIndex < frames.length) {
      setCurrentFrameIndex(newIndex);
      setSelectedAnnotation(null);
    }
  };

  const handleExportYOLO = async () => {
    try {
      window.open('/api/training/export/yolo', '_blank');
      onExport?.();
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const selectedAnn = annotations.find((a) => a.id === selectedAnnotation);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
        <span className="ml-2 text-slate-400">Loading frames...</span>
      </div>
    );
  }

  if (frames.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-slate-900 p-8">
        <Layers className="w-16 h-16 text-slate-600 mb-4" />
        <h3 className="text-xl font-semibold text-slate-300 mb-2">No Training Frames</h3>
        <p className="text-slate-500 text-center mb-4">
          Process a video first to capture frames for annotation.
          <br />
          Frames are automatically captured during video analysis.
        </p>
        <button
          onClick={fetchFrames}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-slate-900">
      {/* Top toolbar */}
      <div className="flex items-center justify-between p-2 bg-slate-800 border-b border-slate-700">
        {/* Drawing tools */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setDrawingMode('select')}
            className={clsx(
              'p-2 rounded',
              drawingMode === 'select'
                ? 'bg-blue-600 text-white'
                : 'hover:bg-slate-700 text-slate-400'
            )}
            title="Select (S)"
          >
            <Crosshair className="w-5 h-5" />
          </button>

          <div className="w-px h-6 bg-slate-600 mx-1" />

          <button
            onClick={() => setDrawingMode('player')}
            className={clsx(
              'p-2 rounded flex items-center gap-1',
              drawingMode === 'player'
                ? 'bg-blue-600 text-white'
                : 'hover:bg-slate-700 text-slate-400'
            )}
            title="Draw Player (P)"
          >
            <Users className="w-5 h-5" />
            <span className="text-xs">Player</span>
          </button>

          <button
            onClick={() => setDrawingMode('ball')}
            className={clsx(
              'p-2 rounded flex items-center gap-1',
              drawingMode === 'ball'
                ? 'bg-yellow-600 text-white'
                : 'hover:bg-slate-700 text-slate-400'
            )}
            title="Draw Ball (B)"
          >
            <CircleDot className="w-5 h-5" />
            <span className="text-xs">Ball</span>
          </button>

          <button
            onClick={() => setDrawingMode('goalkeeper')}
            className={clsx(
              'p-2 rounded flex items-center gap-1',
              drawingMode === 'goalkeeper'
                ? 'bg-green-600 text-white'
                : 'hover:bg-slate-700 text-slate-400'
            )}
            title="Draw Goalkeeper (G)"
          >
            <Users className="w-5 h-5" />
            <span className="text-xs">GK</span>
          </button>

          <button
            onClick={() => setDrawingMode('referee')}
            className={clsx(
              'p-2 rounded flex items-center gap-1',
              drawingMode === 'referee'
                ? 'bg-orange-600 text-white'
                : 'hover:bg-slate-700 text-slate-400'
            )}
            title="Draw Referee (R)"
          >
            <Users className="w-5 h-5" />
            <span className="text-xs">Ref</span>
          </button>
        </div>

        {/* View controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowAnnotations(!showAnnotations)}
            className={clsx(
              'p-2 rounded',
              showAnnotations ? 'text-blue-400' : 'text-slate-500'
            )}
            title="Toggle annotations"
          >
            {showAnnotations ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
          </button>

          <button
            onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
            className="p-2 rounded hover:bg-slate-700 text-slate-400"
            title="Zoom in"
          >
            <ZoomIn className="w-5 h-5" />
          </button>

          <span className="text-sm text-slate-500 w-12 text-center">
            {Math.round(zoom * 100)}%
          </span>

          <button
            onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
            className="p-2 rounded hover:bg-slate-700 text-slate-400"
            title="Zoom out"
          >
            <ZoomOut className="w-5 h-5" />
          </button>

          <button
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
            className="p-2 rounded hover:bg-slate-700 text-slate-400"
            title="Reset view"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={saveAnnotations}
            disabled={saving}
            className={clsx(
              'px-3 py-1.5 rounded flex items-center gap-2',
              saving
                ? 'bg-slate-700 text-slate-500'
                : 'bg-green-600 hover:bg-green-700 text-white'
            )}
          >
            {saving ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Save
          </button>

          <button
            onClick={handleExportYOLO}
            className="px-3 py-1.5 rounded bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export YOLO
          </button>
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Canvas area */}
        <div ref={containerRef} className="flex-1 relative overflow-hidden">
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

          {/* Frame navigation overlay */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-black/60 rounded-lg px-4 py-2">
            <button
              onClick={() => navigateFrame(-1)}
              disabled={currentFrameIndex === 0}
              className="p-1 hover:bg-white/10 rounded disabled:opacity-50"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            <span className="text-sm">
              Frame {currentFrameIndex + 1} / {frames.length}
              {currentFrame?.is_annotated && (
                <Check className="w-4 h-4 inline ml-2 text-green-400" />
              )}
            </span>

            <button
              onClick={() => navigateFrame(1)}
              disabled={currentFrameIndex === frames.length - 1}
              className="p-1 hover:bg-white/10 rounded disabled:opacity-50"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          {/* Stats overlay */}
          {stats && (
            <div className="absolute top-2 left-2 bg-black/60 rounded px-3 py-2 text-xs">
              <div>Total Frames: {stats.total_frames}</div>
              <div>Annotated: {stats.annotated_frames}</div>
              <div>Total Annotations: {stats.total_annotations}</div>
            </div>
          )}
        </div>

        {/* Right sidebar - annotation properties */}
        <div className="w-64 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto">
          <h3 className="text-sm font-semibold text-slate-400 uppercase mb-4">
            Annotations ({annotations.length})
          </h3>

          {/* Annotation list */}
          <div className="space-y-2 mb-4">
            {annotations.map((ann) => (
              <div
                key={ann.id}
                onClick={() => setSelectedAnnotation(ann.id)}
                className={clsx(
                  'p-2 rounded cursor-pointer flex items-center justify-between',
                  ann.id === selectedAnnotation
                    ? 'bg-blue-600/30 border border-blue-500'
                    : 'bg-slate-700/50 hover:bg-slate-700'
                )}
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{
                      backgroundColor:
                        ann.class_name === 'player' && ann.team
                          ? TEAM_COLORS[ann.team]
                          : CLASS_COLORS[ann.class_name],
                    }}
                  />
                  <span className="text-sm capitalize">{ann.class_name}</span>
                  {ann.track_id && (
                    <span className="text-xs text-slate-500">#{ann.track_id}</span>
                  )}
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setAnnotations((prev) => prev.filter((a) => a.id !== ann.id));
                    if (selectedAnnotation === ann.id) setSelectedAnnotation(null);
                  }}
                  className="p-1 hover:bg-red-500/30 rounded text-red-400"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>

          {/* Selected annotation properties */}
          {selectedAnn && (
            <div className="border-t border-slate-700 pt-4">
              <h4 className="text-sm font-semibold text-slate-400 mb-3">Properties</h4>

              {/* Class selection */}
              <div className="mb-3">
                <label className="text-xs text-slate-500 block mb-1">Class</label>
                <select
                  value={selectedAnn.class_name}
                  onChange={(e) =>
                    updateSelectedAnnotation({
                      class_name: e.target.value as Annotation['class_name'],
                    })
                  }
                  className="w-full bg-slate-700 rounded px-2 py-1.5 text-sm"
                >
                  <option value="player">Player</option>
                  <option value="ball">Ball</option>
                  <option value="goalkeeper">Goalkeeper</option>
                  <option value="referee">Referee</option>
                </select>
              </div>

              {/* Team selection (for players) */}
              {(selectedAnn.class_name === 'player' ||
                selectedAnn.class_name === 'goalkeeper') && (
                <div className="mb-3">
                  <label className="text-xs text-slate-500 block mb-1">Team</label>
                  <div className="flex gap-2">
                    {(['home', 'away', 'unknown'] as TeamSide[]).map((team) => (
                      <button
                        key={team}
                        onClick={() => updateSelectedAnnotation({ team })}
                        className={clsx(
                          'flex-1 py-1.5 rounded text-sm capitalize',
                          selectedAnn.team === team
                            ? 'ring-2 ring-white'
                            : 'opacity-70 hover:opacity-100'
                        )}
                        style={{ backgroundColor: TEAM_COLORS[team] }}
                      >
                        {team}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Track ID */}
              <div className="mb-3">
                <label className="text-xs text-slate-500 block mb-1">Track ID</label>
                <input
                  type="number"
                  value={selectedAnn.track_id || ''}
                  onChange={(e) =>
                    updateSelectedAnnotation({
                      track_id: e.target.value ? parseInt(e.target.value) : undefined,
                    })
                  }
                  placeholder="Optional"
                  className="w-full bg-slate-700 rounded px-2 py-1.5 text-sm"
                />
              </div>

              {/* Bounding box info */}
              <div className="mb-3">
                <label className="text-xs text-slate-500 block mb-1">Bounding Box</label>
                <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
                  <div>x1: {Math.round(selectedAnn.bbox.x1)}</div>
                  <div>y1: {Math.round(selectedAnn.bbox.y1)}</div>
                  <div>x2: {Math.round(selectedAnn.bbox.x2)}</div>
                  <div>y2: {Math.round(selectedAnn.bbox.y2)}</div>
                </div>
              </div>

              {/* Delete button */}
              <button
                onClick={deleteSelectedAnnotation}
                className="w-full py-2 bg-red-600/30 hover:bg-red-600/50 text-red-400 rounded flex items-center justify-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete Annotation
              </button>
            </div>
          )}

          {/* Keyboard shortcuts */}
          <div className="border-t border-slate-700 pt-4 mt-4">
            <h4 className="text-xs font-semibold text-slate-500 mb-2">Shortcuts</h4>
            <div className="text-xs text-slate-500 space-y-1">
              <div><kbd className="px-1 bg-slate-700 rounded">S</kbd> Select</div>
              <div><kbd className="px-1 bg-slate-700 rounded">P</kbd> Player</div>
              <div><kbd className="px-1 bg-slate-700 rounded">B</kbd> Ball</div>
              <div><kbd className="px-1 bg-slate-700 rounded">G</kbd> Goalkeeper</div>
              <div><kbd className="px-1 bg-slate-700 rounded">R</kbd> Referee</div>
              <div><kbd className="px-1 bg-slate-700 rounded">Del</kbd> Delete</div>
              <div><kbd className="px-1 bg-slate-700 rounded">←/→</kbd> Navigate</div>
            </div>
          </div>
        </div>
      </div>

      {/* Keyboard event handler */}
      <KeyboardHandler
        onKeyDown={(key) => {
          switch (key) {
            case 's':
              setDrawingMode('select');
              break;
            case 'p':
              setDrawingMode('player');
              break;
            case 'b':
              setDrawingMode('ball');
              break;
            case 'g':
              setDrawingMode('goalkeeper');
              break;
            case 'r':
              setDrawingMode('referee');
              break;
            case 'Delete':
            case 'Backspace':
              deleteSelectedAnnotation();
              break;
            case 'ArrowLeft':
              navigateFrame(-1);
              break;
            case 'ArrowRight':
              navigateFrame(1);
              break;
          }
        }}
      />
    </div>
  );
}

// Keyboard handler component
function KeyboardHandler({
  onKeyDown,
}: {
  onKeyDown: (key: string) => void;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }
      onKeyDown(e.key);
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onKeyDown]);

  return null;
}

export default AnnotationUI;
