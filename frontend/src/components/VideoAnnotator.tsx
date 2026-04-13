import { useEffect, useRef, useState, useCallback } from 'react';
import { api } from '../lib/api';
import {
  Canvas as FabricCanvas,
  Circle,
  Line,
  Triangle,
  Group,
  Textbox,
  Rect,
  Path,
  PencilBrush,
  Gradient,
  FabricObject,
} from 'fabric';
import {
  MousePointer,
  MoveUpRight,
  CircleIcon,
  PenTool,
  Square,
  Type,
  Target,
  Highlighter,
  Hash,
  Minus,
  Route,
  ScanEye,
  Shirt,
  GitBranch,
  Undo2,
  Trash2,
  Download,
  X,
  Sparkles,
  Loader2,
} from 'lucide-react';

type Tool =
  | 'select' | 'arrow' | 'circle' | 'freehand' | 'rect' | 'text' | 'highlight'
  | 'highlighter' | 'numbered' | 'offside' | 'curved' | 'spotlight' | 'shirt' | 'passandmove';

const COLORS = [
  { name: 'Green', value: '#10b981' },
  { name: 'Red', value: '#ef4444' },
  { name: 'Blue', value: '#3b82f6' },
  { name: 'Yellow', value: '#eab308' },
  { name: 'White', value: '#ffffff' },
];

const STROKE_WIDTH = 3;

interface Annotation {
  timestamp_sec: number;
  type: string;
  description: string;
  elements: {
    shape: string;
    x: number; y: number;
    x2?: number; y2?: number;
    label?: string;
    color?: string;
    hex_color?: string;
  }[];
}

interface VideoAnnotatorProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  videoRef: React.RefObject<any>;
  matchId: string;
  isActive: boolean;
  onClose: () => void;
}

// ── Shape helpers ──

function makeArrow(x1: number, y1: number, x2: number, y2: number, color: string) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const line = new Line([x1, y1, x2, y2], {
    stroke: color, strokeWidth: STROKE_WIDTH, strokeLineCap: 'round',
  });
  const tip = new Triangle({
    left: x2, top: y2, width: 14, height: 14, fill: color,
    angle: (angle * 180) / Math.PI + 90,
    originX: 'center', originY: 'center',
  });
  return new Group([line, tip]);
}

function makeCurvedArrow(x1: number, y1: number, x2: number, y2: number, color: string) {
  // Quadratic bezier with control point offset perpendicular to the midpoint
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  // Perpendicular offset (30% of length, curving "up-left" relative to direction)
  const offset = len * 0.3;
  const cx = mx - (dy / len) * offset;
  const cy = my + (dx / len) * offset;

  const pathStr = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
  const curve = new Path(pathStr, {
    fill: 'transparent', stroke: color, strokeWidth: STROKE_WIDTH, strokeLineCap: 'round',
  });

  // Arrowhead: tangent at t=1 of Q(P0,P1,P2) is direction P1→P2
  const headAngle = Math.atan2(y2 - cy, x2 - cx);
  const tip = new Triangle({
    left: x2, top: y2, width: 14, height: 14, fill: color,
    angle: (headAngle * 180) / Math.PI + 90,
    originX: 'center', originY: 'center',
  });
  return new Group([curve, tip]);
}

function makePassAndMove(x1: number, y1: number, x2: number, y2: number, color: string) {
  // Solid arrow: start → end (the pass)
  const pass = makeArrow(x1, y1, x2, y2, color);

  // Dashed curved arrow: start → offset position (the run)
  // Run goes perpendicular to the pass direction at ~60% length
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  const runEnd = {
    x: x1 + dx * 0.5 - (dy / len) * len * 0.4,
    y: y1 + dy * 0.5 + (dx / len) * len * 0.4,
  };
  const runMx = (x1 + runEnd.x) / 2;
  const runMy = (y1 + runEnd.y) / 2;
  const rdx = runEnd.x - x1;
  const rdy = runEnd.y - y1;
  const rLen = Math.hypot(rdx, rdy);
  const rOffset = rLen * 0.25;
  const rcx = runMx - (rdy / rLen) * rOffset;
  const rcy = runMy + (rdx / rLen) * rOffset;

  const pathStr = `M ${x1} ${y1} Q ${rcx} ${rcy} ${runEnd.x} ${runEnd.y}`;
  const runPath = new Path(pathStr, {
    fill: 'transparent', stroke: color, strokeWidth: STROKE_WIDTH - 1,
    strokeLineCap: 'round', strokeDashArray: [8, 5],
  });
  const runAngle = Math.atan2(runEnd.y - rcy, runEnd.x - rcx);
  const runTip = new Triangle({
    left: runEnd.x, top: runEnd.y, width: 11, height: 11, fill: color,
    angle: (runAngle * 180) / Math.PI + 90,
    originX: 'center', originY: 'center',
  });

  return new Group([pass, runPath, runTip]);
}

function makeNumberedMarker(x: number, y: number, num: number, color: string) {
  const r = 14;
  const bg = new Circle({
    originX: 'center', originY: 'center',
    radius: r, fill: color, stroke: '#fff', strokeWidth: 2,
  });
  const label = new Textbox(String(num), {
    originX: 'center', originY: 'center',
    fontSize: 14, fill: '#fff', fontFamily: 'Inter, sans-serif',
    fontWeight: '700', width: r * 2, textAlign: 'center',
  });
  const g = new Group([bg, label], { left: x, top: y, originX: 'center', originY: 'center' });
  return g;
}

function makeShirtMarker(x: number, y: number, color: string) {
  const r = 16;
  const bg = new Circle({
    originX: 'center', originY: 'center',
    radius: r, fill: color, stroke: '#fff', strokeWidth: 2,
  });
  const label = new Textbox('?', {
    originX: 'center', originY: 'center',
    fontSize: 15, fill: '#fff', fontFamily: 'Inter, sans-serif',
    fontWeight: '700', width: r * 2, textAlign: 'center',
    editable: true,
  });
  return new Group([bg, label], { left: x, top: y, originX: 'center', originY: 'center' });
}

function makeOffsideLine(y: number, width: number, color: string) {
  const line = new Line([0, y, width, y], {
    stroke: color, strokeWidth: 2, strokeDashArray: [10, 6],
  });
  const label = new Textbox('LINE', {
    left: 8, top: y - 18,
    fontSize: 11, fill: color, fontFamily: 'Inter, sans-serif',
    fontWeight: '600', width: 50, editable: true,
  });
  return new Group([line, label]);
}

function makeSpotlight(x: number, y: number, canvasW: number, canvasH: number) {
  const r = 60;
  // Full dark overlay with radial gradient: transparent center → dark edges
  const overlay = new Rect({
    left: 0, top: 0, width: canvasW, height: canvasH,
    fill: new Gradient({
      type: 'radial',
      coords: { x1: x, y1: y, r1: r, x2: x, y2: y, r2: Math.max(canvasW, canvasH) * 0.8 },
      colorStops: [
        { offset: 0, color: 'rgba(0,0,0,0)' },
        { offset: 0.25, color: 'rgba(0,0,0,0)' },
        { offset: 0.5, color: 'rgba(0,0,0,0.45)' },
        { offset: 1, color: 'rgba(0,0,0,0.7)' },
      ],
    }),
  });
  // Bright ring to show the spotlight boundary
  const ring = new Circle({
    left: x - r, top: y - r, radius: r,
    fill: 'transparent', stroke: 'rgba(255,255,255,0.3)', strokeWidth: 2,
  });
  return new Group([overlay, ring]);
}

// ── Component ──

/** Render AI-generated annotations onto the Fabric canvas. */
function renderAnnotations(fc: FabricCanvas, annotations: Annotation[], canvasW: number, canvasH: number) {
  for (const ann of annotations) {
    for (const el of ann.elements) {
      const x = el.x * canvasW;
      const y = el.y * canvasH;
      const c = el.hex_color || '#10b981';

      if ((el.shape === 'arrow' || el.shape === 'curved_arrow') && el.x2 !== undefined && el.y2 !== undefined) {
        const x2 = el.x2 * canvasW;
        const y2 = el.y2 * canvasH;
        const obj = el.shape === 'curved_arrow'
          ? makeCurvedArrow(x, y, x2, y2, c)
          : makeArrow(x, y, x2, y2, c);
        fc.add(obj);
        // Add label near the midpoint if present
        if (el.label) {
          fc.add(new Textbox(el.label, {
            left: (x + x2) / 2, top: (y + y2) / 2 - 18,
            fontSize: 12, fill: c, fontFamily: 'Inter, sans-serif',
            fontWeight: '600', width: 120, backgroundColor: 'rgba(0,0,0,0.5)',
          }));
        }
      } else if (el.shape === 'circle') {
        fc.add(new Circle({
          left: x - 25, top: y - 25, radius: 25,
          fill: 'transparent', stroke: c, strokeWidth: STROKE_WIDTH,
        }));
        if (el.label) {
          fc.add(new Textbox(el.label, {
            left: x - 40, top: y + 28, fontSize: 11, fill: c,
            fontFamily: 'Inter, sans-serif', fontWeight: '600', width: 80,
            textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.5)',
          }));
        }
      } else if (el.shape === 'rect') {
        fc.add(new Rect({
          left: x, top: y,
          width: (el.x2 !== undefined ? el.x2 * canvasW - x : canvasW * 0.15),
          height: (el.y2 !== undefined ? el.y2 * canvasH - y : canvasH * 0.1),
          fill: `${c}22`, stroke: c, strokeWidth: STROKE_WIDTH, rx: 4, ry: 4,
        }));
        if (el.label) {
          fc.add(new Textbox(el.label, {
            left: x + 4, top: y + 4, fontSize: 11, fill: c,
            fontFamily: 'Inter, sans-serif', fontWeight: '600', width: 100,
            backgroundColor: 'rgba(0,0,0,0.5)',
          }));
        }
      } else if (el.shape === 'text') {
        fc.add(new Textbox(el.label || ann.description || '', {
          left: x, top: y, fontSize: 14, fill: c,
          fontFamily: 'Inter, sans-serif', fontWeight: '600', width: 180,
          backgroundColor: 'rgba(0,0,0,0.6)',
        }));
      } else if (el.shape === 'marker') {
        fc.add(makeNumberedMarker(x, y, annotations.indexOf(ann) + 1, c));
      }
    }
  }
}

export default function VideoAnnotator({ containerRef, videoRef, matchId, isActive, onClose }: VideoAnnotatorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<FabricCanvas | null>(null);
  const [tool, setTool] = useState<Tool>('arrow');
  const [color, setColor] = useState(COLORS[0].value);
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const previewObj = useRef<FabricObject | null>(null);
  const markerCount = useRef(0);
  const [aiLoading, setAiLoading] = useState(false);

  // --- Lifecycle ---
  useEffect(() => {
    if (!isActive || !canvasRef.current || !containerRef.current) return;
    const { width, height } = containerRef.current.getBoundingClientRect();
    canvasRef.current.width = width;
    canvasRef.current.height = height;
    const fc = new FabricCanvas(canvasRef.current, { width, height, selection: false, backgroundColor: 'transparent' });
    fabricRef.current = fc;
    markerCount.current = 0;
    return () => { fc.dispose(); fabricRef.current = null; };
  }, [isActive, containerRef]);

  // --- Tool switching ---
  useEffect(() => {
    const fc = fabricRef.current;
    if (!fc) return;
    const isFreeMode = tool === 'freehand' || tool === 'highlighter';
    fc.isDrawingMode = isFreeMode;
    if (isFreeMode) {
      const brush = new PencilBrush(fc);
      if (tool === 'highlighter') {
        brush.color = color + '55'; // semi-transparent
        brush.width = 22;
      } else {
        brush.color = color;
        brush.width = STROKE_WIDTH;
      }
      fc.freeDrawingBrush = brush;
    }
    fc.selection = tool === 'select';
    fc.defaultCursor = tool === 'select' ? 'default' : 'crosshair';
    fc.forEachObject((o) => { o.selectable = tool === 'select'; o.evented = tool === 'select'; });
  }, [tool, color]);

  // --- Mouse handlers ---
  useEffect(() => {
    const fc = fabricRef.current;
    if (!fc || tool === 'freehand' || tool === 'highlighter' || tool === 'select') return;

    const canvasW = fc.getWidth();
    const canvasH = fc.getHeight();

    const onDown = (e: any) => {
      const p = fc.getScenePoint(e.e);
      drawStart.current = { x: p.x, y: p.y };

      // Click-to-place tools (no drag)
      if (tool === 'text') {
        const tb = new Textbox('Text', {
          left: p.x, top: p.y, fontSize: 18, fill: color,
          fontFamily: 'Inter, sans-serif', fontWeight: '600', width: 120, editable: true,
        });
        fc.add(tb); fc.setActiveObject(tb); tb.enterEditing();
        drawStart.current = null; return;
      }
      if (tool === 'highlight') {
        fc.add(new Circle({
          left: p.x - 20, top: p.y - 20, radius: 20,
          fill: 'transparent', stroke: color, strokeWidth: STROKE_WIDTH + 1, strokeDashArray: [6, 3],
        }));
        drawStart.current = null; return;
      }
      if (tool === 'numbered') {
        markerCount.current += 1;
        fc.add(makeNumberedMarker(p.x, p.y, markerCount.current, color));
        drawStart.current = null; return;
      }
      if (tool === 'shirt') {
        const marker = makeShirtMarker(p.x, p.y, color);
        fc.add(marker);
        drawStart.current = null; return;
      }
      if (tool === 'offside') {
        fc.add(makeOffsideLine(p.y, canvasW, color));
        drawStart.current = null; return;
      }
      if (tool === 'spotlight') {
        fc.add(makeSpotlight(p.x, p.y, canvasW, canvasH));
        drawStart.current = null; return;
      }
    };

    const onMove = (e: any) => {
      if (!drawStart.current) return;
      const p = fc.getScenePoint(e.e);
      const { x: sx, y: sy } = drawStart.current;

      if (previewObj.current) { fc.remove(previewObj.current); previewObj.current = null; }

      let obj: FabricObject | null = null;
      if (tool === 'arrow') obj = makeArrow(sx, sy, p.x, p.y, color);
      else if (tool === 'curved') obj = makeCurvedArrow(sx, sy, p.x, p.y, color);
      else if (tool === 'passandmove') obj = makePassAndMove(sx, sy, p.x, p.y, color);
      else if (tool === 'circle') {
        const r = Math.hypot(p.x - sx, p.y - sy);
        obj = new Circle({ left: sx - r, top: sy - r, radius: r, fill: 'transparent', stroke: color, strokeWidth: STROKE_WIDTH });
      } else if (tool === 'rect') {
        obj = new Rect({
          left: Math.min(sx, p.x), top: Math.min(sy, p.y),
          width: Math.abs(p.x - sx), height: Math.abs(p.y - sy),
          fill: `${color}22`, stroke: color, strokeWidth: STROKE_WIDTH, rx: 4, ry: 4,
        });
      }

      if (obj) { obj.selectable = false; obj.evented = false; fc.add(obj); previewObj.current = obj; }
    };

    const onUp = () => {
      if (previewObj.current) { previewObj.current.selectable = true; previewObj.current.evented = true; previewObj.current = null; }
      drawStart.current = null;
    };

    fc.on('mouse:down', onDown);
    fc.on('mouse:move', onMove);
    fc.on('mouse:up', onUp);
    return () => { fc.off('mouse:down', onDown); fc.off('mouse:move', onMove); fc.off('mouse:up', onUp); };
  }, [tool, color]);

  // --- Actions ---
  const handleUndo = useCallback(() => {
    const fc = fabricRef.current; if (!fc) return;
    const objs = fc.getObjects();
    if (objs.length) fc.remove(objs[objs.length - 1]);
  }, []);

  const handleClear = useCallback(() => { fabricRef.current?.clear(); markerCount.current = 0; }, []);

  const handleDelete = useCallback(() => {
    const fc = fabricRef.current; if (!fc) return;
    fc.getActiveObjects().forEach((o) => fc.remove(o));
    fc.discardActiveObject();
  }, []);

  const handleSave = useCallback(() => {
    const fc = fabricRef.current;
    const container = containerRef.current;
    if (!fc || !container) return;
    const el = videoRef.current;
    const media: HTMLVideoElement | null = el?.media?.nativeEl || el;
    const { width, height } = container.getBoundingClientRect();
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = width; exportCanvas.height = height;
    const ctx = exportCanvas.getContext('2d')!;
    if (media && media.videoWidth) ctx.drawImage(media, 0, 0, width, height);
    else { ctx.fillStyle = '#18181b'; ctx.fillRect(0, 0, width, height); }
    const annotationURL = fc.toDataURL({ format: 'png' });
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      const link = document.createElement('a');
      link.download = `annotation-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.png`;
      link.href = exportCanvas.toDataURL('image/png');
      link.click();
    };
    img.src = annotationURL;
  }, [containerRef, videoRef]);

  const [aiCoachingPoint, setAiCoachingPoint] = useState<{ point: string; detail: string } | null>(null);

  const handleAiAnnotate = useCallback(async () => {
    const fc = fabricRef.current;
    const container = containerRef.current;
    if (!fc || !container) return;

    const el = videoRef.current;
    const media: HTMLVideoElement | null = el?.media?.nativeEl || el;
    const currentTime = media?.currentTime || 0;

    // Capture the actual video frame as base64 PNG
    const { width, height } = container.getBoundingClientRect();
    let frameBase64 = '';
    if (media && media.videoWidth) {
      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = media.videoWidth;
      captureCanvas.height = media.videoHeight;
      const ctx = captureCanvas.getContext('2d')!;
      ctx.drawImage(media, 0, 0);
      frameBase64 = captureCanvas.toDataURL('image/png');
    }
    if (!frameBase64) {
      console.error('Could not capture video frame');
      return;
    }

    setAiLoading(true);
    setAiCoachingPoint(null);
    try {
      const data = await api.autoAnnotate({
        match_id: matchId,
        timestamp_sec: currentTime,
        frame_base64: frameBase64,
      });
      renderAnnotations(fc, data.annotations, width, height);
      if (data.coaching_point) {
        setAiCoachingPoint({ point: data.coaching_point, detail: data.detail || '' });
      }
    } catch (err: any) {
      console.error('AI annotation failed:', err);
    } finally {
      setAiLoading(false);
    }
  }, [matchId, containerRef, videoRef]);

  if (!isActive) return null;

  // ── Toolbar layout: group tools by category ──
  const drawTools: { id: Tool; icon: typeof MousePointer; label: string }[] = [
    { id: 'select', icon: MousePointer, label: 'Select' },
    { id: 'arrow', icon: MoveUpRight, label: 'Arrow' },
    { id: 'curved', icon: Route, label: 'Curved Arrow' },
    { id: 'passandmove', icon: GitBranch, label: 'Pass & Move' },
    { id: 'circle', icon: CircleIcon, label: 'Circle' },
    { id: 'rect', icon: Square, label: 'Zone' },
    { id: 'freehand', icon: PenTool, label: 'Draw' },
    { id: 'highlighter', icon: Highlighter, label: 'Highlighter' },
    { id: 'text', icon: Type, label: 'Text' },
    { id: 'offside', icon: Minus, label: 'Offside / Line' },
    { id: 'numbered', icon: Hash, label: 'Numbered Marker' },
    { id: 'shirt', icon: Shirt, label: 'Shirt Number' },
    { id: 'highlight', icon: Target, label: 'Player Ring' },
    { id: 'spotlight', icon: ScanEye, label: 'Spotlight' },
  ];

  return (
    <div className="absolute inset-0 z-30">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

      {/* Floating toolbar — two rows for the expanded tool set */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-zinc-950/90 backdrop-blur border border-zinc-700 rounded-xl px-2.5 py-2 shadow-2xl space-y-1.5">
        {/* Row 1: Draw tools */}
        <div className="flex items-center gap-0.5 flex-wrap justify-center">
          {drawTools.map((t) => (
            <button
              key={t.id}
              onClick={() => setTool(t.id)}
              title={t.label}
              className={`p-1.5 rounded-lg transition-colors ${
                tool === t.id
                  ? 'bg-emerald-500/30 text-emerald-300'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'
              }`}
            >
              <t.icon className="w-4 h-4" />
            </button>
          ))}

          <div className="w-px h-6 bg-zinc-700 mx-1" />

          {/* Colors */}
          {COLORS.map((c) => (
            <button
              key={c.value}
              onClick={() => setColor(c.value)}
              title={c.name}
              className={`w-5 h-5 rounded-full border-2 transition-transform ${
                color === c.value ? 'border-white scale-110' : 'border-zinc-600'
              }`}
              style={{ backgroundColor: c.value }}
            />
          ))}

          <div className="w-px h-6 bg-zinc-700 mx-1" />

          {/* Actions */}
          <button onClick={handleUndo} title="Undo" className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800">
            <Undo2 className="w-4 h-4" />
          </button>
          <button onClick={handleDelete} title="Delete selected" className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800">
            <Trash2 className="w-4 h-4" />
          </button>
          <button onClick={handleClear} title="Clear all" className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 text-xs font-bold">
            CLR
          </button>

          <div className="w-px h-6 bg-zinc-700 mx-1" />

          {/* AI Auto-Annotate */}
          <button
            onClick={handleAiAnnotate}
            disabled={aiLoading}
            title="AI auto-annotate this moment"
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium text-xs transition-colors bg-purple-500/20 border border-purple-500/30 text-purple-300 hover:bg-purple-500/30 disabled:opacity-50"
          >
            {aiLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            {aiLoading ? 'Analysing...' : 'AI Annotate'}
          </button>

          <div className="w-px h-6 bg-zinc-700 mx-1" />

          <button onClick={handleSave} title="Save as PNG" className="p-1.5 rounded-lg text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/20">
            <Download className="w-4 h-4" />
          </button>
          <button onClick={onClose} title="Exit annotation" className="p-1.5 rounded-lg text-red-400 hover:text-red-300 hover:bg-red-500/20">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* AI coaching point callout */}
      {aiCoachingPoint && (
        <div className="absolute bottom-14 left-1/2 -translate-x-1/2 max-w-md bg-zinc-950/95 backdrop-blur border border-purple-500/30 rounded-xl px-4 py-3 shadow-2xl">
          <div className="flex items-start gap-2">
            <Sparkles className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-purple-200">{aiCoachingPoint.point}</p>
              {aiCoachingPoint.detail && (
                <p className="text-xs text-zinc-400 mt-1 leading-relaxed">{aiCoachingPoint.detail}</p>
              )}
            </div>
            <button onClick={() => setAiCoachingPoint(null)} className="text-zinc-500 hover:text-zinc-300 flex-shrink-0">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Active tool label */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-xs text-zinc-500 bg-zinc-950/70 px-3 py-1 rounded-full">
        {drawTools.find((t) => t.id === tool)?.label} &middot; Del: remove &middot; Ctrl+Z: undo &middot; Esc: exit
      </div>
    </div>
  );
}
