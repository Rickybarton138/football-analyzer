# Manager Mentor: Battle Plan to Rival VEO & HUDL

> **Date**: 2026-03-08
> **Status**: ACTIVE — This is the master plan for building a product that rivals (and beats) VEO and HUDL
> **Supporting research**: See `VEO_DEEP_TECHNICAL_ANALYSIS.md`, `HUDL_TECHNICAL_ARCHITECTURE.md`, `PLAYER_DETECTION_DEEP_DIVE.md` in this directory

---

## The Opportunity

VEO ($80M funding, ~$72M revenue) and HUDL ($730M valuation, 18 acquisitions) dominate sports video analysis. But they are **vulnerable**:

| Weakness | VEO | HUDL | Our Advantage |
|----------|-----|------|---------------|
| **Processing time** | 24-48 hours | 18+ hours (IQ) | Real-time possible, <1hr for full analysis |
| **Team classification** | Manual input required | Human analysts verify | Automated HSV Hue + SigLIP |
| **Event detection** | Only 8 events auto-detected | Human taggers for most sports | AI detects passes, tackles, formations, xG |
| **Coaching intelligence** | Zero — just shows what happened | Zero | AI tells coaches **what to do next** |
| **Training recommendations** | None | None | AI generates drills + session plans |
| **Platform** | Camera locked to subscription | Sportscode macOS-only | Web-based, works with ANY video |
| **Pricing** | $2-3K/year + camera cost | $400-1,800/year | Grassroots-friendly pricing |
| **Hardware dependency** | Yes (camera required) | Yes (Focus camera) | Software-only — upload any video |
| **API access** | Closed (no new customers) | Limited (Wyscout only) | Open API from day one |

**Our unique value propositions:**
1. **Software-only** — no camera lock-in, works with VEO/HUDL/phone footage
2. **AI coaching intelligence** — not just analytics, but actionable recommendations
3. **Training focus** — AI-generated drills and session plans from match weaknesses
4. **Speed** — minutes, not days
5. **Price** — fraction of VEO/HUDL

---

## Architecture: How VEO & HUDL Actually Work

### VEO's Pipeline
```
Camera (dual 4K fisheye, 30fps)
  → Upload to cloud (1-12+ hours)
  → Dewarp fisheye distortion
  → Stitch panoramic
  → Player detection (unknown model, likely YOLO-variant)
  → Ball tracking
  → "Concentration of action" detection (patented)
  → Virtual camera crop (AI director)
  → Team classification (MANUAL user input required)
  → Event detection (8 types only: goals, corners, free kicks, etc.)
  → Analytics dashboard
  → 24-48 hours total end-to-end
```

### HUDL's Pipeline
```
Focus camera (1080p, 180°) or uploaded video
  → Upload to AWS S3
  → Frame extraction + distortion correction
  → Homography estimation (pitch mapping)
  → Object detection (DETR/Detection Transformer)
  → Multi-object tracking (30fps)
  → Team classification (color clustering + human QA)
  → Jersey number recognition (CNN, unreliable)
  → Event detection (mostly human-tagged via Assist)
  → StatsBomb data enrichment (3000+ events/game, ALL human-tagged)
  → Physical metrics (speed, distance from tracking)
  → 18+ hours for AI analysis
```

### Our Target Pipeline
```
Any video (VEO, HUDL, phone, broadcast)
  → Upload (web interface)
  → SAHI tiled detection (20+ players reliably)
  → ByteTrack/SportSORT tracking
  → Automated team classification (HSV Hue + SigLIP)
  → Pitch detection + homography
  → Event detection (passes, tackles, shots, formations, set pieces)
  → xG model
  → AI coaching analysis (Claude/GPT)
  → Training recommendations + session plans
  → Real-time results in <30 minutes
```

---

## What We Need to Fix (Priority Order)

### Phase 4A: Detection Pipeline (Current Focus)

**Goal**: Reliably detect 20+ players per frame with correct team assignment

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| SAHI tiled inference | +3-5 players/frame | Medium | TO DO |
| Fix jersey color extraction for tiny boxes (vectorize, upscale to 64x64) | Better color accuracy | Small | TO DO |
| Manual team color input via frontend | 95%+ team accuracy | Small | TO DO |
| a*b*-only LAB fallback (drop L channel) | Better dark jersey handling | Small | TO DO |
| Reduce SigLIP UMAP components (10→3) | Better embedding clustering | Tiny | TO DO |
| DBSCAN outlier filtering before KMeans | Removes referee colors from clustering | Small | TO DO |
| Adaptive grass color detection | Better grass filtering | Small | TO DO |
| GK heuristic (color outlier + position) | Identify 2 GKs/frame | Medium | TO DO |
| HSV Hue team clustering | Fixed dark jersey issue | Done | DONE |
| Event detection color bridge | Instant calibration | Done | DONE |
| ByteTrack frame gap reset | Fixed quick_preview | Done | DONE |
| Async model loading | Fixed event loop blocking | Done | DONE |

#### SAHI — The Biggest Win Available

SAHI (Slicing Aided Hyper Inference) is the single highest-impact improvement. Our current pipeline misses far-side players because they're only 20-40px tall — below YOLO's reliable detection zone. SAHI solves this by:

1. Slicing the 1920x1080 frame into overlapping tiles (e.g., 4x 640x640 tiles)
2. Running YOLO on each tile separately (players appear larger)
3. Stitching results back with NMS to remove duplicates

**Expected improvement**: 15.8 → 19-21 players/frame

```python
# pip install sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="data/models/football_best.pt",
    confidence_threshold=0.3,
    device="cpu"
)

result = get_sliced_prediction(
    image=frame,
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

#### Manual Team Color Input — What VEO Does

VEO requires users to "choose and assign team colors" and "confirm sides and periods" before analytics work. This is the industry standard. We should:

1. Add color pickers to the upload form (optional)
2. If provided, skip auto-detection entirely
3. If not provided, use our auto-detection (HSV Hue + event_detection bridge)

#### GK Detection Heuristic

The football model rarely detects GKs at VEO distance. Instead:
1. After team classification, find the color outlier in each team
2. Cross-reference with pitch position (penalty area or behind defensive line)
3. Persist GK identity through tracking
4. If football model does detect class_id=1 (GK), use that as confirmation

### Phase 4B: Tracking Upgrade

| Task | Impact | Effort |
|------|--------|--------|
| Evaluate SportSORT (HOTA 88% on SoccerNet) | Best sports tracker | Medium |
| Add ReID features (OSNet) for re-identification | Reduce ID switches | Medium |
| Out-of-view re-association | Handle players leaving/entering frame | Medium |

### Phase 4C: Pitch Detection & Mapping

| Task | Impact | Effort |
|------|--------|--------|
| Robust pitch keypoint detection | Accurate pitch mapping | Medium |
| Automatic homography from detected lines | Pixel→pitch coordinates | Medium |
| Handle VEO fisheye distortion | Required for wide-angle | Medium |

---

## Full Product Feature Comparison

### Video Capture & Input

| Feature | VEO | HUDL | Manager Mentor (Target) |
|---------|-----|------|------------------------|
| Camera hardware | VEO Cam 3 ($1K+) | Focus ($2K+) | None needed |
| Video sources | VEO camera only | HUDL camera + upload | ANY source (VEO, HUDL, phone, broadcast) |
| Live streaming | Yes (subscription) | Yes | Planned |
| Resolution | 4K panoramic → 1080p crop | 1080p panoramic | Whatever input provides |
| Upload time | 1-12+ hours | Variable | Standard web upload |

### Detection & Tracking

| Feature | VEO | HUDL | Manager Mentor (Target) |
|---------|-----|------|------------------------|
| Player detection | Unknown model | DETR | YOLO + SAHI |
| Player tracking | Proprietary | 30fps tracking | ByteTrack/SportSORT |
| Team classification | Manual input | Color + human QA | Auto (HSV/SigLIP) + manual option |
| Jersey numbers | Yes (manual lineup) | CNN (unreliable) | AI vision (GPT-4V) |
| Ball tracking | Yes | Yes | Yes (SAHI + parabolic) |
| GK identification | Unknown | Unknown | Heuristic + model class |
| Referee detection | Unknown | Unknown | Football model class_id=3 |

### Event Detection

| Feature | VEO | HUDL | Manager Mentor (Target) |
|---------|-----|------|------------------------|
| Goals | Auto | Auto (Am. football) | Auto |
| Corners/Free kicks | Auto | Human tagged | Auto |
| Passes | NO | Human tagged | Auto (pass detector) |
| Tackles | NO | Human tagged | Auto |
| Shots | Auto (on goal) | Human tagged | Auto + xG |
| Formations | NO | NO | Auto (formation detector) |
| Set pieces | Partial | Human tagged | Auto |
| Tactical events | NO | NO | Auto (pressing triggers, defensive shape) |
| Custom tags | Manual only | Manual code windows | AI-suggested |

### Analytics

| Feature | VEO | HUDL | Manager Mentor (Target) |
|---------|-----|------|------------------------|
| Possession % | Yes | Yes | Yes |
| Pass accuracy | Partial | StatsBomb (human) | Auto |
| Pass network | NO | StatsBomb (human) | Auto |
| xG | NO | StatsBomb ($$$) | Auto (built-in model) |
| Formation timeline | NO | NO | Auto |
| Heatmaps | Yes | Yes | Yes |
| Speed/distance | Yes (GPS addon) | Yes (30fps tracking) | From tracking |
| Defensive line height | NO | NO | Auto |
| PPDA (pressing) | NO | StatsBomb | Auto |

### Coaching Intelligence (OUR DIFFERENTIATOR)

| Feature | VEO | HUDL | Manager Mentor |
|---------|-----|------|---------------|
| AI match summary | NO | NO | **YES — natural language** |
| Coaching insights | NO | NO | **YES — categorized by priority** |
| Training drills | NO | NO | **YES — AI-generated from weaknesses** |
| Session plans | NO | NO | **YES — warm-up → cool-down** |
| Team talk scripts | NO | NO | **YES — half-time/full-time** |
| AI coach chat | NO | NO | **YES — ask questions about the match** |
| Tactical alerts | NO | NO | **YES — real-time pressing/shape warnings** |
| Improvement reports | NO | NO | **YES — per-team improvement areas** |

---

## Tech Stack Comparison

| Component | VEO | HUDL | Manager Mentor |
|-----------|-----|------|---------------|
| Backend | Python microservices | C#/.NET | Python/FastAPI |
| Frontend | React | React/TypeScript | React/TypeScript |
| ML | Unknown (Python) | Python/PyTorch | Python/YOLO/supervision |
| Database | Unknown | MongoDB | SQLite/PostgreSQL |
| Cloud | Unknown | AWS (100%) | Flexible (Railway/AWS) |
| CDN | Unknown | CloudFront + Fastly | CloudFront |
| CI/CD | Unknown | TeamCity | GitHub Actions |
| Video encoding | Unknown | Visionular Aurora4 | FFmpeg |
| AI/LLM | None | None | Claude/GPT-4 |
| Container | Unknown | Docker (no K8s) | Docker |

---

## Revenue Model Comparison

| Tier | VEO | HUDL | Manager Mentor (Target) |
|------|-----|------|------------------------|
| Free | No | No | 1 match/month, basic analytics |
| Grassroots | $1,500/yr + camera | $400/yr | $10-20/month |
| Club | $2,500/yr + camera | $1,000-1,800/yr | $50-100/month |
| Pro | Custom | Custom + StatsBomb | $200-500/month |
| Hardware | Required ($1K+) | Required ($2K+) | None |

---

## Implementation Roadmap

### Now: Phase 4A — Detection Pipeline (2-3 sessions)
- [ ] SAHI tiled inference
- [ ] Vectorize jersey color extraction + upscale small crops
- [ ] Manual team color input via upload form
- [ ] a*b* LAB fallback (drop L channel)
- [ ] UMAP components 10→3
- [ ] GK detection heuristic
- [ ] DBSCAN outlier filtering

### Next: Phase 4B — Tracking + Pitch (2-3 sessions)
- [ ] Evaluate SportSORT vs ByteTrack
- [ ] Pitch line detection for homography
- [ ] Handle VEO distortion
- [ ] Out-of-view re-association

### Then: Phase 5 — Product Polish
- [ ] Frontend DugoutIQ wiring (Phase 3 already built the shell)
- [ ] Export reports (PDF/JSON)
- [ ] Video clip generation for highlights
- [ ] Print-friendly training session plans
- [ ] Mobile-responsive frontend

### Future: Phase 6 — Market Differentiators
- [ ] Real-time live analysis mode
- [ ] Multi-match trend analysis
- [ ] Player development tracking across season
- [ ] Opposition scouting database
- [ ] API for third-party integrations
- [ ] White-label for academies/clubs

---

## Key Insight: Why We Can Win

**VEO** is a hardware company that bolts on cloud AI. Their moat is the camera — without it, you can't use VEO. Their AI is limited (8 auto-detected events, manual team assignment, 24-48hr processing).

**HUDL** is an acquisition roll-up with ecosystem lock-in. Their moat is market share + StatsBomb data. But their AI is weak (American football only), Sportscode is macOS-only, and they use expensive human analysts.

**Neither of them have coaching intelligence.** They show you what happened. They don't tell you what to do about it. A grassroots coach doesn't need a $3,000/year analytics platform — they need an AI assistant that watches the game, spots the problems, and says "here's a 60-minute training session to fix it."

That's Manager Mentor.
