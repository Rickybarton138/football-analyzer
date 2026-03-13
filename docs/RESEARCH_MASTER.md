# FootballManager.AI - Master Research Document

**Date:** 2 March 2026
**Project:** Manager Mentor (evolved from Football Analyzer)
**Goal:** Build a VEO/HUDL-killer AI football analysis platform

---

## TABLE OF CONTENTS

1. [Competitive Intelligence](#1-competitive-intelligence)
2. [Market Opportunity](#2-market-opportunity)
3. [Technology Stack](#3-technology-stack)
4. [Implementation Guide](#4-implementation-guide)
5. [Key GitHub Repos](#5-key-github-repos)
6. [Battle Plan](#6-battle-plan)

---

## 1. COMPETITIVE INTELLIGENCE

### VEO (veo.co) - $72M Revenue, $115M Raised

**Products:** Veo Cam 3 (~$1,200 + $42-178/mo subscription), Veo Go (iPhone-based)
**AI:** PyTorch/Lightning, YOLO-based CNN, Neptune.ai tracking, AWS cloud
**Stats:** Possession %, passes, shots, heatmaps, 2D radar, momentum graph
**Gaps:**
- 24-48hr upload/processing delay (ALL cloud, no edge AI)
- No data export / no public API
- No xG, no pressing metrics, no formations
- Camera is a brick without subscription
- Basic stats only
- No coaching recommendations

### HUDL (hudl.com) - $730M Revenue, $230M Raised

**Products:** Focus Flex camera ($2-3K/yr rental), Sportscode, Wyscout, StatsBomb, Titan GPS
**AI:** Human analysts + AI tagging (Assist), auto-tracking cameras
**Stats:** 3,400+ events/match via StatsBomb, xG, OBV, freeze-frame data
**Gaps:**
- Built for American football, adapted for soccer (wrong DNA)
- $2,000+/yr minimum - prices out grassroots
- Sportscode is macOS only
- No real-time tactical AI (1-2hr turnaround)
- Data locked in ecosystem
- Pricing not transparent

### Other Competitors

| Competitor | Focus | Price | Gap |
|-----------|-------|-------|-----|
| Metrica Sports | Software-only, free tier | Free-$88/mo | No coaching recommendations |
| SkillCorner | Broadcast tracking, enterprise | $$$$$ | Pro-only, no grassroots |
| Pixellot | Auto-cameras + streaming | $949 + $69-167/mo | Filming-first, basic analysis |
| Trace | US youth highlights | $180/yr | Parent-facing, no tactics |
| Spiideo | Semi-pro analysis | $3,800/yr | Too expensive for grassroots |
| Track160 | Optical tracking, FIFA certified | Custom | Installation required |
| zone14 | Grassroots Germany | Competitive | Early-stage, limited AI |
| tactico | UK grassroots management | Early | No video analysis |

### THE GAP NOBODY IS FILLING

**Not a single product connects "what happened" to "what to train on Tuesday."**
Every tool stops at data visualization. None provide:
- AI coaching recommendations from match footage
- Age-appropriate coaching intelligence
- Session planning linked to match analysis
- Player development tracking over time
- Affordable team analysis ($10-30/month)

---

## 2. MARKET OPPORTUNITY

### Market Size
- Global sports analytics: $5.79B (2025) -> $31.14B (2034), 20.5% CAGR
- UK: 40,000+ FA-registered clubs, 125,000+ teams, 1.4M+ registered players
- ~160,000 grassroots coaches, 1M+ volunteers
- 70% of grassroots clubs are financially struggling

### Pricing Strategy
| Tier | Price | Target |
|------|-------|--------|
| Free | $0 | Any coach, 2 matches/month |
| Coach | $9.99/mo ($99/yr) | Grassroots, 8 matches/month |
| Club | $29.99/mo ($299/yr) | Multi-team, unlimited |
| Academy | $79.99/mo ($799/yr) | Full analytics, player dev |
| Pro | Custom | Professional clubs, API |

### Cloud Processing Cost Per Match: ~$2-3 (at scale: $0.50-1.50)
### Unit Economics: Profitable from Club tier ($29.99/mo)

### Go-To-Market
1. County FA partnerships (51 County FAs)
2. League-level adoption (network effects)
3. Coach education pathway (FA courses require video analysis for UEFA B+)
4. Social media (TikTok/Instagram short-form AI analysis clips)
5. Parent viral growth (auto-generated player highlights)

### Legal Requirements
- GDPR consent management for filming (especially children)
- FA Safeguarding Guidance Note 8.3 compliance
- Parental consent mandatory before filming children
- Encrypted storage, access controls, auto-delete policies

---

## 3. TECHNOLOGY STACK (State of the Art 2026)

### Player Detection
- **YOLO26** (latest, Jan 2026) or **YOLO11** (stable fallback)
- YOLO26 key: NMS-free, DFL removal (43% faster CPU), STAL for small objects
- Fine-tune on Roboflow football-players-detection dataset (9,068 images, 4 classes)
- Train at `imgsz=1280` for ball detection
- Results: ~98% mAP50 for players, ~65% for ball at 1280

### Multi-Object Tracking
- **BoT-SORT** (default, has ReID + camera motion compensation)
- **ByteTrack** (faster, good for real-time)
- **SportSORT** (2025, purpose-built for sports: HOTA 88% on SoccerNet)
- Custom config: `track_buffer=60` (2s), `match_thresh=0.85`, `gmc_method=sparseOptFlow`

### Ball Detection & Tracking
- Multi-method: YOLO + TrackNet heatmap + motion detection + color filtering
- Kalman filter for trajectory prediction
- Parabolic interpolation for aerial balls
- Physics-based: `distance = speed * time`, gravity for flight paths
- Possession: 2m radius zone, 5-10 frame inertia (FIFA-validated 90%+ accuracy)

### Team Classification
- **SigLIP + UMAP + KMeans** (Roboflow method, 90-95% accuracy)
- Fallback: KMeans on LAB color space jersey pixels
- Temporal smoothing via majority vote over 30-frame window

### Jersey Number Recognition
- **PARSeq** fine-tuned on jersey data (87.4% accuracy, SoccerNet SOTA)
- EasyOCR + temporal voting as free baseline
- GPT-4V / Claude Vision as high-accuracy fallback
- Cost optimization: OCR first, API only when OCR fails

### Pitch Mapping
- YOLOv8-pose for 32 pitch keypoints
- `cv2.findHomography()` + `cv2.perspectiveTransform()`
- Bottom-center of bounding box = foot position
- 1st place SoccerNet 2023: HRNet + 57 keypoints

### Event Detection
- Rule-based for well-defined events (passes, shots, goals, set pieces)
- ML-based (T-DEED architecture) for complex events (tackles, fouls)
- Possession Zone (2m) + Duel Zone (3m) model
- PathCRF for ball-free event detection from trajectories

### xG Model
- Features: distance, angle, body part, GK position, blockers, pressure
- XGBoost with monotonic constraints (StatsBomb approach)
- Train on StatsBomb open data (free)
- Advanced: Skor-xG (CVPR 2025) uses 3D player skeletons via GNN

### Tactical Analytics
- Formation detection: KMeans clustering on player x-positions
- Pressing: PPDA + pitch control model (Spearman)
- Space control: Voronoi diagrams
- Team shape: Convex hull area
- Defensive line height tracking
- Counter-pressing: Ball recovery within 5s

### AI Coaching Intelligence (THE DIFFERENTIATOR)
- Feed structured analytics into Claude/GPT for natural language recommendations
- Age-appropriate coaching intelligence (U8 vs U16)
- Link analysis to specific training drills
- Session plan generation from match weaknesses
- Pattern detection across multiple matches

---

## 4. IMPLEMENTATION GUIDE

### Core Libraries
```
ultralytics          # YOLO models (detection, tracking, pose)
supervision          # CV toolkit (annotation, zones, heatmaps)
roboflow/trackers    # ByteTrack, OC-SORT (Apache 2.0)
opencv-python        # Video processing, homography
mplsoccer            # Football pitch visualization
statsbombpy          # Free training data for xG
scikit-learn         # KMeans, XGBoost
torch/torchvision    # PyTorch for deep learning
```

### Detection Pipeline
```python
from ultralytics import YOLO
import supervision as sv

model = YOLO("best.pt")  # Fine-tuned football model
tracker = sv.ByteTrack(track_activation_threshold=0.6, lost_track_buffer=60)

for frame in sv.get_video_frames_generator("match.mp4"):
    result = model(frame, conf=0.3, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
```

### Ball Tracking
- YOLO detection at 1280 resolution
- SAHI (image slicing) for missed frames
- Kalman filter smoothing + trajectory prediction
- Linear interpolation for ground passes
- Parabolic interpolation for aerial balls
- Possession: closest player within 2m radius

### Event Detection (Rule-Based)
```
Pass: Player A loses ball -> ball travels -> Player B (same team) gains
Shot: Ball trajectory toward goal from <35m
Interception: Pass interrupted, opposing team gains
Corner/GoalKick/ThrowIn: Ball exits play boundaries
Dribble: Player maintains ball while moving past opponents
```

---

## 5. KEY GITHUB REPOS

### Tier 1 (Essential)
| Repo | Stars | Use |
|------|-------|-----|
| roboflow/sports | 4,887 | Complete soccer CV pipeline |
| roboflow/supervision | 36,608 | CV toolkit for everything |
| roboflow/trackers | 2,924 | Modular tracking algorithms |
| SoccerNet/sn-gamestate | 379 | End-to-end game state reconstruction |
| ultralytics/ultralytics | 53,800 | YOLO framework |

### Tier 2 (Reference)
| Repo | Stars | Use |
|------|-------|-----|
| abdullahtarek/football_analysis | 878 | Tutorial pipeline |
| NikolasEnt/soccernet-calibration-sportlight | 59 | 1st place pitch calibration |
| lRomul/ball-action-spotting | 128 | 1st place action detection |
| mguti97/PnLCalib | 57 | Latest pitch calibration (2026) |
| mkoshkina/jersey-number-pipeline | 55 | Best jersey OCR (CVPR 2024) |
| statsbomb/statsbombpy | 696 | Free xG training data |
| tryolabs/norfair | 2,621 | Alternative tracking |
| jac99/FootAndBall | 128 | Lightweight ball+player detector |
| SkalskiP/sports | 542 | GPT-4V jersey experiments |

### Key Datasets
- Roboflow football-players-detection: 9,068 images, 4 classes
- SoccerNet: 500 games, 300K+ annotations
- StatsBomb open data: Free event data with freeze frames
- Metrica Sports sample data: Tracking data (players + ball)

---

## 6. BATTLE PLAN

### Phase 0: Foundation (Week 1-2)
- [ ] Upgrade from YOLOv8 to YOLO26/YOLO11
- [ ] Fine-tune on Roboflow football-players-detection dataset
- [ ] Replace current tracking with BoT-SORT (custom football config)
- [ ] Implement SigLIP+UMAP+KMeans team classification
- [ ] Fix ball detection (SAHI + Kalman filter + interpolation)
- [ ] Add pitch keypoint detection + homography mapping

### Phase 1: Core Analysis Engine (Week 3-4)
- [ ] Rule-based event detection (passes, shots, interceptions, set pieces)
- [ ] Ball possession calculation (2m PZ, 5-frame inertia)
- [ ] Distance covered + speed zones per player
- [ ] xG model (XGBoost on StatsBomb data)
- [ ] Heatmaps, pass networks, shot maps
- [ ] Formation detection

### Phase 2: AI Coaching Intelligence (Week 5-6)
- [ ] Claude/GPT integration for match analysis reports
- [ ] Natural language coaching recommendations
- [ ] Training session suggestions linked to match weaknesses
- [ ] Age-appropriate analysis modes (U8/U10/U12/U14/U16/Adult)
- [ ] Pattern detection across multiple matches

### Phase 3: Platform & Frontend (Week 7-8)
- [ ] New React frontend (rebrand as Manager Mentor)
- [ ] Video upload + cloud processing pipeline
- [ ] Match dashboard with interactive visualizations
- [ ] Player profiles with development tracking
- [ ] Team management features

### Phase 4: Parent & Player Portal (Week 9-10)
- [ ] Auto-generated player highlight reels
- [ ] Individual player development reports
- [ ] Social media sharing (viral growth)
- [ ] Parent-facing progress views
- [ ] GDPR consent management

### Phase 5: Deploy & Go-To-Market (Week 11-12)
- [ ] Deploy to Railway/AWS
- [ ] Stripe subscription billing
- [ ] Free tier with 2 matches/month
- [ ] Marketing site
- [ ] Beta test with 5-10 grassroots teams
- [ ] County FA outreach

### Key Differentiators vs Competition
1. **AI that tells you WHAT TO DO** (not just what happened)
2. **Software-only** (BYOC - bring your own camera/phone)
3. **50-70% cheaper** than VEO/Hudl
4. **Age-appropriate** coaching intelligence
5. **Player development tracking** over time
6. **Open data** (export, API)
7. **Coach education integration** (FA pathway)
8. **Parent engagement** (viral growth engine)

---

## RESEARCH SOURCES

Full agent transcripts available in:
- `C:\Users\info\AppData\Local\Temp\claude\C--Users-info\tasks\` (10 research agents)

Research covered:
1. VEO deep-dive (features, pricing, tech, weaknesses)
2. HUDL deep-dive (products, AI, pricing, acquisitions)
3. Full competitive landscape (15+ competitors analysed)
4. Cutting-edge AI/CV technology (YOLO26, SportSORT, TrackNetV4, etc.)
5. Market size & monetization strategy
6. Player detection & tracking implementation (code examples)
7. Ball tracking implementation (code examples)
8. Pitch mapping & coordinate transformation (code examples)
9. Event detection & xG models (code examples)
10. Best GitHub repos & open-source tools (50+ repos catalogued)
