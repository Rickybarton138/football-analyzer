# State of the Art: Football Video Analysis with AI & Computer Vision
## Comprehensive Technical Research Report -- March 2026

---

## 1. INDUSTRY LANDSCAPE: HOW THE MAJOR PLAYERS WORK

### 1.1 VEO Technologies

**Hardware:** Dual 4K lenses capturing a 180-degree panoramic view of the full pitch. The Veo Cam 3 records at 1080p@30fps with built-in 5G connectivity.

**AI Pipeline:**
- YOLO-based object detection architecture for real-time player/ball identification
- AI-powered "follow-cam" that automatically pans and zooms to follow the action -- no camera operator needed
- Automatic event detection: goals, kickoffs, and basic match events
- Jersey number recognition and player position identification are part of their processing pipeline
- Video is automatically uploaded to the Veo Editor cloud platform post-match

**What they provide to coaches:**
- Automatic highlights and key moment tagging
- Manual player tagging and clip creation in the Editor
- Instant Playback during half-time
- Basic event timeline

**Limitations:** Physical tracking data (distance, speed) is described as "on our roadmap" as of 2021. Their strength is accessible video capture, not deep analytics. Subscription required (~$100-200/month).

---

### 1.2 Hudl (Sportscode + StatsBomb + Wyscout + IQ)

Hudl is the 800-pound gorilla, having acquired StatsBomb (2024), Wyscout, and building Hudl IQ.

**Hudl IQ (the AI engine):**
- Powered by AI, computer vision, and expert human data collectors
- Claims "10x more data than any one provider, and in less time"
- Uses a hybrid Human + AI approach: CV generates initial data, human collectors audit and correct
- Multi-stage quality review: "A, B, and C" audits
- Computer vision handles: player detection, player tracking, pitch homography/camera calibration
- Generates high-frequency tracking data from broadcast video
- Physical metrics (distance, speed, sprints) calculated from CV-derived tracking data

**StatsBomb Event Data:**
- ~3,000+ events captured per match
- Human operators use AI-assisted collection tools
- StatsBomb 360 data: adds visible teammate AND opposition locations to every event
- On-Ball Value (OBV) -- possession-state model measuring +/- impact of every action on scoring/conceding likelihood
- xG model incorporating freeze-frame positional data
- Multi-stage pipeline: CV for homography estimation -> player localization -> human-verified event coding

**Hudl Sportscode:**
- Professional video analysis platform for coding/tagging
- Studio feature uses computer vision for automatic player tracking in telestration graphics
- Integrates with StatsBomb data and Wyscout video
- Custom scripting and workflow automation

**Wyscout:**
- ~2,000 events tagged per match
- Coverage of 600+ leagues, 500K+ player profiles
- Primarily human-coded but increasingly AI-assisted
- Integrated into Hudl StatsBomb platform

**Data provided to coaches:**
- Complete event data (passes, shots, tackles, fouls, carries, pressures)
- Physical metrics (distance, speed, sprints) from tracking
- xG, xA, OBV for every action
- Player scouting and comparison tools
- Customizable dashboards and visualizations

---

### 1.3 Opta (Stats Perform)

**Data Collection:**
- Team of trained analysts capture live data using proprietary collection tools
- Up to 3,000 actions captured per match
- AI-powered live data enrichment via computer vision
- Inter-operator reliability: kappa values of 0.92-0.94, average event time difference of 0.06 +/- 0.04 seconds
- Data output in XML format across multiple feed types

**Event taxonomy:** 70+ event types including passes, shots, tackles, fouls, offsides, clearances, interceptions, aerial duels, dribbles, etc. Each event has multiple qualifiers providing additional context.

**Pressure metric:** Goes beyond PPDA -- uses proprietary CV-based measure of defensive pressure incorporating player positions and movements.

---

### 1.4 ReSpo.Vision

**Key differentiator:** FIFA-certified broadcast tracking system that turns ANY match video into elite tracking data.

- 3D tracking data from standard broadcast footage
- AI and computer vision powered
- No special cameras or hardware required
- Serves clubs, leagues, federations, broadcasters, and betting companies
- Works from grassroots to elite competitions
- Auto-generated reports and real-time insights

---

### 1.5 Zone14

**Approach:** Fixed infrastructure cameras + AI analytics, specifically targeting grassroots/amateur transformation.

- zone14 STATS: Individual player data without GPS trackers
- zone14 REPLAY: Video analysis platform for all club levels
- AI algorithms automatically identify and categorize key moments (goals, set-pieces)
- Philosophy: solve volunteer time constraints and resource gaps in grassroots football

---

### 1.6 Other Notable Platforms

| Platform | Focus |
|----------|-------|
| **Footovision** | Tracks ball, players, referee from any video feed; no installation needed |
| **GAMEFACE.AI** | Auto-tags goals, free kicks, fouls, shots on goal; no manual intervention |
| **Impact Soccer (mpact.ai)** | CV-driven stats and insights from match video |
| **Pixellot** | AI camera + auto-tagging + data analysis; strong in grassroots |
| **Trace** | AI camera system with automated highlight generation |
| **XbotGo** | Single 4K camera with gimbal + AI tracking software |

---

## 2. CORE COMPUTER VISION PIPELINE

### 2.1 Object Detection (Players, Ball, Referees)

**Current State of the Art (2025-2026):**

| Model | Year | Key Feature | mAP | Speed |
|-------|------|-------------|-----|-------|
| **YOLO26** | 2025 | Unified 5-task architecture | Best | Fast |
| **YOLOv12** | 2025 | Attention-centric (NeurIPS 2025) | 80.9+ | Real-time |
| **YOLO11** | 2024 | C3K2 module, 22% fewer params vs v8 | High | Very fast |
| **YOLOv8x** | 2023 | Mature, well-tested | 94.8% mAP (football-tuned) | Fast |
| **RT-DETR / RT-DETRv2** | 2024 | Transformer-based, end-to-end | Competitive | Real-time |
| **RF-DETR** | 2025 | Pareto-optimal over D-FINE | Competitive | Real-time |

**Practical recommendation for football:** Fine-tuned YOLOv8x or YOLO11x on football-specific datasets (e.g., Roboflow Football Players Detection dataset, SoccerNet) consistently delivers >90% mAP for player detection. The SoccerNet 2024/2025 GSR challenge winner used fine-tuned YOLOv5m with high-recall configuration.

**Ball detection challenge:** Football is small, fast-moving, and frequently occluded. Dedicated ball detection models or specialized training with heavy augmentation are required. Ball interpolation between detections is standard practice.

**IMPORTANT licensing note:** Ultralytics YOLO models (v5, v8, v11, v12, v26) are under AGPL-3.0 license -- commercial use requires a paid Ultralytics license. Alternatives with permissive licenses include RT-DETR variants and Chinese YOLO forks.

---

### 2.2 Multi-Object Tracking (MOT)

**Dominant approaches:**

**ByteTrack (ECCV 2022):**
- Associates EVERY detection box, including low-confidence ones
- 80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17
- Excels in static-camera scenarios (which grassroots football typically is)
- Holds tracks through brief occlusions via two-stage association
- Lightweight -- no deep features needed, just IoU + Kalman filter

**BoT-SORT:**
- Extends ByteTrack with: refined box predictions, camera motion compensation (GMC), combined motion + appearance cues
- Better for broadcast video with moving cameras
- SoccerNet tracking AV 2025/26 solution: YOLOv11x + BoT-SORT with GMC + adaptive field masking (NO ReID module needed)

**OC-SORT:** Improves occlusion handling with virtual trajectories.

**StrongSORT:** Adds ReID features + NSA Kalman filter + camera motion compensation.

**DeepSORT:** Classic tracker adding CNN-based ReID embeddings. Used in SoccerNet GSR challenge winner pipeline.

**Ultralytics integration:** Both ByteTrack and BoT-SORT are natively supported in Ultralytics YOLO via YAML config files, making implementation trivial.

**Practical recommendation:** For a fixed-camera grassroots setup, ByteTrack is sufficient and fast. For broadcast-style video with camera panning, BoT-SORT with GMC is the better choice. Add ReID only if identity persistence across long occlusions is critical.

---

### 2.3 Team Classification (Jersey Color)

**Standard approach (works well):**

1. Detect players with YOLO -> get bounding box crops
2. Remove green pixels (grass) from each crop using HSV thresholding
3. Run KMeans clustering (k=2) on remaining pixel colors
4. The dominant non-green cluster center represents the jersey color
5. Cluster all players into 2 teams based on jersey color similarity

**Reported accuracy:** 86-92.5% across various studies.

**Advanced approaches:**
- **DBSCAN clustering** on RGB color distributions (no need to specify k)
- **Lab color space** conversion for better perceptual color separation
- **Image embeddings + UMAP + KMeans** for more robust clustering
- **Temporal anchoring** to stabilize team labels across frames
- **Segmentation models** (SAM/Mask R-CNN) to isolate only the jersey region before color analysis

**Challenges:**
- Similar jersey colors between teams (e.g., dark blue vs black)
- Goalkeepers wearing different colors
- Referees need separate classification
- Varying lighting conditions across the pitch
- Kit changes (home vs away) need per-match calibration

---

### 2.4 Jersey Number Recognition

**State of the art pipeline (from SoccerNet Jersey Number Recognition challenge):**

1. **Legibility classification:** Binary classifier determines if jersey number is readable in a given crop (many frames show backs turned, numbers occluded)
2. **Number region localization:** Use pose estimation (ViTPose) to find shoulder/hip keypoints -> crop the torso region
3. **Number recognition:** CNN-based digit recognition on the cropped region

**Best approaches:**
- **Pose-Guided R-CNN:** Uses body pose to guide the attention of the number recognition network
- **Multi-task learning with orientation-guided weight refinement** (2024 paper)
- **Synthetic data pre-training:** Generate synthetic 2-digit numbers with font/color variations, pre-train, then fine-tune on real data
- **Temporal voting:** Aggregate predictions across multiple frames for the same tracked player -- majority vote for final number assignment

**Key datasets:**
- SoccerNet Jersey Number Recognition dataset (derived from sn-gamestate and sn-tracking)
- SVHN (Street View House Numbers) for pre-training
- Custom synthetic datasets (Simple2D, Complex2D)

**Practical accuracy:** Individual frame accuracy is moderate (~60-75%), but temporal voting across 50+ frames pushes reliability significantly higher for tracked players.

---

### 2.5 Camera Calibration & Pitch Homography

**Why it matters:** Converting pixel coordinates to real-world pitch coordinates (meters) is essential for: distance calculation, speed estimation, heatmaps, tactical maps, formation analysis, and all positional metrics.

**Approaches:**

**Keypoint-based:**
1. Detect pitch keypoints/landmarks (corners, intersections, penalty spots, center circle)
2. Match to known pitch template coordinates
3. Compute homography matrix via `cv2.findHomography()`

**Deep learning-based:**
- **Narya:** DeepHomoModel for direct homography estimation + KeypointDetectorModel (EfficientNetB3 backbone, 29 keypoint classes)
- **SegFormer-based camera parameter estimation** (SoccerNet GSR 2024 winner)
- **TVCalib:** Full camera calibration (not just homography) from broadcast video
- **SoccerNet Camera Calibration Challenge:** Dedicated benchmark with homography decomposition into rotation/translation matrices

**StatsBomb's approach:** "Assuming cameras are well characterized by the pinhole model, retrieving camera pose is equivalent to determining the homography between the observed pitch within the frame and a template."

**SoccerNet 2023 Calibration Challenge winner:** Available open-source at `github.com/NikolasEnt/soccernet-calibration-sportlight`.

**Bayesian approach (2024):** BHITK (Bayesian Homography Inference from Tracked Keypoints) -- uses Kalman filter for temporally smooth homography estimation, reducing jitter.

**For fixed cameras (grassroots):** Homography can be computed ONCE at setup time if the camera position doesn't change, dramatically simplifying the problem. Only need 4+ point correspondences.

---

### 2.6 Player Re-Identification (ReID)

**The problem:** When a player exits and re-enters the frame (or across camera cuts in broadcast), how do you maintain their identity?

**Approaches:**
- **PRTreID (SoccerNet GSR):** Produces ReID embeddings that are identity, team, and role aware
- **Swin Transformer-based ReID** (2024): Enhanced transformer for soccer player re-identification across camera angles
- **DeepSORT-style:** CNN backbone (ResNet) trained with triplet loss + ID loss for feature extraction
- **Sports Re-ID framework:** Based on Torchreid, specifically adapted for sports broadcast video
- **Multi-task learning:** Joint ReID + team affiliation + role classification

**For single fixed camera (grassroots):** ReID is less critical since players rarely leave the frame entirely. ByteTrack/BoT-SORT track buffers handle brief occlusions. Jersey number recognition provides a stronger identity signal when available.

---

## 3. OPEN-SOURCE TOOLS & DATASETS

### 3.1 SoccerNet (THE benchmark)

**Scope (2025-2026):**
- 500+ complete soccer games from 6 European leagues across 3 seasons
- 300K+ manual annotations

**2025 Challenge tasks:**
1. **Team Ball Action Spotting** -- detect ball-related actions AND assign to teams
2. **Monocular Depth Estimation** -- relative depth maps from single-camera clips
3. **Multi-View Foul Recognition** -- balanced accuracy across foul types/severities
4. **Game State Reconstruction** -- full minimap reconstruction with GS-HOTA metric

**2026 Challenge announced** (September 2025), includes Visual Question Answering (VQA) task.

**Sub-datasets:**
- `sn-gamestate` -- Game State Reconstruction annotations
- `sn-tracking` -- Multi-object tracking annotations
- `sn-jersey` -- Jersey number recognition benchmark
- `sn-calibration` -- Camera calibration benchmark
- `sn-reid` -- Re-identification benchmark
- `SN-MVFouls` -- Multi-view foul recognition

**GitHub:** `github.com/SoccerNet` (all repos with baselines and dev kits)

---

### 3.2 Key Open-Source Projects

| Project | What It Does | URL |
|---------|-------------|-----|
| **roboflow/sports** | Player detection, tracking, team clustering, camera calibration (by Piotr Skalski) | github.com/roboflow/sports |
| **Narya** | Player tracking + pitch homography + Expected Discounted Goal agent | github.com/DonsetPG/narya |
| **jersey-number-pipeline** | General framework for jersey number recognition (legibility + recognition) | github.com/mkoshkina/jersey-number-pipeline |
| **SportsLabKit** | Toolkit for turning sports video into CSV data; MOT focus | github.com/AtomScott/SportsLabKit |
| **soccernet-calibration-sportlight** | 1st place SoccerNet Camera Calibration 2023 | github.com/NikolasEnt/soccernet-calibration-sportlight |
| **SoccerNet-tracking AV 2025-26** | YOLO11x + BoT-SORT + adaptive field masking | github.com/apiantonio/SoccerNet-tracking_AV2025-26 |
| **Football-Analysis-System** | YOLO11 + ByteTrack end-to-end system | github.com/AmmarMohamed0/Football-Analysis-System |
| **abdullahtarek/football_analysis** | YOLO + KMeans + optical flow + perspective transform | github.com/abdullahtarek/football_analysis |
| **ByteTrack** | State-of-the-art MOT tracker | github.com/FoundationVision/ByteTrack |
| **BoT-SORT** | Robust multi-pedestrian tracking | github.com/NirAharon/BoT-SORT |
| **sportsreid** | Player re-identification for broadcast video | github.com/shallowlearn/sportsreid |
| **TeamTrack** | Multi-sport MOT benchmark for full-pitch videos | atomscott.github.io/TeamTrack |
| **socceraction** | SPADL/VAEP framework for event data from Opta/StatsBomb/Wyscout | socceraction.readthedocs.io |

---

### 3.3 Key Datasets

| Dataset | Size | Content |
|---------|------|---------|
| SoccerNet v2/v3 | 500+ games | Actions, tracking, calibration, ReID |
| Roboflow Football Players Detection | ~10K images | Player/ball bounding boxes |
| DFL Bundesliga Data Shootout (Kaggle) | Multiple games | Broadcast footage |
| Wyscout Public Dataset | 1,941 matches | Spatio-temporal event data |
| StatsBomb Open Data | Multiple tournaments | Free event + 360 data |

---

## 4. DERIVED ANALYTICS: WHAT CAN BE COMPUTED

### 4.1 From Tracking Data (positions over time)

| Metric | How | Difficulty |
|--------|-----|-----------|
| **Heatmaps** | Aggregate player positions over time, apply kernel density estimation | Easy |
| **Distance covered** | Sum Euclidean distances between consecutive positions (needs homography) | Easy |
| **Sprint detection** | Threshold velocity (>25 km/h = sprint, >20 km/h = high-speed run) | Easy |
| **Speed estimation** | Delta-position / delta-time after homography mapping | Easy |
| **Formation detection** | Cluster player positions by role; compare to template formations | Medium |
| **Team shape / compactness** | Convex hull area of team positions; length/width ratios | Medium |
| **Pressing intensity** | Time-to-intercept models; or simplified PPDA from events | Medium |
| **Pitch control** | Voronoi diagrams or probability models based on player positions + velocities | Medium-Hard |
| **Pass networks** | Combine tracked positions with detected pass events | Medium |

### 4.2 From Event Detection

| Metric | How | Difficulty |
|--------|-----|-----------|
| **Possession %** | Track which team's player is closest to/controlling the ball | Medium |
| **Pass completion** | Detect pass events, check if received by teammate | Medium |
| **Shot detection** | Ball trajectory toward goal + player action recognition | Medium |
| **xG (Expected Goals)** | Shot location + angle + defender/GK positions + body part | Hard |
| **Tackle/interception detection** | Action recognition on player crops when ball changes possession | Hard |
| **Set piece classification** | Detect game stoppages + restart patterns (corner flag + player positions) | Medium |

### 4.3 What Requires Additional Intelligence

| Metric | How | Difficulty |
|--------|-----|-----------|
| **Player ratings** | Composite of all metrics, weighted by position/role | Hard (subjective) |
| **Defensive actions** | Ball ownership change + proximity analysis + action classification | Hard |
| **Pressing triggers** | Detect when pressing activates based on ball position + team shape | Hard |
| **Expected Threat (xT)** | Value each zone on the pitch based on historical goal probability | Medium (needs data) |
| **On-Ball Value (OBV)** | StatsBomb-style: every action's +/- impact on scoring/conceding | Very Hard |

---

## 5. FORMATION DETECTION: DEEP DIVE

**Latest research (2026):** "Deep learning for dynamic tactical formation recognition in professional football" (Scientific Reports, 2026) uses a Spatial-Graph Transformer (SGT) architecture.

**Standard approach:**
1. Get player positions via tracking + homography
2. Remove goalkeeper from analysis
3. Cluster outfield players into defensive lines using y-axis positions
4. Count players per line -> classify formation (e.g., 4 + 3 + 3 = 4-3-3)

**Advanced approach (2025-2026):**
1. Use tracking data to compute average positions over 5-10 minute windows
2. Apply graph neural networks with players as nodes
3. Classify using pre-defined formation templates
4. Handle dynamic formation changes (attacking vs defending shape)

**Tactical compliance scoring (2025):** YOLOv12 for detection -> positional mapping -> compute individual position deviation, formation stability, zone compliance, team compactness -> overall tactical score.

---

## 6. xG FROM VIDEO: THE FRONTIER

**Traditional xG** uses structured event data: shot location, angle to goal, body part, number of defenders, goalkeeper position.

**Skor-xG (CVPR 2025):** Skeleton-Oriented Expected Goal estimation -- uses pose keypoints of the shooter extracted from video to enrich xG models with body orientation and shooting posture.

**Video-derived xG pipeline:**
1. Detect shot event (ball moving toward goal at high velocity)
2. Extract shot location via homography
3. Count and locate defenders between ball and goal
4. Detect goalkeeper position
5. Estimate shooting angle
6. Feed features into trained xG model

**Feasibility assessment:** Computing basic xG from video is achievable. The accuracy won't match StatsBomb's freeze-frame xG (which uses manually verified positions of all 22 players), but it can provide a useful approximation for grassroots teams who have zero data otherwise.

---

## 7. GOOGLE DeepMind's TacticAI

**Published:** Nature Communications, 2024

**What it does:**
- Geometric deep learning over corner kick setups
- Players modeled as graph nodes with features (position, velocity, height)
- Edges represent inter-player relations
- Predicts first receiver after corner kick
- Predicts probability of shot resulting from corner
- Suggests tactical variations that improve outcomes
- Professional coaches rated TacticAI suggestions as favorable 90% of the time

**Significance:** Demonstrates that AI can not only ANALYZE but PRESCRIBE better tactics. This is the direction the field is heading.

---

## 8. THE COMPLETE GRASSROOTS ANALYSIS PACKAGE

Here is what a comprehensive Manager Mentor platform should deliver, with feasibility ratings:

### Tier 1: Achievable Now (Well-Proven Technology)

| Feature | Technology | Feasibility |
|---------|-----------|-------------|
| Player detection & tracking | YOLO + ByteTrack/BoT-SORT | HIGH |
| Team classification | KMeans on jersey colors | HIGH |
| Heatmaps per player | Tracking + homography + KDE | HIGH |
| Distance covered per player | Tracking + homography | HIGH |
| Speed / sprint detection | Velocity from position delta | HIGH |
| Ball tracking | YOLO fine-tuned + interpolation | HIGH |
| Possession % | Ball-player proximity | HIGH |
| Tactical map / minimap | Real-time 2D pitch projection | HIGH |
| Formation snapshot | Average positions + line clustering | HIGH |
| Automatic highlights | Event detection (goals, shots) | HIGH |

### Tier 2: Achievable with Effort (Proven but Needs Tuning)

| Feature | Technology | Feasibility |
|---------|-----------|-------------|
| Jersey number recognition | Pose-guided CNN + temporal voting | MEDIUM |
| Pass detection & mapping | Ball trajectory + possession change | MEDIUM |
| Pass networks | Event detection + player identity | MEDIUM |
| Shot detection | Ball trajectory + goal proximity | MEDIUM |
| Basic xG | Shot location + angle + defenders | MEDIUM |
| Pressing intensity (PPDA) | Event counting in zones | MEDIUM |
| Team shape / compactness | Convex hull + width/height ratios | MEDIUM |
| Set piece detection | Game stoppage + restart classification | MEDIUM |
| Player ratings (composite) | Weighted combination of all metrics | MEDIUM |

### Tier 3: Cutting Edge (Research-Level, Needs Significant Work)

| Feature | Technology | Feasibility |
|---------|-----------|-------------|
| Tackle/interception detection | Action recognition models | HARD |
| Defensive action classification | Ball ownership change + pose | HARD |
| Advanced xG with freeze-frame | All-player positional awareness | HARD |
| Expected Threat (xT) | Zone value model + tracking | HARD |
| Re-identification across cuts | ReID networks | HARD |
| Pitch control models | Voronoi + velocity vectors | HARD |
| Tactical prescription (TacticAI-style) | Graph neural networks | VERY HARD |

---

## 9. RECOMMENDED ARCHITECTURE FOR MANAGER MENTOR

### Pipeline Design

```
Video Input (single fixed camera, 1080p+)
    |
    v
[Frame Extraction] (ffmpeg, 5-10 fps for analysis)
    |
    v
[Object Detection] (YOLO11/YOLOv8 fine-tuned)
    |-- Players (bounding boxes + confidence)
    |-- Ball (bounding box + interpolation)
    |-- Referees (separate class)
    |
    v
[Multi-Object Tracking] (ByteTrack for fixed cam)
    |-- Track IDs assigned
    |-- Track buffers for occlusion handling
    |
    v
[Team Classification] (KMeans on jersey crops)
    |-- Team A vs Team B assignment
    |-- Goalkeeper identification
    |
    v
[Pitch Homography] (keypoint detection OR pre-calibrated)
    |-- Pixel coords -> pitch coords (meters)
    |-- For fixed camera: compute once at setup
    |
    v
[Jersey Number Recognition] (optional, hard)
    |-- Pose-guided crop -> CNN -> temporal voting
    |-- Maps track ID to player identity
    |
    v
[Analytics Engine]
    |-- Tracking metrics (distance, speed, heatmaps)
    |-- Event detection (passes, shots, possession)
    |-- Tactical metrics (formation, shape, pressing)
    |-- Composite player ratings
    |
    v
[Report Generation]
    |-- Per-player dashboards
    |-- Team tactical overview
    |-- Match summary with key events
    |-- Exportable data (CSV/JSON)
```

### Key Technical Decisions

1. **Fixed camera advantage:** A single fixed camera (like Veo/zone14) dramatically simplifies homography (compute once), tracking (no camera motion compensation needed), and reduces compute requirements.

2. **Processing approach:** Offline batch processing (not real-time) is perfectly acceptable for grassroots post-match analysis and allows for multi-pass processing, interpolation, and quality improvement.

3. **Model choices for Manager Mentor:**
   - Detection: YOLO11m or YOLOv8m (balance of speed and accuracy) -- or RT-DETR for permissive licensing
   - Tracking: ByteTrack (fixed camera) or BoT-SORT (moving camera)
   - Team classification: KMeans on HSV-filtered jersey crops
   - Homography: Pre-calibrated with manual point selection for fixed cameras
   - Jersey numbers: ViTPose + CNN + temporal voting (as enhancement, not MVP)

4. **What makes this BETTER than VEO for a grassroots manager:**
   - VEO gives you video + basic highlights
   - Manager Mentor should give you the ANALYTICS: heatmaps, distance data, formation analysis, pass maps, player ratings -- things that currently only professional clubs with Hudl IQ/StatsBomb access can get

---

## 10. KEY RESEARCH PAPERS TO REFERENCE

1. **SoccerNet Game State Reconstruction** (CVPRW 2024) -- End-to-end tracking + identification on minimap
2. **From Broadcast to Minimap** (CVPR 2025) -- 1st place GSR solution: YOLOv5m + SegFormer + DeepSORT + jersey number recognition
3. **SoccerNet 2025 Challenges Results** -- Latest benchmarks across 4 tasks
4. **Deep learning for dynamic tactical formation recognition** (Scientific Reports, 2026) -- SGT architecture for formation classification
5. **TacticAI** (Nature Communications, 2024) -- Google DeepMind's tactical AI assistant
6. **ByteTrack** (ECCV 2022) -- Multi-object tracking by associating every detection box
7. **A General Framework for Jersey Number Recognition** (CVPRW 2024) -- Koshkina et al.
8. **Skor-xG** (CVPRW 2025) -- Skeleton-oriented expected goals from video
9. **Pressing Intensity: An Intuitive Measure** (2025) -- Time-to-intercept probabilistic model
10. **Evaluating Football Players' Tactical Compliance** (2025) -- YOLOv12 + positional metrics + tactical scoring
11. **Creating Better Data: AI & Homography Estimation** (StatsBomb/Hudl IQ blog) -- Industry perspective on CV in data collection
12. **A Review of Computer Vision Technology for Football Videos** (MDPI Information, 2025) -- Comprehensive survey

---

## 11. COMPETITIVE POSITIONING SUMMARY

| Feature | VEO | Hudl/StatsBomb | Zone14 | Manager Mentor (Target) |
|---------|-----|---------------|--------|------------------------|
| Video capture | Built-in | External | Built-in | Any camera |
| Auto-tracking | Yes | N/A | Yes | N/A (fixed cam) |
| Player detection | Yes | Yes | Yes | Yes (YOLO) |
| Team classification | Basic | Yes | Yes | Yes (KMeans) |
| Jersey numbers | Partial | Yes (human-aided) | No | Yes (CV pipeline) |
| Event tagging | Basic auto | Comprehensive (human+AI) | Basic auto | Auto (CV-based) |
| Heatmaps | No | Yes | Partial | Yes |
| Distance/speed | No | Yes (IQ) | Yes (STATS) | Yes |
| xG | No | Yes (advanced) | No | Yes (basic) |
| Formation analysis | No | Yes | No | Yes |
| Pass networks | No | Yes | No | Yes |
| Pressing metrics | No | Yes | No | Yes |
| Player ratings | No | Partial | No | Yes |
| Price point | $100-200/mo | $$$$ (enterprise) | Mid-range | Accessible |
| Target market | Amateur-Pro | Professional | Amateur-Pro | Grassroots |

---

## 12. CONCLUSION

The technology to build a comprehensive football analysis platform from video alone is mature and proven. The key components (YOLO detection, ByteTrack/BoT-SORT tracking, KMeans team classification, homography for pitch mapping) are well-established with open-source implementations. The gap in the market is clear: **no platform currently delivers StatsBomb-level analytics to grassroots managers from a simple video recording**. VEO gives you video but not analytics. Hudl gives you analytics but at enterprise prices. Zone14 is closest but lacks the depth of tactical analysis.

Manager Mentor's opportunity is to combine open-source CV models into a pipeline that delivers professional-level analysis (heatmaps, formations, pass networks, pressing metrics, player ratings, basic xG) from a single fixed camera video -- at a price point accessible to grassroots clubs. The SoccerNet Game State Reconstruction task is essentially the benchmark for exactly this capability, and the 2024/2025 winning solutions provide a blueprint for the full pipeline.
