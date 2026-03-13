# HUDL Technical Architecture Deep Dive

**Date:** 8 March 2026
**Analyst:** Claude (for Manager Mentor project)
**Purpose:** Reverse-engineer Hudl's technical architecture, ML pipeline, infrastructure, and engineering stack to inform Manager Mentor's technical decisions.

---

## TABLE OF CONTENTS

1. [Product Suite Technical Breakdown](#1-product-suite-technical-breakdown)
2. [Computer Vision Pipeline](#2-computer-vision-pipeline)
3. [Player Detection and Tracking](#3-player-detection-and-tracking)
4. [Team Classification](#4-team-classification)
5. [Hudl Focus Camera System](#5-hudl-focus-camera-system)
6. [Automatic Highlight Generation](#6-automatic-highlight-generation)
7. [Data and Analytics Engine](#7-data-and-analytics-engine)
8. [Cloud Infrastructure](#8-cloud-infrastructure)
9. [Acquisitions and Technology Integration](#9-acquisitions-and-technology-integration)
10. [APIs and Integrations](#10-apis-and-integrations)
11. [Known Limitations](#11-known-limitations)
12. [Engineering Tech Stack](#12-engineering-tech-stack)
13. [Patents](#13-patents)
14. [Strategic Implications for Manager Mentor](#14-strategic-implications-for-manager-mentor)

---

## 1. PRODUCT SUITE TECHNICAL BREAKDOWN

### 1.1 Hudl Sportscode (Elite/Pro Analysis)

**What it is:** Desktop video coding and analysis software for professional performance analysts. Originally "SportsCode" by Australian company Sportstec (acquired 2015).

**Technical details:**
- **Platform:** macOS ONLY (no Windows, no web, no Linux)
- **Architecture:** Native macOS application with offline-first design
- **Video capture:** Multi-angle simultaneous capture from SDI/HDMI inputs
- **Coding engine:** Custom scripting language for building code windows (tag buttons)
- **Timeline:** Proprietary horizontal timeline with event markers, linked to video timestamps
- **Data model:** Events are stored as time-coded rows with customizable label columns
- **Export formats:** XML (Sportscode XML format), CSV, video clips
- **Integration points:** Can import Wyscout XML, StatsBomb data, GPS/tracking data overlays
- **Scripting system:** Requires near-programming knowledge to configure -- acts as a DSL (domain-specific language) for defining event taxonomies and conditional logic
- **Multi-angle sync:** Telestration on one angle auto-replicates across all synced angles via timecode alignment
- **Pricing:** ~$1,800/seat/year (Elite tier for pro clubs, Pro Review for lighter use)

**Key limitation:** macOS-only is a massive restriction. Many grassroots and amateur clubs use Windows.

### 1.2 Hudl Coda (Live Coding Companion)

**What it is:** iOS companion app for real-time event tagging during live matches.

**Technical details:**
- **Platform:** iOS (iPad/iPhone)
- **Architecture:** Mobile app that syncs with Sportscode timeline in real-time
- **Protocol:** Imports data live from iCoda into a Sportscode package
- **UI:** Customizable code windows with configurable colors, labels, and actions
- **Use case:** Secondary analyst in stadium can tag events from anywhere (stands, touchline)
- **Sync:** Everything tagged links to video for post-match review within minutes

### 1.3 Hudl Assist (Human-Powered Breakdown)

**What it is:** Outsourced human analysis service -- NOT an AI product despite marketing language.

**Technical details:**
- **Workforce:** Team of human analysts working 24/7 shifts
- **Workflow:** Coach uploads video -> Hudl analyst watches and tags -> stats and clips returned
- **Turnaround:** Standard priority (24-48 hrs), Express priority (faster)
- **Data output:** Filterable stats linked to video clips, custom and default reports
- **Sports covered:** Football, Basketball, Volleyball, Soccer, Lacrosse, Hockey, Baseball, Softball, Wrestling
- **Pricing:** $900-$1,500/team/season (on top of platform subscription)

**Critical insight for Manager Mentor:** This is Hudl's biggest vulnerability. They are scaling human labour, not AI. Their "automatic" stats are largely human-generated. An AI-first approach that delivers comparable or better results without human analysts is a massive cost and speed advantage.

### 1.4 Hudl IQ (American Football AI Data)

**What it is:** The closest thing Hudl has to genuine AI-driven analysis, focused on American football.

**Technical details:**
- **Data collection:** Computer vision tracks x,y positions of all 22 players at 30 frames per second
- **Tagging methodology:** Tags as FEW things as possible manually, then relies on an enrichment pipeline of data science models and logic-based rules
- **Formation classification:** ML model takes player x,y coordinates as inputs to classify offensive/defensive formations automatically
- **Route classification:** Models classify receiver routes from tracking data
- **Coverage classification:** Models classify defensive coverage schemes from player positions
- **Blitz detection:** Logic-based rules applied to player location data at event frames
- **Enrichment pipeline:** Models for EPA (Expected Points Added), defensive coverage, receiver routes, completion probability over expected -- all built into pipeline so 100% of data is available when game finishes
- **Turnaround:** Full dataset available ~18 hours after video is available
- **Data density:** Formation, route, coverage, blitz for every play automatically

**This is the most technically advanced thing Hudl does.** But it is American football specific, not soccer.

---

## 2. COMPUTER VISION PIPELINE

### 2.1 Overall Architecture

Hudl's CV pipeline is a **hybrid human-machine system**, not a fully automated one. Based on their engineering blog posts and the Sloan Sports Conference paper, the pipeline works as follows:

```
Video Input (Focus cameras / uploaded footage)
    |
    v
[Stage 1] Frame Extraction & Pre-processing
    |-- Distortion correction (for wide-angle lenses)
    |-- Camera calibration / homography estimation
    |
    v
[Stage 2] Object Detection
    |-- Player detection (bounding boxes)
    |-- Ball detection
    |-- Referee detection
    |
    v
[Stage 3] Player Tracking
    |-- Multi-object tracking across frames
    |-- Track association (maintaining identity across frames)
    |-- x,y coordinate extraction at 30fps
    |
    v
[Stage 4] Team Classification
    |-- Jersey color clustering
    |-- Assignment of tracks to teams
    |
    v
[Stage 5] Player Identification
    |-- Jersey number recognition (CNN-based)
    |-- Manual correction by human analysts
    |
    v
[Stage 6] Event Detection & Classification
    |-- Formation classification (ML model)
    |-- Route/coverage/blitz classification
    |-- Key moment detection (goals, fouls, etc.)
    |
    v
[Stage 7] Enrichment Pipeline
    |-- Statistical models (xG, EPA, etc.)
    |-- Logic-based rules
    |-- Physical metrics calculation
    |
    v
[Stage 8] Human QA / Correction
    |-- Analysts review and fix CV errors
    |-- Player ID corrections
    |-- Missing event additions
    |
    v
Output: Tagged, enriched dataset linked to video
```

### 2.2 Camera Calibration & Homography

Based on StatsBomb (now Hudl) blog posts and the open-source distortion correction tool:

- **Homography estimation:** Maps pixel coordinates in video frames to real-world pitch coordinates (x,y in meters)
- **Pitch model:** Uses a standard pitch template and finds the transformation matrix between observed pitch lines and the template
- **Requirement:** At least 4 characteristic points (pitch markings) must be visible in any frame
- **Distortion correction:** Open-source tool (github.com/hudl/HudlDistortionCorrectionTool) handles wide-angle lens distortion using chessboard calibration pattern
- **Camera model:** Assumes pinhole camera model for standard broadcast cameras
- **StatsBomb approach:** Uses AI for homography estimation in their data pipeline (referenced in their "Creating Better Data: AI & Homography Estimation" blog)

### 2.3 What ML Models Do They Use?

Based on research papers, job postings, and engineering blogs:

- **Object detection:** Likely DETR (Detection Transformer) or YOLO variants for player detection in crowded scenes. Their Sloan paper references "detection transformer" for tackling player detection in crowded contexts.
- **Deep learning framework:** Have evaluated multiple frameworks. Their Medium blog discussed Apache MXNet (Amazon's framework) alongside TensorFlow and PyTorch
- **Tracking:** Multi-object tracking (likely DeepSORT or similar) for maintaining player identity across frames
- **Classification:** CNNs for jersey number recognition, formation classification models taking coordinate inputs
- **Player Re-ID:** This remains difficult -- Hudl acknowledges jersey number reading is still hard for CV, especially with difficult-to-read numbers or large player groupings

### 2.4 R&D Team Structure

- ~12 Data Scientists and Engineers split across three teams
- Focus on computer vision problems in football (soccer)
- Work presented at internal engineering conferences
- Topics include: GANs applied to sports video, player tracking, formation detection
- Python is the primary language for R&D/data science work

---

## 3. PLAYER DETECTION AND TRACKING

### 3.1 Detection Approach

**For broadcast/professional footage:**
- Object detection network (DETR/Detection Transformer) for player bounding boxes
- Handles crowded scenes (scrum, corner kicks, goal celebrations) where traditional detectors struggle
- Ball detection runs alongside player detection
- Referee detection to separate officials from players

**For amateur/Hudl Focus footage:**
- Same pipeline but with added challenges: lower resolution (1080p vs 4K broadcast), wider angles, more distortion
- Focus cameras capture 180-degree panoramic view that gets stitched and cropped
- AI pan-and-zoom engine selects the active area of play from the panoramic capture

### 3.2 Tracking System

Based on the UX design case study by Mary Collins (Hudl contractor):

**Camera setup (in-stadium):**
- 3 cameras installed recording from 3 different angles
- Creates a map of the pitch that translates video into an x,y matrix
- Each player identified with a bounding box
- Algorithm records each player's movement as x,y data points

**Tracking pipeline:**
1. Player detected and assigned bounding box
2. Bounding box tracked across frames using MOT (multi-object tracking)
3. Homography applied to convert pixel positions to pitch coordinates
4. Track smoothing to handle occlusions and detection gaps
5. Physical metrics derived from position data (speed, distance, acceleration)

**Data output:**
- x,y position for every player, 30 times per second
- Speed, acceleration, deceleration per frame
- Cumulative distance over time
- Sprint detection (>24 km/h threshold)

### 3.3 Player Identification (The Hard Part)

Hudl acknowledges this is where the system struggles:

- **Jersey number recognition:** CNN-based, but only visible in a fraction of frames
- **Challenges:** Small, blurry numbers; occlusion; similar kit colors; difficult camera angles
- **Approach:** Multi-task learning -- holistic (entire number as one class) + digit-wise (individual digits)
- **Manual correction:** Hudl analysts watch the match and fix CV errors, correctly identifying players where the system fails
- **Re-identification:** When a player leaves and re-enters frame, the system must reconnect the track -- this is error-prone

**Key Hudl support page note:** There is a "Manually Identify Players" support article, confirming that automated player ID is not reliable enough to work without human correction.

---

## 4. TEAM CLASSIFICATION

### 4.1 Approaches Used in the Industry (Likely by Hudl)

Based on published research and Hudl's technical context:

**Color-Based Clustering (Baseline approach):**
- Extract dominant color from each player's bounding box crop
- K-Means clustering on color features (RGB or HSV space) to separate into 2-3 clusters (team A, team B, referees)
- Fast but fails with similar kit colors, shadows, or mixed-color kits

**Convolutional Autoencoder (Advanced approach):**
- Train CAE end-to-end on unlabeled data to learn feature representations
- Generalizes to any video without pre-training on specific team colors
- Does not require knowing team colors in advance

**Contrastive Learning (State-of-the-art):**
- Embedding network maximizes distance between representations of players on different teams
- Minimizes distance between representations of same-team players
- Works when jersey colors are not known a priori
- Hudl's R&D team is likely using or evaluating this approach given their focus on unsupervised methods

### 4.2 Hudl's Actual Implementation

From their tracking product design documentation:
- Initial automated team assignment from appearance features
- Human analysts verify and correct team assignments
- System struggles when teams have similar kit colors
- Goalkeeper differentiation is handled separately

---

## 5. HUDL FOCUS CAMERA SYSTEM

### 5.1 Product Line

| Model | Mount | Resolution | Lenses | Connectivity | Use Case |
|-------|-------|-----------|--------|-------------|----------|
| **Focus Outdoor** | Permanent (stadium) | HD 1080p | 180-degree panoramic | Ethernet/WiFi | Outdoor stadiums |
| **Focus Indoor** | Permanent (gym) | HD 1080p | 180-degree panoramic | Ethernet/WiFi | Indoor arenas |
| **Focus Flex** | Portable (tripod) | HD 1080p | 2 lenses, 180-degree | Built-in 4G cellular | Anywhere, portable |
| **Focus Point** | Fixed static | HD 1080p | Standard | Wired | Supplementary angle |

### 5.2 Technical Specifications (Focus Indoor/Outdoor)

- **Dimensions:** 10" W x 5" H x 8" D (25.4 x 12.7 x 20.3 cm)
- **Weight:** 7 lbs (3.2 kg)
- **Video:** HD 1080p, 180-degree panoramic
- **Audio:** Stereo microphones (announcers, whistles, crowd)
- **Processing:** On-device or cloud-based stitching of panoramic to tracked output
- **Firmware:** Self-updating over network
- **Recording trigger:** Scheduled automatic recording (no human operator)
- **Upload:** Automatic upload to Hudl cloud after recording

### 5.3 AI Pan-and-Zoom Engine

The key technology in Focus cameras:
- Records full 180-degree panoramic view continuously
- AI engine selects the relevant portion of the frame (where the action is)
- Produces a "directed" output that mimics a human camera operator
- Real-time ball tracking drives the pan-and-zoom decisions
- Player tracking algorithms inform framing for specific sports

### 5.4 Hudl Focus vs VEO Comparison

| Feature | Hudl Focus | VEO Cam 3 |
|---------|-----------|-----------|
| **Resolution** | HD 1080p | 4K |
| **Lens design** | 2 lenses, 180-degree | 2 x 4K lenses, 180-degree |
| **AI tracking** | Ball + player tracking | Ball tracking + AI director |
| **Mount** | Permanent or portable (Flex) | Portable (tripod) |
| **Connectivity** | 4G (Flex), Ethernet (fixed) | WiFi + optional 4G |
| **Auto-upload** | Yes (to Hudl platform) | Yes (to VEO platform) |
| **Livestream** | Yes (Hudl TV) | Yes (VEO Live) |
| **Platform lock-in** | Hudl ecosystem only | VEO platform only |
| **Key advantage** | Seamless Hudl integration, permanent mount | Higher resolution (4K), portable-first |
| **Key weakness** | Lower resolution | No deep analysis platform like Sportscode |
| **Pricing** | Bundled with Hudl subscription | Hardware purchase + subscription |

**Critical difference:** Hudl Focus is designed to lock customers into the Hudl ecosystem. VEO sells standalone hardware. Hudl's permanent mount strategy means switching costs are extremely high once installed.

---

## 6. AUTOMATIC HIGHLIGHT GENERATION

### 6.1 How Hudl Does It

Hudl's highlight generation is NOT fully automatic for most users:

**For Hudl IQ (American Football):**
- AI automatically tags every formation, route, coverage, blitz
- Automatic playlist generation based on tagged events
- Position-specific highlight templates filter clips automatically
- "AI-Generated Highlight Templates" filter and organize clips by skill or position

**For Hudl Assist (All Sports):**
- Human analysts tag key events (goals, fouls, turnovers, etc.)
- System auto-generates playlists from human-tagged events
- Players get notified when their highlights are ready

**For Self-Service Users:**
- Manual clip creation -- scrub through video, set start/end points
- Build custom playlists by dragging clips
- No automated highlight detection without paying for Assist

### 6.2 Notable Moments Detection (from Patents)

Based on Agile Sports Technologies patents:
- Method for detecting notable moments by analyzing motion levels in video
- Threshold-based: identifies frames/segments where motion exceeds a defined threshold
- Generates video clips with surrounding context (pre/post buffer)
- This is relatively primitive compared to modern event detection

### 6.3 What This Means for Manager Mentor

Hudl's highlight generation is either:
1. Human-powered (expensive, slow) via Assist
2. Rule-based/threshold-based (crude) via their patents
3. ML-driven but only for American football (Hudl IQ)

**There is a massive gap** for automated, AI-driven highlight generation in soccer/football at grassroots level.

---

## 7. DATA AND ANALYTICS ENGINE

### 7.1 Physical Metrics (from Tracking Data)

Hudl IQ and their tracking products compute these metrics from x,y position data at 30fps:

| Metric | Description | Method |
|--------|------------|--------|
| **Distance covered** | Total distance per player per game | Sum of frame-to-frame Euclidean distances |
| **Maximum speed** | Peak speed reached | Max of frame-to-frame velocity |
| **Sprint count** | Number of high-speed runs | Count of periods above 24 km/h threshold |
| **High-intensity distance** | Distance at speed tiers | Distance above configurable speed thresholds |
| **Acceleration events** | Count of high acceleration | Positive velocity change above threshold |
| **Deceleration events** | Count of high deceleration | Negative velocity change above threshold |
| **Cumulative distance** | Season-long distance totals | Aggregation across multiple games |
| **Heat maps** | Spatial distribution | 2D histogram of player positions |
| **Speed over time** | Time-series speed chart | Velocity per frame smoothed |

**14 aggregated physical metrics** are provided by default.

### 7.2 Tactical Metrics (Hudl IQ - American Football)

| Metric | Method |
|--------|--------|
| **Formation** | ML classification from x,y coordinates |
| **Receiver routes** | ML classification from player tracks |
| **Defensive coverage** | ML classification from defender positions |
| **Blitz detection** | Logic rules on player positions at snap |
| **EPA (Expected Points Added)** | Statistical model |
| **Completion % over expected** | Statistical model |
| **Play type** | Tagged by AI or manual |

### 7.3 StatsBomb Metrics (Soccer - Professional Only)

Acquired August 2024. Provides:
- **3,000+ events per game** manually tagged by human analysts
- **50+ additional metrics** across 40+ leagues
- **xG (Expected Goals):** Industry-leading model using shot location, impact height, goalkeeper position, defender positions, freeze-frame data
- **xPass:** Pass difficulty and completion probability
- **On-Ball Value (OBV):** Valuation of every on-ball action
- **Pressures:** Defensive pressing data (pioneered by StatsBomb)
- **Freeze frames:** Snapshot of all player positions at moment of shot
- **Pass footedness:** Which foot was used
- **xG model details:** Accounts for blocker/goalkeeper positioning, post-shot xG measures finishing skill

**Critical note:** StatsBomb data is human-tagged, not automated CV. Hudl has 2,500 data points manually added per game by human analysts. This is extremely labor-intensive.

### 7.4 ADI Multidirectional Metrics (Acquired October 2025)

- **Mechanical Power:** True cost of every sprint, cut, deceleration
- **Movement Intensity:** Beyond linear speed metrics
- **Step Symmetry:** Biomechanical balance indicator
- **Direction-change acceleration:** Measures speed-change AND direction-change
- **Data source:** Ingests from any GPS, LPS, or optical tracking system
- **Integration target:** Hudl Signal (athlete management platform)

### 7.5 Titan GPS Metrics

From the wearable device (acquired June 2025):
- 150+ metrics from GPS and motion sensors
- Top speed, total distance, effort, acceleration, deceleration
- Position-specific metrics (e.g., "Truck Stick" = speed x weight for linemen)
- Automated leaderboards
- Heat maps and 3D replays
- 7-hour battery life

---

## 8. CLOUD INFRASTRUCTURE

### 8.1 AWS Foundation

Hudl runs entirely on Amazon Web Services. Key details from the AWS case study:

**Compute:**
- Amazon EC2 instances for all workloads
- Auto Scaling groups for each microservice cluster
- Can spin up **2,000 servers just for video encoding** on a single Friday night during football season
- Handles 30% annual growth through elastic scaling

**Storage:**
- Amazon S3 for all video storage (billions of hours of footage)
- S3 Transfer Acceleration: 20%+ improvement in upload/encoding speeds

**Video Processing:**
- Ingests and encodes **39+ hours of HD video every minute** during peak sports seasons
- Serves 4.5 million coaches and athletes on 130,000 teams
- Partnership with Visionular for encoding optimization (see below)

**Data:**
- Amazon Redshift as data warehouse for internal analytics
- Amazon ElastiCache (Redis) for near-real-time data feeds
- MongoDB as primary application database

**Content Delivery:**
- Amazon CloudFront CDN for global video delivery
- Fastly (additional CDN layer)

**Messaging:**
- Amazon SQS for message queuing
- Amazon SNS for notifications
- RabbitMQ for inter-service messaging

**Search:**
- Amazon Elasticsearch Service

**DNS:**
- Amazon Route 53

### 8.2 Microservices Architecture

- Each service runs in its own Auto Scaling group
- Service discovery via Netflix Eureka (they have an open-source Go client: github.com/hudl)
- Load varies dramatically week-to-week (Friday night football vs off-season)
- Containerized with Docker
- Infrastructure as code with Terraform and Packer
- CI/CD via TeamCity

### 8.3 Video Encoding Pipeline

**Partnership with Visionular (AI-Driven Encoding):**
- Uses Visionular Aurora4 encoder (with Aurora5 planned)
- Three-step encoding: scene classification -> AI video enhancement -> content-adaptive encoding
- Aurora4 is 50% faster than x264
- Supports H.264/AVC, H.265/HEVC, and AV1 codecs
- AI-driven compression achieves significant bitrate savings while maintaining quality
- Hudl lowered rendition bitrates after Visionular integration (smaller files, same quality)
- Reduces storage footprint and makes high-quality video more accessible on slower connections

**Encoding flow:**
```
Video Upload (S3 Transfer Acceleration)
    |
    v
S3 Ingest Bucket
    |
    v
Encoding Workers (EC2 Auto Scaling)
    |-- Visionular Aurora4 encoder
    |-- Multiple rendition profiles (adaptive bitrate)
    |-- H.264 primary, AV1 for supported devices
    |
    v
S3 Output Bucket (encoded videos)
    |
    v
CloudFront / Fastly CDN
    |
    v
End Users (HLS/DASH adaptive streaming)
```

---

## 9. ACQUISITIONS AND TECHNOLOGY INTEGRATION

### 9.1 Complete Acquisition Timeline

| Year | Company | Technology Acquired | Current Status |
|------|---------|-------------------|---------------|
| **2015** | **Sportstec** (Australia) | SportsCode video coding software (macOS), elite performance analysis | Became Hudl Sportscode |
| **2019 May** | **Krossover** | AI video breakdown, auto-generated shot charts, human analyst network | Platform shut down June 2020, tech absorbed into Hudl Assist |
| **2019 Aug** | **Wyscout** (Italy) | Global soccer scouting database, 600+ competitions, player comparison tools, API | Became Hudl Wyscout, still operates |
| **2024 Aug** | **StatsBomb** | Advanced soccer event data, xG models, freeze-frame data, 3,000+ events/game | Became Hudl StatsBomb |
| **2025 Feb** | **Balltime** | AI volleyball analysis, jump heights, ball trajectories, 3D play views, 12,000+ teams | Being integrated |
| **2025 Mar** | **FastModel Sports** | Play diagramming (FastDraw), scouting (FastScout), recruiting (FastRecruit), 40,000+ customers | Being integrated |
| **2025 Jun** | **Titan Sports** | GPS wearable trackers, 150+ metrics, athlete tracking platform | Became Hudl Titan |
| **2025 Aug** | **SportContract** | Hockey video analytics, scouting platform, 28 pro leagues of data since 2003 | Being integrated |
| **2025 Oct** | **ADI (Athletic Data Innovations)** | Multidirectional movement metrics, mechanical power, step symmetry | Integrating into Hudl Signal |

Additional acquisitions include: Volleymetrics, Instat, WIMU, and others (18+ total acquisitions).

### 9.2 What Each Acquisition Brought Technically

**Sportstec (Sportscode):**
- Native macOS video coding engine
- Custom scripting language for analysis workflows
- Multi-angle video capture and sync
- Professional-grade timeline and event model
- Existing customer base of pro sports teams worldwide

**Krossover:**
- Pioneered outsourced human video breakdown at scale
- AI partnership with WSC Sports for automated clip detection
- Shot chart auto-generation technology
- Network of hundreds of human analysts
- **Most of this technology was sunsetted**, not integrated -- Hudl took the analyst workforce and customer base

**Wyscout:**
- Largest global soccer video and data library
- Scouting workflow tools (shadow teams, player comparison)
- API delivering data in JSON format with full schema
- 2,500 manually tagged data points per game
- 2,000 games tagged per week, 4 million new events weekly
- Competition database covering 600+ leagues

**StatsBomb:**
- Industry-leading xG model with freeze-frame data
- Defensive pressure metrics (pioneered)
- On-Ball Value (OBV) model
- Shot impact height data
- Post-shot xG for finishing skill measurement
- 3,000+ events per game, 50+ metrics, 40+ leagues
- Aggregated data API

**Balltime:**
- AI models for volleyball-specific metrics (jump height, ball trajectory, attack height, serve speed)
- 3D play visualization
- Fast analysis turnaround (minutes after match)
- 12,000+ volleyball teams, 125,000+ athletes

**Titan Sports:**
- GPS hardware design and manufacturing
- 150+ metric extraction from IMU/GPS sensors
- Coach-friendly visualization platform
- Leaderboard and gamification features
- Video + GPS overlay (stack-up) technology

**ADI:**
- Mechanical power calculation algorithms
- Movement intensity scoring
- Step symmetry analysis
- Hardware-agnostic (works with any GPS/LPS/optical system)
- Sports science domain expertise

---

## 10. APIS AND INTEGRATIONS

### 10.1 Available APIs

**Wyscout API:**
- RESTful API delivering JSON data
- Full schema definition for all resources
- Available to Wyscout platform subscribers
- Covers: players, teams, matches, competitions, events
- Used by clubs to feed data into their own systems (e.g., K.V. Mechelen case study)

**StatsBomb Aggregated API:**
- Serves aggregated data from the analysis platform
- Accessible through the Data Hub
- Documentation with metric names and definitions
- Allows direct integration into end user's software without CSV export
- Replaces manual download-and-import workflows

### 10.2 Data Formats

- **Sportscode XML:** Proprietary format for coded event data, importable into Sportscode
- **JSON:** Wyscout and StatsBomb API responses
- **CSV:** Export format for stats and reports
- **Video:** HLS/DASH for streaming, MP4 for downloads

### 10.3 Integration Partners

- **Stream Chat:** In-app messaging powered by Stream SDK (getstream.io partnership)
- **Visionular:** AI-driven video encoding
- **GPS providers:** ADI platform ingests data from any GPS/LPS/optical system

### 10.4 Open Source Projects (github.com/hudl)

| Project | Language | Purpose |
|---------|----------|---------|
| **Fargo** (Eureka Client) | Go | Netflix Eureka service discovery client |
| **HudlDistortionCorrectionTool** | Python | Camera calibration and lens distortion correction |
| **Server provisioning library** | Python | Automating server/cluster spin-up |
| **Room status display** | C# | Meeting room availability (internal tool) |
| **StackDriver metrics wrapper** | C# | Custom metrics to Google StackDriver |
| **Communal dashboard** | JavaScript | Information radiator / team dashboard |

The Distortion Correction Tool is the most technically relevant -- it confirms they do their own camera calibration using chessboard patterns and OpenCV.

---

## 11. KNOWN LIMITATIONS AND USER COMPLAINTS

### 11.1 Technical Limitations

| Area | Limitation | Impact |
|------|-----------|--------|
| **Player ID** | Jersey number recognition fails frequently | Requires manual human correction |
| **Team classification** | Struggles with similar kit colors | Human analysts must verify |
| **Processing time** | 18 hours for full Hudl IQ dataset | Not real-time or near-real-time |
| **Resolution** | Focus cameras at 1080p vs VEO's 4K | Lower quality for zoom-in analysis |
| **Platform** | Sportscode is macOS-only | Excludes majority of amateur market |
| **AI scope** | Hudl IQ only works for American football | No automated soccer/football analysis at this depth |
| **Assist dependency** | Most analysis requires paid human service | Expensive, slow, not scalable |
| **Mobile app** | Significantly less functionality than desktop | Coaches on sidelines get inferior experience |

### 11.2 User Complaints (from Reviews)

**Performance:**
- "Hudl is literally the worst" -- frequent downtime complaints
- "Slow" and "down all the time"
- App "continually freezes" after updates
- "There's always something freezing up" on app and browser

**Mobile App:**
- Android app "constantly closes", "laggy", "buggy"
- iPhone app "crashes consistently on startup"
- WiFi-only upload (no mobile data)
- iPad zoom bug acknowledged but deprioritized

**Video Editing:**
- Editor described as "buggy"
- Each splice forces restart at beginning of video
- Stats tagging function produces "completely wrong" stats with no way to review/correct

**Pricing:**
- "Turned their backs on youth sports by jacking up annual pricing from $99 to $400 minimum"
- Youth coaches feel abandoned
- "Really expensive" at $1,800/seat for Sportscode

**Cross-Platform:**
- "Confusing going back and forth between mobile app and browser"
- Functionality differs between platforms
- Desktop has slow-mo, rewind, zoom that mobile lacks

**Learning Curve:**
- Sportscode scripting "requires almost a basic programming course"
- Platform described as "complex and confusing to learn"

### 11.3 Structural Weaknesses

1. **Human-dependent scaling:** Assist model requires hiring proportionally more analysts as customer base grows
2. **Fragmented product portfolio:** 15+ sub-products creates confusion for buyers
3. **Acquisition integration debt:** Rapid M&A means disparate tech stacks being stitched together
4. **US-centric AI:** Hudl IQ only covers American football -- soccer AI lags significantly
5. **No real-time capability:** Nothing works during a live match (except manual Coda tagging)
6. **No public developer API:** Core Hudl platform has no public API for third-party developers

---

## 12. ENGINEERING TECH STACK

### 12.1 Confirmed Technologies (from StackShare, job postings, GitHub)

**Backend:**
- **C# / .NET / .NET Core** -- Core backend language
- **Go (Golang)** -- Service discovery, infrastructure tools
- **Python** -- Data science, ML, server automation
- **Node.js** -- Backend services

**Frontend:**
- **React** -- Primary frontend framework
- **TypeScript** -- Frontend language
- **JavaScript** -- Legacy frontend
- **React Native** -- Mobile apps
- **Redux** -- State management
- **GraphQL** -- API query layer
- **Sass** -- CSS preprocessing
- **Webpack** -- Build tooling

**Databases:**
- **MongoDB** -- Primary application database
- **Amazon Redshift** -- Data warehouse
- **Amazon ElastiCache (Redis)** -- Caching and real-time feeds
- **Amazon Elasticsearch** -- Search

**Infrastructure:**
- **AWS** -- Sole cloud provider
- **Docker** -- Containerization
- **Amazon EC2** -- Compute
- **Amazon S3** -- Object storage
- **Amazon EC2 Container Service (ECS)** -- Container orchestration
- **Terraform** -- Infrastructure as code
- **Packer** -- Machine image building

**Messaging / Queuing:**
- **RabbitMQ** -- Message broker
- **Amazon SQS** -- Queue service
- **Amazon SNS** -- Notification service

**CI/CD & DevOps:**
- **TeamCity** -- CI/CD pipeline
- **Git / GitHub** -- Version control
- **Sentry** -- Error tracking
- **Sumo Logic** -- Log management
- **Codecov** -- Code coverage

**ML / Data Science:**
- **Python** -- Primary ML language
- **Apache MXNet** -- Evaluated (Amazon's preferred framework)
- **TensorFlow / PyTorch** -- Likely used alongside MXNet
- **OpenCV** -- Computer vision processing (confirmed by distortion correction tool)

**Web Servers:**
- **NGINX** -- Reverse proxy / web server
- **Microsoft IIS** -- Legacy .NET hosting

**CDN:**
- **Amazon CloudFront** -- Primary CDN
- **Fastly** -- Additional CDN layer

**Service Discovery:**
- **Netflix Eureka** -- Via custom Go client (open-sourced)

**Monitoring:**
- **Google StackDriver** -- Custom metrics (via open-source C# wrapper)

### 12.2 Tech Stack Summary

Hudl's stack is fundamentally a **C# / .NET backend with React frontend, running on AWS microservices with MongoDB**. Their ML pipeline is Python-based. They are NOT using Kubernetes (they use ECS), and their CI is TeamCity rather than GitHub Actions.

This is a mature, somewhat legacy stack. The C# / .NET foundation is unusual for a data-heavy sports tech company and reflects their Nebraska engineering roots. Most modern competitors would choose Python/Go backends with Kubernetes.

---

## 13. PATENTS

### 13.1 Assigned to Agile Sports Technologies, Inc.

Based on Justia Patents and patent analysis:

**Patent 1: Team Communication Platform**
- Combines messaging, video, testing, reporting, workflow diagramming, presentations, and performance analysis into a portal system
- Made mobile through synchronization services
- Covers the core Hudl platform concept

**Patent 2: Digital Video Editing and Playback System**
- Video processor receives video segments from multiple sources with synchronization information
- Covers multi-angle video sync and editing
- Relevant to Sportscode's multi-angle workflow

**Patent 3: Notable Moments Detection**
- Method for detecting and outputting notable moments of a performance
- Receives input video stream
- Determines moments with motion greater than a threshold value
- Generates video clips with surrounding context (buffer before and after)
- This is a relatively simple motion-threshold patent, not sophisticated CV

### 13.2 Patent Strategy Assessment

From Patent Forecast analysis (March 2021):
- "Hudl introduces a new Sports Analytics product without patents. Where's the strategy?"
- Hudl has a **surprisingly thin patent portfolio** for a company of their size
- Only ~3 patents directly assigned to Agile Sports Technologies
- They may rely on trade secrets rather than patents for their ML/CV innovations
- Competitor Pixellot has a much more aggressive patent strategy

**Implication for Manager Mentor:** The patent landscape in sports video analysis is NOT heavily locked down by Hudl. Their notable moments patent is basic (motion threshold) and would be easy to design around with modern ML approaches. There is room to build and even patent more sophisticated approaches.

### 13.3 Legal Activity

- **QwikCut LLC v. Agile Sports Technologies (2026):** Active lawsuit in NJ District Court -- competitor suing Hudl (details under seal)
- This confirms the competitive tension in the space and potential IP disputes

---

## 14. STRATEGIC IMPLICATIONS FOR MANAGER MENTOR

### 14.1 Where Hudl's Architecture is Weak (Our Opportunities)

| Hudl Weakness | Manager Mentor Advantage |
|--------------|-------------------------|
| Human analysts (Assist) = expensive, slow | Fully automated AI pipeline = fast, cheap, scalable |
| 18-hour turnaround for Hudl IQ | Target <30 minutes for basic analysis, real-time for live |
| Hudl IQ only covers American football | Build soccer/football-first AI from day one |
| 1080p Focus cameras | Support any input: phone cameras, 4K, GoPro, broadcast |
| macOS-only Sportscode | Web-first, cross-platform from day one |
| No public API | API-first architecture, enable third-party ecosystem |
| C#/.NET legacy backend | Modern Python/Go backend optimized for ML workloads |
| ECS (no Kubernetes) | Kubernetes-native for better scaling and portability |
| $400+ minimum pricing | Target under $200/season for grassroots |
| No real-time analysis | Build real-time event detection as core capability |

### 14.2 Where Hudl's Architecture is Strong (Respect & Learn)

| Hudl Strength | What We Should Learn |
|--------------|---------------------|
| AWS Auto Scaling for video encoding | Design for elastic scaling from day one |
| Visionular AI encoding partnership | Invest in encoding quality early (AV1, HEVC) |
| Enrichment pipeline concept | Build our own enrichment pipeline (models run after tracking) |
| 30fps tracking data | Match or exceed this framerate for tracking |
| Homography estimation | Implement robust camera calibration for pitch mapping |
| Distortion correction | Handle wide-angle lenses properly |
| StatsBomb xG model sophistication | Build xG with freeze-frame data as a differentiator |
| Wyscout data density (2,500 events/game) | Set high bar for event detection density |

### 14.3 Technical Architecture Recommendations

Based on this analysis, Manager Mentor should:

1. **Use Python + Go backend** (not C#) -- optimized for ML serving and high-concurrency video processing
2. **Kubernetes on AWS or GCP** -- not ECS, for better portability and scaling
3. **YOLO v8/v9 + DeepSORT** for detection and tracking -- proven in sports CV literature
4. **Transformer-based models** for event detection and classification
5. **Automated homography** using pitch line detection (no manual calibration)
6. **Color clustering + contrastive learning** for team classification
7. **WebRTC for real-time** -- enable live analysis during matches
8. **HLS/DASH adaptive streaming** with AV1 encoding for video delivery
9. **PostgreSQL + TimescaleDB** for time-series tracking data (not MongoDB)
10. **Public REST + GraphQL API** from day one

---

## SOURCES

### Hudl Official
- https://www.hudl.com/products/sportscode
- https://www.hudl.com/products/focus
- https://www.hudl.com/products/focus/details
- https://www.hudl.com/products/assist
- https://www.hudl.com/products/football-iq
- https://www.hudl.com/products/coda
- https://www.hudl.com/products/statsbomb
- https://www.hudl.com/products/titan
- https://www.hudl.com/products/adi
- https://www.hudl.com/blog/how-hudl-iq-creates-football-data
- https://www.hudl.com/blog/hudl-iq-calculating-physical-metrics
- https://www.hudl.com/blog/hudl-statsbomb-press-release-en
- https://www.hudl.com/blog/upgrading-expected-goals
- https://www.hudl.com/blog/the-inside-view-of-hudls-wyscout-acquisition
- https://www.hudl.com/blog/krossover-joins-hudl
- https://www.hudl.com/blog/hudl-expands-volleyball-focus-through-game-changing-acquisition-of-balltime
- https://www.hudl.com/blog/hudl-acquires-fastmodel-sports
- https://www.hudl.com/blog/hudl-acquires-sportcontract
- https://www.hudl.com/blog/hudl-acquires-adi-multidirectional-metrics
- https://www.hudl.com/blog/hudl-acquires-titan-sports
- https://www.hudl.com/blog/learn-more-about-the-statsbomb-iq-api
- https://www.hudl.com/blog/latest-insight-updates-player-tracking-enhancements

### Engineering & Technical
- https://medium.com/hudl-data-science/tagged/computer-vision
- https://medium.com/hudl-data-science/tagged/machine-learning
- https://medium.com/hudl-data-science/hudl-distortion-correction-tool-dedeff2efc0
- https://medium.com/in-the-hudl/how-hudl-and-visionular-are-upping-the-quality-game-4469ff98d170
- https://medium.com/in-the-hudl/running-hudls-first-engineering-conference-26f58e11495f
- https://hudl.github.io/ (Open Source at Hudl)
- https://github.com/hudl
- https://github.com/hudl/HudlDistortionCorrectionTool
- https://stackshare.io/hudl/hudl-web-stack
- https://himalayas.app/companies/hudl/tech-stack
- https://aws.amazon.com/solutions/case-studies/hudl/
- https://visionular.ai/how-we-help-hudl-up-their-video-quality-game/

### Research Papers
- https://www.sloansportsconference.com/research-papers/using-computer-vision-and-machine-learning-to-automatically-classify-nfl-game-film-and-develop-a-player-tracking-system
- https://blogarchive.statsbomb.com/articles/soccer/creating-better-data-ai-homography-estimation/
- https://blogarchive.statsbomb.com/articles/soccer/sb-labs-camera-calibration/
- https://www.nature.com/articles/s41598-023-36657-5

### Patents & Legal
- https://patents.justia.com/assignee/agile-sports-technologies-inc
- https://www.patentforecast.com/2021/03/04/hudl-introduces-a-new-sports-analytics-product-without-patents-what-is-its-strategy/

### UX & Design
- https://www.maryecollins.com/hudl-tracking

### Reviews & Complaints
- https://texags.com/forums/9/topics/2855815
- https://www.g2.com/products/hudl-sportscode/reviews
- https://sourceforge.net/software/product/Hudl-Sportscode/
- https://justuseapp.com/en/app/412223222/hudl/reviews

### Comparisons
- https://www.arsturn.com/blog/veo-3-vs-flow-unpacking-the-hype-finding-the-right-sports-camera-for-you
- https://www.sportsvisio.com/stories/sportsvisio-vs-hudl-2026-comparison-for-youth-teams
