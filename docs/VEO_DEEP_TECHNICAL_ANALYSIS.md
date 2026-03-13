# VEO Technologies Deep Technical Analysis
> **Date**: 2026-03-08
> **Purpose**: Reverse-engineer VEO's architecture to build a superior rival product (Manager Mentor)
> **Sources**: VEO patents, job postings, help docs, user forums, competitor analyses, engineering blog, API docs

---

## Table of Contents
1. [Company Overview](#1-company-overview)
2. [Camera Hardware](#2-camera-hardware)
3. [Computer Vision Pipeline](#3-computer-vision-pipeline)
4. [Player Detection Approach](#4-player-detection-approach)
5. [Team Classification](#5-team-classification)
6. [Auto-Follow / Virtual Camera](#6-auto-follow--virtual-camera)
7. [Tagging & Event Detection](#7-tagging--event-detection)
8. [Infrastructure & Processing Pipeline](#8-infrastructure--processing-pipeline)
9. [API / SDK](#9-api--sdk)
10. [Patents](#10-patents)
11. [Known Limitations & User Complaints](#11-known-limitations--user-complaints)
12. [Tech Stack (from Job Postings)](#12-tech-stack-from-job-postings)
13. [Competitive Landscape](#13-competitive-landscape)
14. [Attack Surface for Manager Mentor](#14-attack-surface-for-manager-mentor)

---

## 1. Company Overview

| Attribute | Detail |
|-----------|--------|
| **Founded** | 2015, Copenhagen, Denmark |
| **Founder** | Henrik Teisbaek |
| **Employees** | ~380+ across 50+ nationalities, 5 offices |
| **Funding** | Series C: $80M (ATP lead, Ventech, Seed Capital, Chr. Augustinus Fabrikker) |
| **Previous Rounds** | Series B: EUR 20M (~$24.5M, Jan 2021), Series A: Ventech (May 2019) |
| **Revenue** | Estimated ~$72M (from earlier research) |
| **Market Position** | Global leader in AI sports camera for amateur/semi-pro, strong D2C model |
| **HQ** | Agesrogade 6 C, 4. 2100 Copenhagen East |
| **Products** | Veo Cam 3, Veo Cam 3 5G, Veo Go (smartphone app), Veo Editor, Veo Analytics, Veo Live, Veo Player Spotlight |

---

## 2. Camera Hardware

### Veo Cam 3 (Current Generation)

| Spec | Detail |
|------|--------|
| **Lenses** | Dual 4K fisheye lenses (labeled: 100 FOV, 30 FPS, 4K HDR) |
| **Field of View** | 180-degree panoramic (two lenses stitched) |
| **Raw Capture** | Each lens captures at 4K resolution (3840x2160) at 30fps |
| **Output Resolution** | 1080p (after stitching, dewarping, and follow-cam crop) |
| **HDR** | Yes - High Dynamic Range recording (new in Cam 3) |
| **Dimensions** | Width: 209mm, Height: 63mm (excluding mount), Depth: 189mm |
| **Weight** | 1.25 kg |
| **Battery** | ~4.5 hours (recording + livestreaming) |
| **Storage** | Internal (non-expandable), upload regularly to free space |
| **Cooling** | Active cooling system (new in Cam 3) |
| **Connectivity** | WiFi, Hotspot, Ethernet (via USB-C dongle), 4G/5G nano SIM (5G model only) |
| **Ports** | USB-C 3.2 Gen 2 |
| **Charging** | Can record while charging |
| **Construction** | Durable, weather-resistant |

### How the 180-Degree Panoramic Works

1. **Dual Fisheye Capture**: Two 4K fisheye lenses are positioned inside the housing, angled to collectively cover a 180-degree horizontal field of view across the entire playing field
2. **Synchronized Streams**: Both lenses record simultaneously at 30fps, producing two synchronized 4K video streams
3. **Stitching**: The two streams are stitched together into a single panoramic video during processing (NOT in real-time on the camera for the non-live version)
4. **Dewarping**: Fisheye distortion is corrected during the stitching process using calibrated lens profiles
5. **Raw Footage**: The raw footage is the full 180-degree panoramic view - essentially a very wide, distorted image that gets corrected and cropped

### Evolution Across Generations

| Feature | Cam 1 | Cam 2 | Cam 3 |
|---------|-------|-------|-------|
| Lenses | Dual 4K fisheye | Dual 4K fisheye | Dual 4K fisheye + HDR |
| Processing | All cloud | Some on-camera | More on-camera |
| Livestream | No | Yes | Yes (improved) |
| Stabilization | None | Basic | SteadyView (software) |
| Upload Speed | Very slow | Faster | Faster |
| 5G Option | No | No | Yes (5G model) |

### SteadyView (Software Stabilization - Cam 3 only)
- Not hardware-based (no gimbal)
- Works by detecting **stationary background objects** (houses, trees, street signs)
- Tracks these reference points frame-by-frame
- Adjusts each frame slightly to keep the background fixed
- Unique to VEO - no other sports camera does this in software
- Developed by Marius Simonsen (VEO engineer)

### Key Hardware Insight
**VEO does NOT do on-device AI inference for the follow-cam.** The camera captures raw panoramic footage. All AI processing (follow-cam, event detection, analytics) happens in the cloud AFTER upload. The camera itself is essentially a dumb dual-4K recorder with WiFi/5G connectivity. The Cam 2 introduced "some processing on camera" which appears to be basic stitching/compression to speed up uploads, NOT AI inference.

---

## 3. Computer Vision Pipeline

### VEO's AI Pipeline (Three Core Components)

From VEO's own engineering blog ("The Road to Simplicity: Building Veo Go"):

> "The AI pipeline they developed includes three core components:
> 1. **Ball Tracking** - training a model to keep track of where the ball is
> 2. **Event Detection** - finding timestamps in the recording where specific events occur
> 3. **Follow Cam** - the broadcast experience you get, where a virtual camera follows the action"

### Processing Flow (Reconstructed)

```
[Camera Records] -> [Raw Dual 4K Fisheye Streams]
        |
        v
[Upload to Cloud] (WiFi/5G/Ethernet, 1-12+ hours)
        |
        v
[Cloud Processing Pipeline]
    |
    |-- Step 1: Video Stitching (merge 2 streams into panoramic)
    |-- Step 2: Fisheye Dewarping (lens distortion correction)
    |-- Step 3: SteadyView Stabilization (background point tracking)
    |-- Step 4: Player Detection (object detection model)
    |-- Step 5: Ball Tracking (specialized ball detection model)
    |-- Step 6: Team Classification (jersey color / shirt number)
    |-- Step 7: Follow-Cam Generation (virtual camera crop & pan)
    |-- Step 8: Event Detection (goals, shots, set pieces)
    |-- Step 9: Analytics Generation (heatmaps, pass maps, possession)
    |-- Step 10: Clip Rendering (auto-generated highlight clips)
        |
        v
[Available in Veo Editor]
    Follow-cam: ~1 hour after upload complete
    AI clips: additional time after follow-cam
    Full analytics: additional processing
    Player Spotlight: separate detection pass
```

### Processing Times (from VEO Help Center)
- **Follow-cam**: Typically available within **1 hour** after upload completes
- **AI-detected events**: Available with the follow-cam
- **Auto-rendered AI clips**: Additional time after follow-cam
- **Full analytics (heatmaps, pass maps)**: Requires sides/periods confirmation, then additional processing
- **Player Spotlight (jersey detection)**: Separate processing pass

### What Models They Likely Use

VEO has not published specific model architectures, but based on job postings, patents, and inference:

1. **Player/Object Detection**: Almost certainly a YOLO variant or similar single-stage detector (given their Python/PyTorch stack and the industry standard). Job postings ask for "computer vision" and "deep learning" experience.

2. **Ball Tracking**: Their patent describes detecting "concentration of action" which suggests they may use player clustering as a proxy for ball location, supplemented by direct ball detection. The ball is notoriously hard to detect at wide-angle distances.

3. **Jersey Number Detection**: Player Spotlight uses "shirt number detection" - likely OCR or a custom CNN trained on sports jersey numbers. They detect jersey numbers and then allow users to "assign your lineup" by linking players to detected numbers.

4. **Event Classification**: Likely a combination of rule-based heuristics (ball near goal + sudden player movement = shot) and trained classifiers.

### No Published Academic Papers

Despite extensive searching, VEO Technologies has **no published academic papers** on their computer vision methods. This is deliberate - their IP is their competitive advantage. They rely on patents rather than publications.

---

## 4. Player Detection Approach

### Handling Wide-Angle Distortion

VEO's approach to the fisheye distortion problem:

1. **Dewarping First, Then Detect**: The panoramic video is dewarped (fisheye correction applied) BEFORE player detection runs. This means the detection model sees a relatively normal (though wide) image.

2. **Resolution Challenge**: Even at 4K per lens, when you stitch and dewarp a 180-degree view of a full-size football pitch, each player is only a small number of pixels tall. At the far side of the pitch, players might be 20-40 pixels tall.

3. **Detection at Scale**: VEO likely uses multi-scale detection or a high-resolution input with large anchor boxes to handle the varying player sizes across the pitch (players near the camera are large, players far away are tiny).

### Player Counting Reliability

Based on user reports and our own testing:
- VEO detects most players most of the time on the near half
- Far-side players are harder to detect reliably
- Goalkeepers at extreme distance are frequently missed
- Overlapping players (tackle situations, set pieces) cause tracking failures
- The "Player Spotlight" feature suggests they achieve ~80-90% jersey number detection accuracy (they allow manual correction)

### What VEO Gets Right
- Consistent detection on the camera-side half of the pitch
- Reasonable tracking continuity (players maintain IDs)
- Jersey number detection is functional (though not perfect)

### What VEO Gets Wrong
- Far-side detection is weaker
- Crowded scenes (corners, free kicks) cause issues
- Goalkeepers at distance are often missed
- No referee-specific tracking exposed to users
- Ball tracking can lose the ball for extended periods

---

## 5. Team Classification

### VEO's Approach

Based on Player Spotlight documentation and user-facing features:

1. **Jersey Color-Based**: VEO uses jersey color to distinguish teams. This is evident from their requirement to "confirm sides and periods" - the system needs to know which team is on which side to correctly assign home/away.

2. **Sides and Periods Confirmation**: Users MUST confirm:
   - Which side of the pitch each team starts on
   - When half-time occurs (period change = teams swap sides)
   - This is a **manual step** that triggers correct team assignment

3. **Jersey Number Detection**: Player Spotlight detects shirt numbers and uses them for individual player identification WITHIN a team. The team assignment itself comes from positional data + color.

4. **No Pure ML Team Classification**: The requirement for manual "sides and periods" confirmation strongly suggests VEO does NOT have a fully automated team classifier. They likely:
   - Cluster players by color into two groups
   - Use the user's sides/periods input to label which cluster is "home" and which is "away"
   - If a user doesn't confirm sides, the analytics don't work properly

### User Workflow
1. Record game
2. Upload to cloud
3. VEO processes follow-cam and basic events
4. User opens recording in Editor
5. User confirms sides (which team starts on which side)
6. User confirms periods (when halves start/end)
7. VEO then generates team-specific analytics
8. User assigns lineup (maps jersey numbers to player names)

### Implications for Manager Mentor
This is a significant gap we can exploit. A fully automated team classifier that doesn't require manual input would be a major differentiator. Our current SigLIP + UMAP + KMeans approach with HSV Hue clustering is already more automated than VEO's system.

---

## 6. Auto-Follow / Virtual Camera

### How the AI Cameraman Works

VEO's follow-cam is NOT a physical camera movement. It's a **virtual crop from the panoramic footage**:

1. **Full Panoramic Capture**: The camera records the entire 180-degree view at all times
2. **AI Determines Crop Region**: After upload, AI analyzes each frame to determine where the "action" is
3. **Virtual Pan/Zoom**: A 1080p window is cropped from the panoramic footage and moved smoothly to follow the action
4. **Output**: A broadcast-style video that appears to pan and zoom like a human operator

### What Triggers Pan/Zoom Decisions

Based on VEO's patent (WO2019141813A1) and product descriptions:

1. **Ball Position (Primary)**: "The camera automatically follows the ball" - ball tracking is the primary signal
2. **Player Concentration**: Patent describes detecting "concentration of action" - clusters of players indicate where play is happening
3. **Predictive Analysis**: The system uses "predictive analysis" to anticipate where play will move
4. **Contextual Awareness**: Understands game context (kickoff, corner, etc.) to frame appropriately
5. **Adaptive Framing**: Adjusts the crop window size based on the spread of action

### Patent-Described Method (WO2019141813A1)

The patent describes:
- Capturing images via a video camera system producing a video stream
- Digitally processing the stream to "continuously identify a detected concentration of action within the boundaries of the field"
- The "concentration of action" can be based on:
  - Detecting the presence of a sports game ball
  - Detecting the concentration/clustering of players
  - Combining both signals
- The detected area of interest moves dynamically as play progresses

### Follow-Cam Quality
- The output is 1080p (cropped from higher-res panoramic)
- Smooth camera movements (not jerky)
- Occasionally loses the action during fast transitions (goal kicks to counter-attacks)
- Can be confused by referee movements near sideline
- Users can switch between follow-cam and interactive (manual pan) views

### Veo Go (Smartphone Version)
- Uses iPhones as cameras instead of the VEO Cam
- The same AI pipeline runs in the cloud
- "Uses advanced AI to stitch your iPhone feeds into a single, stunning panoramic view that follows the ball automatically"
- Requires multiple phones positioned to cover the pitch

---

## 7. Tagging & Event Detection

### Automatically Detected Events

VEO's AI detects the following events automatically:

**Football (Soccer)**:
- Goals
- Kickoffs
- Half-time
- Corners
- Free kicks
- Goal kicks
- Penalty kicks
- Shots on goal
- Set pieces (general)

**Other Sports** (varies by sport):
- Scrums (rugby)
- Sport-specific events

### How Events Are Detected

1. **Rule-Based + ML Hybrid**: VEO likely uses a combination of:
   - Ball position relative to goal area (shots, goals)
   - Game state detection (kickoff = ball at center, players in formation)
   - Player clustering patterns (corner = cluster near corner flag)
   - Temporal patterns (goal = celebration movement after ball in goal area)

2. **Event Detection Accuracy**: From user reports:
   - Goals: High accuracy (clear ball-in-net signal)
   - Kickoffs: High accuracy (distinct formation)
   - Shots: Moderate accuracy (some false positives/negatives)
   - Corners/Free kicks: Moderate accuracy
   - Tackles/Passes: NOT auto-detected (these require manual tagging)

### What Requires Manual Input

VEO does NOT automatically detect:
- Individual passes
- Tackles
- Interceptions
- Dribbles
- Player-specific actions (who scored, who assisted)

Users can:
- Create custom tags/clips manually
- Assign players to events
- Add comments to highlights
- Duplicate and customize AI-generated clips

### Veo Analytics (Add-on - ~$26/month+)

The analytics package provides:
- **Shot Map**: Where shots originated, which resulted in goals (for both teams)
- **Pass Strings**: Count of consecutive passes before losing possession
- **Passing Flow**: Where passing patterns thrive and break down
- **Heatmaps**: Team positioning over time
- **Possession Location**: Where on the pitch possession occurs
- **2D Radar**: Top-down view of player positions and movements
- **Match Stats**: Passes, shots, possession %, free kicks breakdown
- **Player Positioning**: Team shape and positional trends

### Veo Player Spotlight (Add-on)

- **Shirt Number Detection**: AI detects jersey numbers
- **Individual Tracking**: Creates per-player highlight reels automatically
- **Player Moments**: Shows key moments for each player based on proximity to ball
- **Settings**: Distance to ball, Time on ball, Time before/after moment (adjustable)
- **Requires**: Sides and periods confirmation + lineup assignment

---

## 8. Infrastructure & Processing Pipeline

### Cloud-First Architecture

VEO is fundamentally a **cloud-processing company**. The camera is the capture device; all intelligence runs in the cloud.

### Confirmed Technology

From job postings, subprocessors page, and API documentation:

**Backend/Platform**:
- **Salesforce** - CRM
- **Zuora** - Subscription billing
- **Adyen** - Payment processing
- Python microservices (from job postings mentioning "Python microservices")
- React frontends (from job postings mentioning "React frontends")

**APIs**:
- Production API: `https://api.veo.co.uk/api`
- Camera Service API: `https://api.prod.camera.veo.co`
- Token API: `https://tokenapi.veo.co.uk/oauth2/token`
- Swagger documentation available at `https://api.veo.co.uk/swagger/ui/index`

**ML/AI Stack** (from job postings):
- Python (primary ML language)
- Machine learning frameworks (TensorFlow/PyTorch - standard for the role)
- Computer vision libraries
- "Comprehensive skills in Python and machine learning libraries"

**Cloud Provider**: Not explicitly confirmed, but evidence suggests:
- Likely AWS or GCP (standard for video processing at scale)
- The subprocessors page exists but requires JavaScript rendering
- Camera service API at `api.prod.camera.veo.co` suggests a production cloud environment

### Processing Pipeline Architecture (Reconstructed)

```
[Veo Cam 3]
    |
    | (WiFi/5G/Ethernet upload)
    v
[Ingestion Service]
    |-- Receives raw dual-stream video
    |-- Validates format/integrity
    |-- Queues for processing
    |
    v
[Video Processing Pipeline]
    |
    |-- [Stitching Service]
    |     |-- Merges two 4K fisheye streams
    |     |-- Applies lens calibration profiles
    |     |-- Outputs single panoramic video
    |
    |-- [Stabilization Service (SteadyView)]
    |     |-- Detects background reference points
    |     |-- Computes frame-to-frame adjustments
    |     |-- Outputs stabilized panoramic
    |
    |-- [ML Inference Pipeline]
    |     |-- Player Detection (per-frame)
    |     |-- Ball Tracking (per-frame)
    |     |-- Jersey Number Detection
    |     |-- Event Classification
    |     |
    |     v
    |   [Tracking Data Store]
    |
    |-- [Follow-Cam Generator]
    |     |-- Reads ball position + player clusters
    |     |-- Computes smooth virtual camera path
    |     |-- Renders 1080p follow-cam video
    |
    |-- [Analytics Engine]
    |     |-- Heatmaps, pass maps, possession
    |     |-- Shot map, pass strings
    |     |-- 2D radar positions
    |
    |-- [Clip Renderer]
          |-- Auto-generates highlight clips
          |-- Renders AI-detected events as video clips
    |
    v
[CDN / Video Hosting]
    |-- Serves follow-cam video
    |-- Serves panoramic (interactive) video
    |-- Serves clips
    |
    v
[Veo Editor (Web/App)]
    |-- User watches, clips, shares
    |-- Manual tagging interface
    |-- Analytics dashboards
```

### Upload & Processing Times

| Stage | Duration |
|-------|----------|
| Upload (WiFi) | 1-12+ hours depending on connection speed |
| Upload (5G) | Faster, but still significant for full game |
| Follow-cam processing | ~1 hour after upload complete |
| AI clips rendering | Additional time after follow-cam |
| Analytics (after sides confirmation) | Minutes to hours |
| Player Spotlight | Additional processing pass |
| **Total end-to-end** | **Typically 24-48 hours from game end to full analysis** |

### Edge vs Cloud

- **Camera (Edge)**: Minimal processing. Records, compresses, uploads. Cam 2+ does "more processing on camera" for faster uploads (likely basic encoding/compression, NOT AI).
- **Cloud (Everything Else)**: All AI inference, stitching, rendering, analytics happen in the cloud.
- **Live Streaming**: When livestreaming, some processing must happen in near-real-time. The live stream provides the panoramic/follow-cam view with basic tracking, but full analytics are still cloud-processed after the fact.

---

## 9. API / SDK

### VEO Developer API (developer.veo.co.uk)

VEO has a documented API, but it's **restricted access**:

**Authentication**:
- OAuth2 token-based
- Production: `https://tokenapi.veo.co.uk/oauth2/token`
- UAT: `https://tokenapiuat.veo.co.uk/oauth2/token`
- Requires Client ID + Secret (must be obtained from VEO directly)
- "To get started, you'll need a new application which can only be created by one of the team at VEO"

**Base URLs**:
- Production: `https://api.veo.co.uk/api`
- Swagger docs: `https://api.veo.co.uk/swagger/ui/index`

**Capabilities**:
- Access users, groups, tagsets, comments, and videos
- Video tagging and AI analysis integration
- Create new video resources
- Manage video states and metadata

**API Endpoints (from developer docs)**:
- Videos: Create, read, update video resources with states
- Tags: Video tagging and clip management
- Comments: Add/read comments on videos
- Users/Groups: User and organization management

### Camera Service API (api.prod.camera.veo.co)

A separate API for camera management:
- Swagger UI: `https://api.prod.camera.veo.co/docs/explorer`
- ReDoc: `https://api.prod.camera.veo.co/docs/reference`
- Also `https://next.api.prod.camera.veo.co/docs/reference` (next version)

### API Access Reality

From Reddit (r/VeoCamera):
> "Veo support wasn't helpful with me when I asked. They just said they aren't onboarding new customers."
> "Feels like they used to have one - but have removed it."

The API exists but VEO is **not actively onboarding third-party developers**. They appear to have closed or severely restricted API access, likely to maintain control of the ecosystem.

### Custom Streaming (RTMP)
- VEO Live supports RTMP streaming to custom destinations
- Users can stream to their own website or third-party platforms
- This is the most "open" integration point

---

## 10. Patents

### Patent Portfolio

VEO Technologies APS has filed at least **2 patents** (CBInsights data):

#### Patent 1: US11310418B2 / WO2019141813A1
**Title**: "A computer-implemented method for automated detection of a moving area of interest in a video stream of field sports with a common object of interest"

**Filed**: January 2018 (WO filing)
**Granted**: US11310418 (April 2022)

**Key Claims** (reconstructed from abstracts):
1. Capturing images of a sports ground via a video camera system producing a video stream
2. Digitally processing the stream to continuously identify a "detected concentration of action" within field boundaries
3. The concentration of action is based on:
   - Detecting the presence of a sports game ball
   - Detecting the concentration/clustering of players
   - Combining both signals
4. The method produces a virtual camera view that follows the area of interest
5. Applicable to: football, handball, floorball, rugby, hockey, ice hockey, tennis, badminton, equestrian, water polo

**Patent Classification**:
- H04N23/698: Control of cameras for enlarged field of view (panoramic)
- H04N: Pictorial communication (television)

**Jurisdictions**: International (WO), US, European (ES3021734T3 - Spanish grant)

#### Patent 2: (Referenced but Details Less Clear)
CBInsights indicates 2 patents total. The second patent is likely a continuation, divisional, or related filing covering specific aspects of the same technology.

### Patent Analysis - What's Protected

VEO's patent specifically covers:
1. The method of detecting "concentration of action" (player clustering) to find where the game is happening
2. Using this to drive a virtual camera system
3. The combination of ball detection + player clustering for action area identification

### What's NOT Protected (Exploitable Gaps)
1. The specific ML models used (YOLO, etc.) are not patentable and are open-source
2. Team classification methods are not covered
3. Individual event detection (goals, shots, etc.) is not covered in the patent
4. Player tracking / identity maintenance is not covered
5. Analytics generation (heatmaps, etc.) is not covered
6. Jersey number detection is not covered
7. Software-only approaches (no camera hardware) are not covered by their camera-centric patent

### Risk Assessment for Manager Mentor
**LOW RISK**: Manager Mentor is a software-only platform that processes existing video. VEO's patent covers a complete camera-to-virtual-camera system. Our approach of:
- Processing user-uploaded video (not capturing it)
- Using standard open-source detection models
- Generating analytics and coaching insights
...does not appear to infringe on VEO's patent claims, which are focused on the camera capture + virtual camera generation method.

However, **consult a patent attorney** before launch, especially regarding Claim 1's broader language about "digitally processing a video stream to identify concentration of action."

---

## 11. Known Limitations & User Complaints

### Hardware Issues

| Issue | Source | Severity |
|-------|--------|----------|
| **Upload times** | Multiple Reddit threads, Reeplayer review | CRITICAL - 1-12+ hours upload, 24-48hr total |
| **WiFi dependency** | Cannot handle hotel/captive portal WiFi | HIGH - Camera has no browser |
| **Restart loops** | "Stuck in a restart loop - apparently common" | HIGH |
| **Camera mount falls** | "Tripod was kicked and it fell off" | MEDIUM |
| **Battery insufficient** | "Battery died before match ends" | MEDIUM |
| **Firmware issues** | "Firmware update caused problems from day 1" | HIGH |
| **Cam 1 aging** | "Less and less of us can get the app to work" | MEDIUM |

### Software/AI Issues

| Issue | Source | Severity |
|-------|--------|----------|
| **No real-time processing** | AI runs post-upload, not live | HIGH - users want instant feedback |
| **Manual sides/periods** | Must manually confirm which team is which side | MEDIUM |
| **Missing events** | Tackles, passes, dribbles not auto-detected | HIGH |
| **Player assignment manual** | Must manually link jersey numbers to player names | MEDIUM |
| **Video quality concerns** | "Quality of Veo recording is subpar IMHO" (Veo Go) | MEDIUM |
| **Far-side detection** | Players on far side of pitch less reliably detected | HIGH |
| **Follow-cam errors** | Sometimes loses the action during fast transitions | MEDIUM |

### Service/Business Issues

| Issue | Source | Severity |
|-------|--------|----------|
| **Price** | "Overpriced, unreliable, pay for more than you get" | HIGH |
| **Customer service** | "Customer service left a bad taste" - slow, text-only | HIGH |
| **Subscription model** | Camera requires ongoing subscription to function | HIGH - lock-in frustration |
| **Email/registration** | "Emails not going through, links not working" | MEDIUM |
| **API access closed** | "Not onboarding new customers" for API | MEDIUM |
| **No audio control** | Audio on/off appears unreliable | LOW |

### Pricing Pain Points

| Product | Cost |
|---------|------|
| Veo Cam 3 (hardware) | ~$1,000-1,500 |
| Monthly subscription | Required (various tiers) |
| Veo Live (livestreaming) | ~$360/year |
| Veo Analytics (team) | ~$400/year (~$26/month) |
| Team 12-month plan | ~$1,300/year |
| Total annual cost | **$2,000-3,000+** per team |

### Most Damning Quote
> "Since we got the Veo camera, there have been nothing but problems. The expense was ridiculous for a small club and then when our school got HUDL, we just saw how vast the difference is." - Trustpilot review

---

## 12. Tech Stack (from Job Postings)

### Confirmed/Strongly Inferred

| Layer | Technology |
|-------|-----------|
| **ML/AI** | Python, PyTorch (or TensorFlow), Computer Vision libraries |
| **Backend** | Python microservices |
| **Frontend** | React |
| **Billing** | Zuora (subscription management) |
| **CRM** | Salesforce |
| **Payments** | Adyen |
| **Video Processing** | Custom pipeline (likely FFmpeg-based for encoding) |
| **Camera Firmware** | Custom embedded software |
| **Mobile Apps** | iOS (Veo Camera app), Android |
| **API** | REST (Swagger/OpenAPI documented) |
| **Auth** | OAuth2 |

### From Job Postings (Current Hiring)

- **Machine Learning Engineer** (Copenhagen, Remote): "Proven experience from real-world ML projects, computer vision advantageous"
- **Student Worker - ML Engineer**: "Good theoretical foundations in ML and preferably computer vision, comprehensive skills in Python and ML libraries"
- **Senior Software Engineer - Monetization**: "Integrating platform with Zuora, Salesforce, and Adyen"
- **Full Stack/Platform Engineers**: "React frontends to Python microservices and cloud infrastructure"

### What This Tells Us
1. **They're still hiring ML engineers** = their CV pipeline is still being improved, not "done"
2. **Student workers in ML** = they use cheaper labor for model training/data labeling
3. **Monetization focus** = subscription/billing is complex and actively evolving
4. **React + Python microservices** = modern but standard web architecture
5. **No mention of edge AI chips** = confirms cloud-first processing strategy

---

## 13. Competitive Landscape

### VEO vs Key Competitors

| Feature | VEO | Trace | Pixellot | HUDL |
|---------|-----|-------|----------|------|
| **Camera** | Own hardware | Own hardware (wearable) | Own hardware (installed) | Own hardware (Focus) |
| **Price** | $2-3K+/year | ~$200/year per player | Varies (installation) | $$$ (enterprise) |
| **Target** | Teams/clubs | Players/parents | Venues/leagues | Schools/pro |
| **Follow-cam** | AI cloud | Per-player tracking | AI cloud | AI cloud |
| **Analytics** | Add-on ($) | Per-player stats | League analytics | Full platform |
| **Upload Time** | Hours | Hours | Instant (installed) | Hours |
| **API** | Restricted | Limited | Enterprise | Enterprise |
| **Highlight Delivery** | 24-48 hours | Within hours | Varies | Varies |

### VEO's Unique Advantages
1. **D2C model**: Sells directly to amateur teams (no venue installation needed)
2. **Portable**: One person sets it up in minutes
3. **Brand recognition**: Market leader in amateur sports AI camera
4. **180-degree coverage**: Captures entire pitch, no blind spots
5. **Price point**: Cheaper than Pixellot/HUDL for individual teams

### VEO's Key Weaknesses (Our Attack Vectors)
1. **Requires proprietary hardware**: $1000+ camera purchase
2. **24-48 hour delay**: No real-time analysis
3. **Manual team setup**: Sides, periods, lineup assignment required
4. **Limited event detection**: Only shots/goals/set pieces, no passes/tackles
5. **No coaching intelligence**: Shows WHAT happened, not WHAT TO DO
6. **Closed ecosystem**: No API access for third parties
7. **Upload dependency**: Needs stable internet for hours
8. **Subscription lock-in**: Camera is useless without subscription

---

## 14. Attack Surface for Manager Mentor

### Where We Can Win

| VEO Weakness | Manager Mentor Advantage |
|--------------|--------------------------|
| Requires $1000+ camera | Software-only, works with ANY video source (phone, existing cameras, VEO exports) |
| 24-48 hour processing delay | Could offer real-time or near-real-time processing with edge inference |
| Manual team setup required | Automated team classification (HSV Hue + SigLIP already working) |
| Limited event detection | Full event detection: passes, tackles, interceptions, dribbles, through-balls |
| No coaching intelligence | **AI tells coaches WHAT TO DO** - formation suggestions, tactical alerts, training drills |
| Closed API | Open platform, integrations welcome |
| $2-3K+/year | $9.99-79.99/month (fraction of VEO cost) |
| No referee tracking | Referee detection and offside analysis |
| Basic analytics only | Advanced xG, progressive passes, pressing triggers, defensive shape |
| Cloud-only processing | Hybrid: edge inference possible for real-time, cloud for deep analysis |

### Technical Approach Differences

| Aspect | VEO (Hardware Company) | Manager Mentor (AI Company) |
|--------|----------------------|---------------------------|
| **Input** | Proprietary dual 4K fisheye | Any video (phone, GoPro, broadcast, VEO export) |
| **Detection** | Cloud-only, unknown models | YOLO11 fallback chain, local or cloud GPU |
| **Tracking** | Proprietary | supervision ByteTrack (open-source, battle-tested) |
| **Team ID** | Color + manual sides input | SigLIP + HSV Hue clustering (automated) |
| **Ball** | Proprietary ball tracker | SAHI sliced inference + parabolic interpolation |
| **Events** | Goals, shots, set pieces only | Full: passes, tackles, interceptions, xG, formations |
| **Output** | Video + basic stats | Video + stats + AI coaching recommendations + drill plans |
| **Speed** | 24-48 hours | Target: <1 hour for full analysis |

### The "VEO Killer" Feature Set

1. **Software-Only Entry**: No hardware purchase needed. Upload from anything.
2. **AI Coach**: Not just "here's what happened" but "here's what to change and how to train it"
3. **Instant(ish) Results**: Target under 1 hour for full analysis (vs VEO's 24-48 hours)
4. **Full Event Detection**: Every pass, tackle, interception - not just goals and shots
5. **Automated Everything**: No manual sides/periods/lineup - AI figures it all out
6. **Training Plan Generator**: Converts match weaknesses into actual FA-style training sessions
7. **Player Development**: Track improvement over time with individual metrics
8. **Open Platform**: API-first, integrates with existing tools
9. **10x Cheaper**: $9.99-29.99/month vs $2,000+/year

---

## Appendix: Key URLs

| Resource | URL |
|----------|-----|
| VEO Product | https://www.veo.com/product/veo-cam-3 |
| VEO Analytics | https://www.veo.com/product/veo-analytics-2025 |
| VEO Player Spotlight | https://www.veo.com/product/veo-player-spotlight |
| VEO Developer Docs | https://developer.veo.co.uk/ |
| VEO API Swagger | https://api.veo.co.uk/swagger/ui/index |
| Camera Service API | https://api.prod.camera.veo.co/docs/explorer |
| VEO Patent (WO) | https://patents.google.com/patent/WO2019141813A1/en |
| VEO Patent (US) | US11310418B2 |
| VEO Careers | https://www.veo.com/careers |
| VEO Help Center | https://support.veo.co/hc/en-us |
| Processing Times | https://support.veo.co/hc/en-us/articles/37891575336849 |
| VEO Ideas Board | https://veo.nolt.io/ |
| SteadyView | https://www.veo.com/en-us/feature/steadyview |
| Veo Go (Smartphone) | https://www.veo.com/product/veo-go |
| r/VeoCamera | https://www.reddit.com/r/VeoCamera/ |

---

*This analysis was compiled from public sources including VEO's own documentation, patents, job postings, user forums, competitor reviews, and engineering blog posts. No proprietary information was accessed or used.*
