# Player Detection & Tracking: State-of-the-Art Deep Dive

**Date:** 8 March 2026
**Purpose:** Practical, implementable knowledge for detecting and tracking 22 players from VEO-style wide-angle football footage
**Context:** Manager Mentor / Football Analyzer project - current pipeline gets 15.8 players/frame avg, target is 20+

---

## TABLE OF CONTENTS

1. [The VEO Camera Problem](#1-the-veo-camera-problem)
2. [Detection Models: YOLO and Beyond](#2-detection-models-yolo-and-beyond)
3. [SoccerNet Ecosystem](#3-soccernet-ecosystem)
4. [Small Object Detection & SAHI](#4-small-object-detection--sahi)
5. [Multi-Object Tracking](#5-multi-object-tracking)
6. [Team Classification](#6-team-classification)
7. [Pitch/Field Detection](#7-pitchfield-detection)
8. [Goalkeeper Detection](#8-goalkeeper-detection)
9. [Camera Calibration](#9-camera-calibration)
10. [CPU vs GPU: What's Achievable](#10-cpu-vs-gpu-whats-achievable)
11. [Key Papers 2023-2025](#11-key-papers-2023-2025)
12. [Open Source Tools](#12-open-source-tools)
13. [Practical Implementation Plan](#13-practical-implementation-plan)

---

## 1. THE VEO CAMERA PROBLEM

### What We're Dealing With

VEO cameras (Cam 3 and Go) use **two 4K lenses** that capture a **180-degree panoramic view** of the entire pitch. The output is stitched into a single wide image. This creates a specific set of computer vision challenges:

**Resolution Math:**
- Combined output: ~7680x2160 pixels (dual 4K stitched)
- A football pitch is ~105m x 68m
- At the far side, a player ~1.8m tall occupies roughly **20-40 pixels** in height
- At the near side, players can be 100-200px tall
- This is a **5-10x scale variation** within a single frame

**Key Challenges:**
1. **Extreme small objects**: Far-side players are 20-40px — below YOLO's reliable detection threshold
2. **Barrel distortion**: 180-degree lenses produce significant lens distortion at edges
3. **Variable scale**: Near-side vs far-side players differ by 5-10x in pixel size
4. **Digital zoom artifacts**: VEO's "zoom" is crop-and-upscale, reducing effective resolution
5. **Stitching seam**: The boundary between the two 4K sensors can introduce artifacts
6. **Crowd/sideline confusion**: 180-degree view captures spectators, coaches, substitutes

### What "Good" Looks Like

| Metric | Current (Our Pipeline) | Target | World-Class (SoccerNet Winners) |
|--------|----------------------|--------|--------------------------------|
| Players detected/frame | 15.8 | 20+ | 22 (broadcast, not wide-angle) |
| Team split accuracy | 7.5 H / 8.2 A | 10 H / 10 A + refs | 95%+ team accuracy |
| GK detection | ~0/frame | 2/frame | Separate class in detector |
| Referee detection | Mixed with players | Separate class | Separate class |
| Tracking consistency | ByteTrack basic | <5% ID switches | HOTA 60%+ (GTATrack) |

---

## 2. DETECTION MODELS: YOLO AND BEYOND

### 2.1 YOLO Model Landscape (2024-2026)

| Model | Release | Key Innovation | Best For | License |
|-------|---------|---------------|----------|---------|
| YOLOv5 | 2020 | Mature, stable, huge community | Production baseline | AGPL-3.0 |
| YOLOv8 | Jan 2023 | Anchor-free, decoupled head | General purpose | AGPL-3.0 |
| YOLO11 | Oct 2024 | C3k2 blocks, improved small obj | Balanced speed/accuracy | AGPL-3.0 |
| YOLOv12 | Feb 2025 | Attention-centric architecture | Complex scenes | AGPL-3.0 |
| YOLO26 | Jan 2026 | NMS-free, STAL for small objects, 43% faster CPU | Latest & greatest | AGPL-3.0 |
| **RF-DETR** | Mar 2025 | DINOv2 backbone, no NMS needed | **Small objects, crowded scenes** | **Apache 2.0** |

### 2.2 Football-Specific Fine-Tuned Models

**Available Pre-Trained Weights:**

1. **Roboflow football-players-detection** (universe.roboflow.com)
   - 9,068 images, 4 classes: player, goalkeeper, referee, ball
   - Pre-trained YOLOv8/v9/v11 and RF-DETR weights available
   - mAP50 ~85-92% depending on model variant
   - Source: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

2. **HuggingFace keremberke/football-object-detection**
   - 1,232 images in COCO format
   - Classes: player, ball, referee, goalkeeper
   - Source: https://huggingface.co/datasets/keremberke/football-object-detection

3. **SportsVision-YOLO** (github.com/forzasys-students/SportsVision-YOLO)
   - Fine-tuned YOLOv8 for soccer and ice hockey
   - Detects balls, players, and logos across camera angles

4. **Our current model: football_best.pt**
   - From HuggingFace, 4 classes: ball, goalkeeper, player, referee
   - mAP 0.785
   - **Known issue**: Rarely detects GKs at VEO distance (2/53 frames)

5. **SoccerSynth-Detection** (arxiv.org/html/2501.09281)
   - **Synthetic dataset** for soccer player detection
   - Random lighting, textures, simulated camera motion blur
   - Useful for **pretraining before fine-tuning** on real data

### 2.3 RF-DETR: The New Contender

RF-DETR deserves special attention for our use case:

- **Architecture**: DINOv2 vision transformer backbone (not CNN-based like YOLO)
- **No NMS required**: Eliminates post-processing overhead
- **Small object strength**: RF-DETR-Small beats YOLO11-x on COCO by 1.8 mAP50:95 points
- **Better with occlusion**: Transformer attention handles overlapping players better
- **Fine-tuning friendly**: Designed specifically for domain adaptation with limited data
- **Apache 2.0 license**: Unlike YOLO's AGPL-3.0, this is commercially permissive
- **Latency**: 7.77ms faster than YOLO11-x at higher accuracy

**Recommendation**: Evaluate RF-DETR alongside YOLO11 for our football detection. The transformer architecture's ability to handle small objects and crowded scenes directly addresses our VEO camera challenges.

### 2.4 What SoccerNet Competition Winners Actually Use

Based on the 2024 and 2025 SoccerNet Challenge results:

**Game State Reconstruction 2024 Winner (GS-HOTA 63.81):**
- Detection: **Fine-tuned YOLOv5m**
- Camera: **SegFormer-based** camera parameter estimator
- Tracking: **DeepSORT** enhanced with ReID, orientation prediction
- Jersey numbers: Custom recognition module
- Team affiliation: Jersey color clustering + positional inference

**SoccerTrack 2025 Winner (GTATrack, HOTA 0.60):**
- Detection: **YOLOX** with pseudo-labeling for small/distorted targets
- Tracking: **Deep-EIoU** (Deep Expansion IoU) + Global Tracklet Association
- ReID: **OSNet** embeddings (outperformed transformer-based ReID)
- Refinement: Split-and-merge strategy for identity consistency

**Key Takeaway**: Winners use **fine-tuned YOLO variants** (not the latest versions -- YOLOv5m and YOLOX are older but well-understood). The edge comes from the **tracking and post-processing pipeline**, not from chasing the newest detector.

---

## 3. SOCCERNET ECOSYSTEM

### 3.1 What Is SoccerNet?

SoccerNet is the largest open benchmark for soccer video understanding. It provides:

- **500+ complete broadcast matches** with annotations
- **Annual challenges** (5th edition in 2025) across multiple tasks
- **Standardized evaluation metrics** (HOTA, GS-HOTA)
- **Open-source devkits** for every challenge task

### 3.2 Available Datasets

| Dataset | Size | Task | Annotations |
|---------|------|------|-------------|
| SoccerNet-v2 | 500 games | Action spotting | 300K+ temporal annotations |
| SoccerNet-Tracking | 12 games, 106 clips | MOT | Player bounding boxes + IDs |
| SoccerNet-ReID | -- | Re-identification | Player crops with IDs |
| SoccerNet-Calibration | -- | Camera calibration | Pitch keypoints per frame |
| SoccerNet-GameState | -- | Full reconstruction | Player position, team, role, jersey# |
| SoccerSynth-Detection | Synthetic | Detection pretraining | Bounding boxes, varied conditions |

### 3.3 Key Repos

- **sn-tracking**: https://github.com/SoccerNet/sn-tracking
- **sn-gamestate**: https://github.com/SoccerNet/sn-gamestate (built on TrackLab)
- **sn-calibration**: https://github.com/SoccerNet/sn-calibration
- **sn-reid**: https://github.com/SoccerNet/sn-reid
- **sn-trackeval**: https://github.com/SoccerNet/sn-trackeval

### 3.4 2025 Challenge Tasks

1. **Team Ball Action Spotting**: Detect which team performs actions
2. **Monocular Depth Estimation**: Estimate depth from broadcast video
3. **Multi-View Foul Recognition**: Classify fouls from multiple angles
4. **Game State Reconstruction**: Full pipeline -- detect, track, identify all players, map to 2D minimap

**Major trend from 2025**: Pretrained large-scale models (vision transformers, vision-language models) were omnipresent, especially when combined with soccer-specific fine-tuning.

---

## 4. SMALL OBJECT DETECTION & SAHI

### 4.1 The Small Object Problem

This is our #1 technical challenge. From a VEO 180-degree camera:

- Far-side players: **20-40px tall** (below YOLO's comfortable detection zone)
- YOLO's feature pyramid typically loses fine detail at this scale
- Standard 640x640 inference resolution **downsamples** these players into ~5-10px blobs
- Even at 1280x1280 inference, far-side players may only be 10-20px in the resized image

**Why standard YOLO struggles:**
- Convolutional downsampling progressively loses spatial detail
- Small objects get heavily compressed through the backbone's stride-32 pathway
- The P3 feature map (stride 8) helps, but 20px objects at stride 8 are only 2.5 feature pixels

### 4.2 SAHI: Slicing Aided Hyper Inference

SAHI is the most practical solution for small object detection in high-resolution images.

**How It Works:**
1. **Slice**: Divide the high-resolution image (e.g., 7680x2160) into overlapping tiles
2. **Detect**: Run YOLO/RF-DETR on each tile independently at the model's native resolution
3. **Stitch**: Merge all detections back into the original coordinate space
4. **NMS**: Apply Non-Maximum Suppression to remove duplicate detections from overlapping regions

**Visual Example:**
```
Original 7680x2160 image
+------+------+------+------+------+------+------+------+
|      |      |      |      |      |      |      |      |
| Tile | Tile | Tile | Tile | Tile | Tile | Tile | Tile |
|  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |
|      |      |      |      |      |      |      |      |
+------+------+------+------+------+------+------+------+
|      |      |      |      |      |      |      |      |
| Tile | Tile | Tile | Tile | Tile | Tile | Tile | Tile |
|  9   |  10  |  11  |  12  |  13  |  14  |  15  |  16  |
|      |      |      |      |      |      |      |      |
+------+------+------+------+------+------+------+------+
Each tile is ~1280x1280 with 20% overlap
```

### 4.3 Optimal SAHI Settings for Football

**For VEO-style 7680x2160 footage:**

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="football_best.pt",
    confidence_threshold=0.25,
    device="cuda"
)

result = get_sliced_prediction(
    image=frame,
    detection_model=detection_model,
    slice_height=1280,        # Match YOLO training resolution
    slice_width=1280,         # Square tiles
    overlap_height_ratio=0.2, # 20% overlap prevents edge misses
    overlap_width_ratio=0.2,  # 20% overlap
    postprocess_type="NMS",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False
)
```

**Key Parameters:**

| Parameter | Recommended | Why |
|-----------|------------|-----|
| `slice_height` | 1280 | Match training resolution for best feature extraction |
| `slice_width` | 1280 | Square tiles work best with YOLO |
| `overlap_height_ratio` | 0.2 | 20% prevents players at tile boundaries being missed |
| `overlap_width_ratio` | 0.2 | Same horizontal |
| `confidence_threshold` | 0.25 | Lower than usual -- catch small distant players |
| `postprocess_match_threshold` | 0.5 | Merge threshold for overlapping detections |

**Alternative for maximum recall (slower):**
- `slice_height=640`, `slice_width=640`, `overlap=0.3` -- more tiles, each player appears larger within its tile
- Tradeoff: ~4x more inference calls, but far-side players become ~40-80px within each tile

### 4.4 SAHI + Full-Image Hybrid

Best practice is to run **both** SAHI sliced inference AND full-image inference, then merge:

```python
# Full-image inference catches large near-side players well
full_result = get_prediction(image, detection_model)

# SAHI sliced inference catches small far-side players
sliced_result = get_sliced_prediction(image, detection_model, ...)

# Merge and NMS
combined = merge_predictions(full_result, sliced_result)
```

This hybrid approach gives the best of both worlds: reliable large-object detection plus small-object recovery.

### 4.5 Performance Impact

| Approach | Tiles | Inference Calls | Time (RTX 3060) | Expected Players |
|----------|-------|----------------|-----------------|-----------------|
| Full image 640px | 1 | 1 | ~15ms | 10-14 (misses far side) |
| Full image 1280px | 1 | 1 | ~40ms | 12-16 |
| SAHI 1280 tiles | ~12 | 12 | ~180ms | 18-22 |
| SAHI 640 tiles | ~48 | 48 | ~720ms | 20-22 |
| Hybrid (full + SAHI 1280) | ~13 | 13 | ~200ms | 19-22 |

**Recommendation**: Use **Hybrid (full + SAHI 1280)** for offline processing. This is not real-time at 30fps but is practical for post-match analysis at 5fps.

---

## 5. MULTI-OBJECT TRACKING

### 5.1 Tracker Comparison for Football

| Tracker | HOTA (SoccerNet) | Speed | ReID | Camera Motion | Best For |
|---------|-----------------|-------|------|---------------|----------|
| **ByteTrack** | ~70% | Very Fast | No | No | Real-time, simple scenes |
| **BoT-SORT** | ~75% | Moderate | Yes | Yes (CMC) | General sports tracking |
| **Deep SORT** | ~68% | Moderate | Yes (ResNet) | No | Legacy, well-understood |
| **StrongSORT** | ~73% | Slow | Yes (BoT) | Yes | High accuracy, offline |
| **SportSORT** | **88.0%** | Moderate | Yes (domain) | Yes | **Sports-specific SOTA** |
| **GTATrack** | **60%** (fisheye) | Moderate | Yes (OSNet) | Yes | **Challenge winner 2025** |
| **Deep HM-SORT** | ~76% | Moderate | Yes (deep) | No | Sports with occlusion |

### 5.2 ByteTrack (Our Current Tracker)

**How it works:**
1. Split detections into high-confidence and low-confidence groups
2. First association: Match high-confidence detections to existing tracks using IoU
3. Second association: Match remaining tracks to low-confidence detections
4. This two-stage approach recovers partially occluded players

**Strengths for football:**
- Very fast (can run on CPU at reasonable FPS)
- No deep appearance features needed (simpler pipeline)
- Good at handling temporary occlusions via the low-confidence recovery

**Weaknesses:**
- No appearance features means players in similar positions get swapped
- No camera motion compensation (VEO footage has no camera movement, so less of an issue)
- No re-identification -- once a track is lost, identity is gone

### 5.3 BoT-SORT (Strong Alternative)

**How it works:**
- Extends ByteTrack with:
  1. **Camera Motion Compensation (CMC)**: Estimates global camera motion to adjust Kalman predictions
  2. **ReID features**: Extracts appearance embeddings to maintain identity through occlusion
  3. **Refined Kalman Filter**: Better state estimation for non-linear motion

**For VEO footage**: CMC is less critical (camera is static), but the ReID features are valuable for:
- Players crossing paths
- Maintaining identity after temporary occlusion by another player
- Recovering tracks after brief out-of-frame moments

### 5.4 SportSORT (Sports-Specific SOTA, 2025)

Published in Machine Vision and Applications, 2025. Three key innovations:

1. **Domain-Specific Feature Matching**: Leverages jersey colors and numbers to reduce IoU-related misidentification
2. **Corrective Matching Stage**: Resolves identity mismatches from long-term occlusions
3. **Out-of-View Re-Association**: Re-identifies players who temporarily exit and re-enter the frame

**Results**: HOTA 81.3% on SportsMOT, **88.0% on SoccerNet-Tracking** (SOTA)

**Why this matters for us**: VEO's 180-degree view means players near the sideline frequently enter/exit the visible area. SportSORT's OoV re-association directly addresses this.

### 5.5 GTATrack (SoccerTrack 2025 Winner)

**Architecture**: Two-stage hierarchical tracking:
1. **Online stage**: Deep-EIoU for frame-to-frame association (combines Expansion IoU with ReID appearance similarity)
2. **Offline stage**: Global Tracklet Association (GTA) for trajectory-level refinement

**Key insight**: Uses a **pseudo-labeling strategy** to boost detector recall on small and distorted targets -- directly relevant for our VEO fisheye-like distortion problem.

**ReID backbone**: **OSNet** consistently outperformed transformer-based ReID models. Its lightweight multi-scale feature extraction was more resilient to occlusions and fast motion.

### 5.6 Practical Recommendation

**For Manager Mentor:**

**Phase 1 (Now)**: Keep ByteTrack but add:
- Frame gap detection (already done)
- Track smoothing (Kalman filter on bounding box positions)
- Simple appearance matching using jersey color histograms

**Phase 2 (Next)**: Upgrade to BoT-SORT with:
- OSNet ReID embeddings (lightweight, proven best in SoccerNet 2025)
- `track_buffer=60` (2 seconds at 30fps) for temporary occlusion recovery
- Identity-prediction consistency check (split tracks that switch identity attributes)

**Phase 3 (Future)**: Implement SportSORT's Out-of-View Re-Association or adopt GTATrack's global tracklet association for long-form game analysis.

---

## 6. TEAM CLASSIFICATION

### 6.1 Approaches Overview

| Method | Accuracy | Speed | Robustness | Complexity |
|--------|----------|-------|------------|------------|
| HSV Hue clustering | 75-85% | Very Fast | Poor (lighting) | Low |
| LAB color clustering | 80-88% | Very Fast | Moderate | Low |
| **SigLIP + UMAP + KMeans** | **90-95%** | Moderate | **High** | Medium |
| CLIP embeddings + KMeans | 88-93% | Moderate | High | Medium |
| ResNet crop classification | 85-92% | Moderate | Moderate | Medium |
| GNN-based assignment | 92-96% | Slow | Very High | High |

### 6.2 HSV Hue Clustering (Our Current Method)

**How it works:**
1. Crop the upper body region of each detected player bounding box
2. Convert to HSV color space
3. Extract the Hue channel (ignores brightness/saturation variation)
4. Use KMeans (k=2) to cluster Hue values
5. Assign players to clusters

**Current implementation strengths:**
- Fixed the LAB clustering failure on dark jerseys (Hue correctly separates)
- Current ratio: 7.5 Home / 8.2 Away (good split)
- Very fast computation

**Current weaknesses:**
- Fails when both teams have similar hue (rare but possible)
- Sensitive to shadows and stadium lighting
- No semantic understanding -- treats all visible clothing equally
- Can't handle teams with multicolor/patterned jerseys well

### 6.3 SigLIP + UMAP + KMeans (Recommended Upgrade)

This is the approach used by Roboflow's sports analysis pipeline and is the current practical SOTA for team classification.

**Pipeline:**

```python
# Step 1: Detect all players
detections = model.predict(frame)

# Step 2: Crop the CENTRAL region of each bounding box
# (Avoids pitch/background at edges, focuses on jersey)
crops = []
for box in detections.xyxy:
    x1, y1, x2, y2 = box
    # Take central 50% to focus on torso/jersey
    cx = (x2 - x1) * 0.25
    cy = (y2 - y1) * 0.15  # Less top crop (head), more bottom crop (legs)
    crop = frame[int(y1+cy):int(y2-cy*2), int(x1+cx):int(x2-cx)]
    crops.append(crop)

# Step 3: Generate SigLIP embeddings
from transformers import AutoProcessor, AutoModel
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")

inputs = processor(images=crops, return_tensors="pt")
embeddings = model.get_image_features(**inputs)  # (N, 768)
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

# Step 4: Dimensionality reduction with UMAP
import umap
reducer = umap.UMAP(n_components=3, random_state=42)
reduced = reducer.fit_transform(embeddings.numpy())

# Step 5: Cluster into teams
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 = team1, team2, referee
labels = kmeans.fit_predict(reduced)
```

**Why SigLIP works better than color clustering:**
- Captures **semantic visual similarity** (jersey pattern, style, texture), not just color
- Robust to **lighting changes** across the pitch (shadows vs sunlight)
- Handles **multi-color jerseys** and patterned kits
- **Zero-shot**: Works across any game without manual color calibration
- The 768-dim embedding captures far more information than a 3-channel color histogram

**Practical tips:**
- Sample embeddings from **multiple frames** (e.g., every 30th frame) and cluster the aggregated set for more robust assignment
- Use `n_clusters=3` to separate team1, team2, and referees (referees typically cluster separately)
- For goalkeepers in different colored jerseys, may need `n_clusters=4` or post-hoc assignment

### 6.4 What Competition Winners Use

**SoccerNet GSR 2024 Winner:**
- Team affiliation inferred by **clustering jersey colors** and comparing **cluster-wise mean x-positions** (assuming teams tend to remain on their respective sides)
- Simple but effective for broadcast footage

**SoccerNet GSR 2025 (Constructor Tech):**
- **LLAMA-3.2-Vision** for open-set role/team/jersey recognition via instruction prompts
- Used as a verification layer on top of color clustering

**Key insight**: Even competition winners use relatively simple team classification. The hard part is **detection and tracking**, not team assignment.

### 6.5 GNN-Based Team Assignment (Advanced)

Graph Neural Networks can model player relationships:
- Nodes: Player detections
- Edges: Spatial proximity, appearance similarity
- The GNN learns to partition the graph into teams

This handles edge cases (substitutions, formation changes) but adds significant complexity. Not recommended until basic pipeline is solid.

---

## 7. PITCH/FIELD DETECTION

### 7.1 Why It Matters

Pitch detection serves two critical purposes:
1. **Filter out non-players**: Spectators, coaches, ball boys, substitutes on the sideline
2. **Enable homography**: Map pixel coordinates to real-world pitch positions

### 7.2 Approaches

**Method 1: Color-based segmentation (Simple)**
```python
import cv2
import numpy as np

# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Green grass mask
lower_green = np.array([30, 40, 40])
upper_green = np.array([80, 255, 255])
pitch_mask = cv2.inRange(hsv, lower_green, upper_green)

# Morphological cleanup
kernel = np.ones((15, 15), np.uint8)
pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, kernel)
pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_OPEN, kernel)

# Filter detections: only keep players whose foot position is on the pitch
for detection in detections:
    foot_x = int((detection.x1 + detection.x2) / 2)
    foot_y = int(detection.y2)  # Bottom of bbox
    if pitch_mask[foot_y, foot_x] > 0:
        keep_detection(detection)
```

**Pros**: Fast, no model needed, works well on natural grass
**Cons**: Fails on artificial surfaces, snowy pitches, or when sideline is also green

**Method 2: Semantic segmentation (Robust)**
- Train a lightweight segmentation model (e.g., SegFormer-B0) on pitch images
- Binary mask: pitch vs. non-pitch
- More robust than color thresholding

**Method 3: Keypoint-based field registration (Best)**
- Detect pitch keypoints (corners, intersections, penalty spots, center circle)
- Use homography to define the pitch boundary precisely
- This is the approach used by Roboflow (32 characteristic pitch points) and SoccerNet calibration challenge winners

### 7.3 Practical Recommendation

Use a **two-stage approach**:
1. **Fast green mask** for initial filtering (removes ~80% of non-pitch detections)
2. **Pitch keypoint model** for precise boundary when available
3. **Convex hull of detected players** as a sanity check -- if a "player" is far outside the convex hull of other players, it's likely a false positive

---

## 8. GOALKEEPER DETECTION

### 8.1 The GK Problem

Goalkeepers are challenging because:
- They wear **different colored jerseys** from their teammates (and from each other)
- They are often **far from the camera** (at the far goal)
- They **stand relatively still** compared to outfield players
- There are only **2 per match** vs. 20 outfield players (severe class imbalance in training data)
- They occupy the **goal area** which has nets, posts, and advertising boards (cluttered background)

### 8.2 Current State

From research and our experience:
- Most YOLO models trained with a "goalkeeper" class have **poor GK recall** (class imbalance)
- Our current model: 2 GK detections in 53 frames (3.8% recall)
- The Roboflow football dataset includes GK as a class but the training data is insufficient for wide-angle footage

### 8.3 Best Approaches

**Approach 1: Detection model with GK class (current, insufficient)**
- Fine-tune YOLO with more GK-annotated data
- Use focal loss or class-weighted loss to handle imbalance
- Augment GK training data with crops at various scales

**Approach 2: Position-based inference (practical)**
```python
def classify_goalkeeper(player_detections, pitch_homography):
    """Classify GKs based on their position on the pitch."""
    for detection in player_detections:
        pitch_pos = homography_transform(detection.foot_position)
        # GKs are within ~16m of the goal line (penalty area depth)
        if pitch_pos.x < 16.5 or pitch_pos.x > (105 - 16.5):
            # Additional check: GKs are usually the most isolated player
            # in their team's defensive third
            if is_most_isolated_in_area(detection, player_detections):
                detection.role = "goalkeeper"
```

**Approach 3: Jersey color anomaly (practical)**
- After team classification, the GK is the **outlier** in each team's color cluster
- The player whose jersey color is most different from their team's mean is likely the GK

**Approach 4: Combined heuristic (recommended)**
1. Check if the detection model identifies GK class with confidence > 0.5 (rare but authoritative)
2. If not, use position-based inference (player in penalty area, isolated from teammates)
3. Validate with jersey color anomaly detection
4. Persist GK identity through tracking (once identified, maintain through the match)

---

## 9. CAMERA CALIBRATION

### 9.1 The Wide-Angle Challenge

VEO's 180-degree view introduces **barrel distortion** where:
- Straight lines appear curved, especially at the edges
- Objects at the periphery are stretched/distorted
- Scale varies non-linearly across the image

### 9.2 Standard Approach: Pitch Keypoint Homography

**Step 1: Define pitch keypoints**
The standard set (from Roboflow) uses **32 characteristic points**:
- 4 corner flags
- 4 penalty area corners
- 2 penalty spots
- 4 goal area corners
- 2 goal post pairs (4 points)
- Center circle (8 points around circumference)
- Halfway line intersections (2 points)
- Arc intersections (2 points)

**Step 2: Detect keypoints in the image**
- Use YOLOv8-pose fine-tuned for pitch keypoints
- Or use a custom keypoint detection model (HRNet for 57 keypoints was 1st place SoccerNet 2023)

**Step 3: Compute homography**
```python
import cv2
import numpy as np

# source_points: detected keypoint pixel coordinates
# target_points: known real-world pitch coordinates (in meters)
H, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

# Transform player foot positions to pitch coordinates
foot_pixels = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
pitch_coords = cv2.perspectiveTransform(foot_pixels, H)
```

### 9.3 Wide-Angle Lens Distortion Correction

For VEO's 180-degree cameras, standard homography is insufficient. You need **lens distortion correction first**:

```python
# Camera calibration using checkerboard or known field dimensions
# Distortion coefficients: k1, k2, p1, p2, k3
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Undistort the image first
undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

# THEN apply homography to the undistorted image
H, mask = cv2.findHomography(source_points_undistorted, target_points, cv2.RANSAC)
```

### 9.4 TVCalib: Purpose-Built for Sports

**TVCalib** (arxiv.org/abs/2207.11709) is specifically designed for camera calibration in soccer:
- Handles single and multi-camera setups
- Robust to partial field visibility
- Works with both broadcast and static cameras
- Open source: https://mm4spa.github.io/tvcalib/

### 9.5 SoccerNet Calibration Challenge Insights

The 2023 winner used:
- Deep learning for keypoint AND line detection simultaneously
- Geometric constraints from known real-world pitch dimensions
- A **voter algorithm** that iteratively selects the most reliable keypoints
- Significantly increased usable calibration points by exploiting line-line and line-conic intersections

**Practical recommendation**: Use Roboflow's 32-point keypoint model as a baseline. If accuracy is insufficient for VEO footage, consider TVCalib or training a custom model with VEO-specific distortion patterns.

---

## 10. CPU VS GPU: WHAT'S ACHIEVABLE

### 10.1 Inference Speed Benchmarks

| Model | GPU (RTX 3060) | GPU (RTX 4090) | CPU (i7-12th) | CPU (M2 Pro) |
|-------|---------------|----------------|---------------|--------------|
| YOLOv8-N (Nano) | ~2ms | ~1ms | ~50ms (20fps) | ~30ms (33fps) |
| YOLOv8-S (Small) | ~5ms | ~2ms | ~100ms (10fps) | ~60ms (17fps) |
| YOLOv8-M (Medium) | ~10ms | ~4ms | ~200ms (5fps) | ~120ms (8fps) |
| YOLOv8-L (Large) | ~15ms | ~6ms | ~400ms (2.5fps) | ~250ms (4fps) |
| YOLO11-M | ~8ms | ~3ms | ~180ms (5.5fps) | ~100ms (10fps) |
| RF-DETR-S | ~12ms | ~5ms | ~300ms (3fps) | ~180ms (5.5fps) |
| YOLO26-N | -- | ~1ms | ~29ms (34fps)* | -- |

*YOLO26-N claims 43% faster CPU inference than YOLO11-N

### 10.2 Full Pipeline Speed (Detection + Tracking + Classification)

| Pipeline | GPU (RTX 3060) | CPU (i7-12th) |
|----------|---------------|---------------|
| YOLO-S + ByteTrack + HSV color | ~10ms (100fps) | ~150ms (7fps) |
| YOLO-M + ByteTrack + SigLIP | ~30ms (33fps) | ~400ms (2.5fps) |
| YOLO-M + SAHI(1280) + ByteTrack | ~200ms (5fps) | ~2.5s (0.4fps) |
| YOLO-M + SAHI(1280) + BoT-SORT + SigLIP | ~250ms (4fps) | ~3s (0.3fps) |
| RF-DETR + SAHI + SportSORT + SigLIP | ~300ms (3fps) | ~4s (0.25fps) |

### 10.3 What This Means for Manager Mentor

**Offline post-match analysis** (our primary use case):
- A 90-minute match at 30fps = 162,000 frames
- Processing every frame at 5fps pipeline speed = **9 hours on GPU**
- Processing every 3rd frame (10fps effective) at 5fps = **3 hours on GPU**
- Processing every 6th frame (5fps effective) at 5fps = **1.5 hours on GPU**

**Practical strategy:**
1. **Quick preview** (current): Process every 10th frame, simple pipeline = ~30 min on GPU
2. **Standard analysis**: Process every 3rd frame, full pipeline = ~3 hours on GPU
3. **Full analysis**: Every frame, full pipeline = overnight batch job

**CPU-only is viable** for quick preview mode (process every 30th frame with YOLO-N + ByteTrack = ~15 minutes for a full match), but GPU is required for production-quality analysis.

### 10.4 Cloud GPU Options

| Service | GPU | Cost/hr | 90-min Match (full) | 90-min Match (3rd frame) |
|---------|-----|---------|---------------------|--------------------------|
| RunPod | RTX 3090 | $0.44 | ~$4 (9hr) | ~$1.30 (3hr) |
| Lambda Labs | A10G | $0.60 | ~$5.40 | ~$1.80 |
| AWS (g4dn.xlarge) | T4 | $0.53 | ~$6 (slower GPU) | ~$2 |
| Modal | A100 | ~$1.10 | ~$3 (faster) | ~$1 |

**Unit economics**: At the Club tier ($29.99/mo) with ~8 matches/month, processing cost is $8-15/month on cloud GPU = healthy margins.

---

## 11. KEY PAPERS 2023-2025

### Detection & Tracking

1. **SoccerNet 2025 Challenges Results** (Aug 2025)
   - Comprehensive overview of SOTA across all soccer CV tasks
   - https://arxiv.org/abs/2508.19182

2. **GTATrack: Winner Solution to SoccerTrack 2025** (Feb 2026)
   - Deep-EIoU + Global Tracklet Association
   - HOTA 0.60 on fisheye soccer tracking
   - https://arxiv.org/abs/2602.00484

3. **From Broadcast to Minimap: SOTA SoccerNet Game State Reconstruction** (Apr 2025)
   - YOLOv5m + SegFormer + DeepSORT pipeline, GS-HOTA 63.81
   - https://arxiv.org/abs/2504.06357

4. **SportSORT: Overcoming MOT Challenges in Sports** (2025)
   - Domain-specific features + OoV re-association, HOTA 88% on SoccerNet
   - https://link.springer.com/article/10.1007/s00138-025-01756-y

5. **SportsMOT: Large Multi-Object Tracking Dataset** (ICCV 2023)
   - Benchmark dataset for sports tracking across multiple sports
   - https://arxiv.org/abs/2304.05170

### Small Object Detection

6. **SAHI: Slicing Aided Hyper Inference** (2022, cited 500+)
   - Foundation paper for tiled inference approach
   - https://arxiv.org/abs/2202.06934

7. **Exploring Small Object Detection with YOLO11** (2024)
   - Ultralytics guidance on YOLO11 for small objects
   - https://www.ultralytics.com/blog/exploring-small-object-detection-with-ultralytics-yolo11

### Camera Calibration

8. **TVCalib: Camera Calibration for Sports Field Registration** (2022)
   - Purpose-built for soccer camera calibration
   - https://arxiv.org/abs/2207.11709

9. **Enhancing Soccer Camera Calibration Through Keypoint Exploitation** (Oct 2024)
   - 1st place SoccerNet Calibration 2023
   - https://arxiv.org/abs/2410.07401

### Team Classification & ReID

10. **Multi-task Learning for Joint ReID, Team Affiliation, and Role Classification** (Jan 2024)
    - Joint learning for tracking identity attributes
    - https://arxiv.org/abs/2401.09942

11. **Towards Long-term Player Tracking with Graph Hierarchies** (WACV 2025)
    - Graph-based approach to long-term identity maintenance
    - https://openaccess.thecvf.com/content/WACV2025W/CV4WS/papers/Koshkina_Towards_long-term_player_tracking_with_graph_hierarchies_and_domain-specific_features_WACVW_2025_paper.pdf

### Architecture

12. **RF-DETR: Real-Time Object Detection Model** (ICLR 2026)
    - DINOv2 backbone, SOTA on COCO, designed for fine-tuning
    - https://github.com/roboflow/rf-detr

13. **Deep HM-SORT: Enhancing MOT in Sports** (Jun 2024)
    - Harmonic mean association + Expansion IoU
    - https://arxiv.org/abs/2406.12081

---

## 12. OPEN SOURCE TOOLS

### 12.1 Supervision (Roboflow)

**What**: Comprehensive computer vision toolkit for detection, tracking, and annotation.
**GitHub**: https://github.com/roboflow/supervision
**Stars**: 25K+
**Key features**:
- Annotators: bounding boxes, masks, labels, traces, heatmaps
- Trackers: ByteTrack built-in
- Utilities: NMS, polygon zones, line counters
- Dataset management: convert between formats

**We already use this** -- it's the backbone of our tracking pipeline.

### 12.2 Roboflow Sports

**What**: Sports-specific CV tools and examples built on supervision.
**GitHub**: https://github.com/roboflow/sports
**Key features**:
- Soccer: player detection, radar visualization, pitch keypoint detection, homography
- Basketball: player identification pipeline (RF-DETR + SAM2 + SigLIP)
- Pre-trained models for pitch keypoint detection (32 points)
- Example notebooks for complete pipelines

**Highly relevant** -- their soccer example is essentially what we're building.

### 12.3 SportsLabKit

**What**: Toolkit for converting sports video into structured data.
**GitHub**: https://github.com/AtomScott/SportsLabKit
**Key features**:
- Built-in SORT, DeepSORT, ByteTrack, and TeamTrack
- 2D pitch calibration (bbox to pitch coordinates)
- BoundingBoxDataFrame and CoordinatesDataFrame wrappers
- Designed for academic benchmarking

**Useful for**: Pitch coordinate transformation and data format standardization.

### 12.4 TrackLab

**What**: Modular end-to-end tracking framework.
**GitHub**: https://github.com/TrackingLaboratory/tracklab
**Key features**:
- Hydra-based YAML configuration for every component
- Pluggable detectors, trackers, re-identifiers
- SoccerNet Game State devkit is built on top of TrackLab
- Research-focused but well-structured

**Useful for**: If we want to adopt the exact SoccerNet winner pipeline structure.

### 12.5 TrackEval

**What**: Multi-object tracking evaluation metrics.
**GitHub**: https://github.com/JonathonLuiten/TrackEval
**Key features**:
- HOTA metric (standard for modern MOT evaluation)
- MOTA, MOTP, IDF1, and other metrics
- Pure Python (NumPy + SciPy only)
- SoccerNet has their own fork: https://github.com/SoccerNet/sn-trackeval

**Useful for**: Quantitatively evaluating our tracking quality if we create ground truth annotations.

### 12.6 Tool Summary

| Tool | Use In Our Pipeline | Priority |
|------|-------------------|----------|
| **supervision** | Already using -- detection, tracking, annotation | ACTIVE |
| **roboflow/sports** | Pitch keypoints, homography, radar viz examples | HIGH |
| **SportsLabKit** | Pitch coordinate transforms, data formats | MEDIUM |
| **TrackLab** | Reference architecture for full pipeline | LOW (reference) |
| **TrackEval** | Evaluation when we have ground truth | LOW (future) |
| **SAHI** | Small object detection for VEO footage | **HIGH - IMPLEMENT NEXT** |

---

## 13. PRACTICAL IMPLEMENTATION PLAN

### What We Should Actually Build

Based on all the research above, here is the prioritized implementation plan to go from 15.8 to 22 detected players per frame.

### Phase A: SAHI Integration (Biggest Impact)

**Why first**: This is the single highest-impact change. Our current pipeline misses far-side players because they're too small. SAHI directly solves this.

**Implementation**:
1. Install SAHI: `pip install sahi`
2. Replace single-inference detection with SAHI hybrid (full + sliced)
3. Use `slice_height=1280, slice_width=1280, overlap=0.2`
4. Expected improvement: 15.8 -> 19-21 players/frame

**Estimated effort**: 1-2 sessions
**Expected impact**: +3-5 players/frame

### Phase B: Pitch Mask Filtering

**Why next**: Once SAHI finds more objects, we need to filter false positives from sideline/stands.

**Implementation**:
1. Add HSV green grass segmentation
2. Create pitch mask with morphological operations
3. Filter detections: only keep players with foot position on pitch mask
4. Optional: Use Roboflow's pitch keypoint model for precise boundary

**Estimated effort**: 1 session
**Expected impact**: Removes 2-5 false positives/frame, improves precision

### Phase C: SigLIP Team Classification Upgrade

**Why**: Current HSV Hue clustering works (7.5/8.2 split) but fragile. SigLIP is dramatically more robust.

**Implementation**:
1. Install: `pip install transformers`
2. Sample player crops from every 30th frame
3. Generate SigLIP embeddings, reduce with UMAP, cluster with KMeans(k=3)
4. Assign team labels based on cluster membership
5. Use temporal smoothing (majority vote over 30-frame window per track ID)

**Estimated effort**: 1-2 sessions
**Expected impact**: Team classification accuracy 85% -> 92%+

### Phase D: GK Detection Heuristic

**Implementation**:
1. After team classification, find the color outlier in each team
2. Cross-reference with position (in penalty area or behind team's defensive line)
3. Persist GK identity through tracking
4. Override detection class with "goalkeeper" for identified players

**Estimated effort**: 0.5 sessions
**Expected impact**: 0 -> 2 GKs identified per frame

### Phase E: Tracking Upgrade (BoT-SORT or OSNet ReID)

**When**: After detection is solid (20+ players consistently).

**Implementation**:
1. Add OSNet ReID model (lightweight, proven best in SoccerNet 2025)
2. Switch from ByteTrack to BoT-SORT (or add ReID to ByteTrack)
3. Implement tracklet split-and-merge for identity consistency
4. Add track smoothing and interpolation for brief occlusions

**Estimated effort**: 2-3 sessions
**Expected impact**: ID switches reduced by 50%+, stable identities across full match

### Phase F: RF-DETR Evaluation

**When**: After SAHI is working. Test if RF-DETR can detect small players better than YOLO without SAHI.

**Implementation**:
1. Fine-tune RF-DETR on football dataset (Apache 2.0 license -- commercially clean)
2. Compare detection recall at various player sizes vs YOLO + SAHI
3. If better, replace YOLO; if similar, keep YOLO (faster)

**Estimated effort**: 2-3 sessions
**Expected impact**: Potentially +1-2 players in difficult conditions

### Priority Matrix

| Phase | Impact | Effort | Priority |
|-------|--------|--------|----------|
| A: SAHI | +3-5 players/frame | 1-2 sessions | **DO FIRST** |
| B: Pitch mask | -2-5 false positives | 1 session | **DO SECOND** |
| C: SigLIP teams | +7% team accuracy | 1-2 sessions | **DO THIRD** |
| D: GK heuristic | +2 GKs identified | 0.5 session | DO FOURTH |
| E: BoT-SORT/ReID | -50% ID switches | 2-3 sessions | DO FIFTH |
| F: RF-DETR eval | +1-2 edge cases | 2-3 sessions | EVALUATE LATER |

### Target Pipeline Architecture

```
VEO 7680x2160 Frame
       |
  [Lens Undistort] (if calibration available)
       |
  [SAHI Slicing] -> 12 tiles @ 1280x1280 + full image
       |
  [YOLO11-M / RF-DETR] -> per-tile detections
       |
  [SAHI Merge + NMS] -> unified detections
       |
  [Pitch Mask Filter] -> remove sideline/spectator detections
       |
  [BoT-SORT + OSNet ReID] -> tracked player identities
       |
  [SigLIP Team Classification] -> team1 / team2 / referee
       |
  [GK Heuristic] -> identify goalkeepers
       |
  [Pitch Homography] -> 2D pitch coordinates
       |
  [Event Detection] -> passes, shots, formations, etc.
       |
  [AI Coaching] -> tactical recommendations
```

---

## APPENDIX: Quick Reference

### Key Numbers to Remember

- VEO output: 7680x2160 pixels (dual 4K stitched, 180-degree)
- Far-side player height: 20-40 pixels
- Near-side player height: 100-200 pixels
- YOLO comfortable detection zone: 32px+ height
- SAHI recommended tile size: 1280x1280 with 20% overlap
- SigLIP embedding dimension: 768
- OSNet ReID: lightweight, outperforms transformer ReID
- SportSORT HOTA: 88% (SoccerNet SOTA)
- GTATrack HOTA: 60% (fisheye SOTA, more relevant to VEO)
- Processing cost per match (cloud GPU): $1-5
- YOLO26 CPU improvement: 43% faster than YOLO11

### Key URLs

- SAHI GitHub: https://github.com/obss/sahi
- Supervision: https://github.com/roboflow/supervision
- Roboflow Sports: https://github.com/roboflow/sports
- RF-DETR: https://github.com/roboflow/rf-detr
- SoccerNet GameState: https://github.com/SoccerNet/sn-gamestate
- SoccerNet Tracking: https://github.com/SoccerNet/sn-tracking
- TrackLab: https://github.com/TrackingLaboratory/tracklab
- SportsMOT: https://github.com/MCG-NJU/SportsMOT
- SportsLabKit: https://github.com/AtomScott/SportsLabKit
- TrackEval: https://github.com/JonathonLuiten/TrackEval
- Football Detection Dataset (Roboflow): https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- Football Detection Dataset (HuggingFace): https://huggingface.co/datasets/keremberke/football-object-detection
