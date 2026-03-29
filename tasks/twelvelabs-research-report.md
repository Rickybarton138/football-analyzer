# TwelveLabs API Research Report — Manager Mentor

**Date**: 2026-03-28
**Purpose**: Exhaustive technical research for football video analysis platform

---

## 1. TWELVELABS PLATFORM OVERVIEW

TwelveLabs is a multimodal video understanding platform that processes visual, audio, and text modalities simultaneously. It offers four core APIs: **Index**, **Search**, **Analyze**, and **Embed**.

### Models

| Model | Purpose | Current Version | Status |
|-------|---------|----------------|--------|
| **Marengo** | Search + Embeddings | 3.0 | GA (Nov 2025) |
| **Marengo** | Search + Embeddings | 2.7 | Deprecated mid-March 2026 |
| **Pegasus** | Video-to-Text Analysis | 1.2 | GA (Feb 2025) |

**Key**: Marengo powers search/embeddings. Pegasus powers text generation/analysis. Both can be enabled per index.

---

## 2. API ENDPOINTS — FULL DETAILS

### 2.1 POST /analyze (formerly /generate)

Analyzes videos and generates customizable text output based on prompts. Renamed from `/generate` in June 2025. The old `/gist` and `/summarize` endpoints were sunset on Feb 15, 2026 — everything now goes through `/analyze`.

**Parameters:**
- `video_id` (string, required) — Video to analyze
- `prompt` (string, required) — Guides output format/content. Max 2,000 tokens
- `temperature` (float, optional) — Randomness control. Default 0.2, range 0-1
- `response_format` (object, optional) — Structured JSON schema for output
- `max_tokens` (integer, optional) — Max tokens to generate

**Structured JSON Response Example:**
```python
result = client.analyze(
    video_id=video_id,
    prompt="Analyze this football match. Identify formations, key events, and tactical patterns.",
    temperature=0.2,
    response_format=ResponseFormat(
        type="json_schema",
        json_schema={
            "type": "object",
            "properties": {
                "formation": {"type": "string"},
                "key_events": {"type": "array", "items": {"type": "object"}},
                "tactical_patterns": {"type": "array", "items": {"type": "string"}},
            },
        },
    ),
    max_tokens=2000,
)
```

**Streaming Response:**
```python
response = client.analyze_stream(
    video_id="YOUR_VIDEO_ID",
    prompt="What are the main tactical events in this match?",
)
for chunk in response:
    if chunk.event_type == "text_generation":
        print(chunk.text, end="", flush=True)
    elif isinstance(chunk, StreamAnalyzeResponse_StreamEnd):
        print(f"\nFinished: {chunk.finish_reason}")
        print(f"Token usage: {chunk.metadata.usage}")
```

**Football Use Cases:**
- Generate match summaries with structured JSON (title, events, formations, highlights)
- Create chapter-by-chapter breakdowns (first half, second half, key moments)
- Extract tactical analysis with custom prompts
- Generate highlight descriptions for clip metadata

### 2.2 POST /search

Semantic search across indexed videos using text, images, or both.

**Parameters:**
- `index_id` (string, required) — Index to search
- `search_options` (array, required) — `["visual"]`, `["audio"]`, `["visual", "audio"]`, or `["transcription"]` (new in 3.0)
- `query_text` (string, optional) — Natural language query
- `query_media_type` (string, optional) — `"image"` for image queries
- `query_media_url` (string, optional) — URL of query image
- `query_media_file` (file, optional) — Local image file

**Search Types:**
1. **Text search**: `query_text="goalkeeper makes a diving save"`
2. **Image search**: Provide an image to find visually similar moments
3. **Composed search** (Marengo 3.0): Text + image together — e.g., provide image of a player + text "scoring a goal"
4. **Entity search** (Marengo 3.0): `query_text="<@{entity_id}> taking a free kick"`
5. **Transcription search** (Marengo 3.0): Search spoken words with lexical, semantic, or combined matching

**Football Search Examples:**
```python
# Find corner kicks
results = client.search.query(
    index_id=index_id,
    search_options=["visual"],
    query_text="corner kick being taken from the right side"
)

# Find a specific player using entity search
results = client.search.query(
    index_id=index_id,
    search_options=["visual"],
    query_text=f"<@{player_entity_id}> receiving a pass in the penalty area"
)

# Composed search: image of player + context
results = client.search.query(
    index_id=index_id,
    search_options=["visual"],
    query_text="celebrating after scoring",
    query_media_type="image",
    query_media_url="https://example.com/player-photo.jpg"
)
```

**Response Structure:**
```json
{
  "results": [
    {
      "id": "result_id",
      "score": 0.95,
      "start": 120.5,
      "end": 135.2,
      "video_id": "video_id",
      "confidence": "high"
    }
  ],
  "total": 15
}
```

### 2.3 POST /embed/v2 (Synchronous)

Creates multimodal embeddings for similarity search.

**Parameters:**
- `input_type` (string, required) — `"audio"`, `"video"`, `"image"`, `"text"`, `"text_image"`
- `model_name` (string, required) — `"marengo3.0"` only
- Content body varies by input type

**Specs:**
- 512-dimensional embeddings (Marengo 3.0)
- Synchronous for content up to 10 minutes
- Supports text, images, audio, and short video

**Football Use Case:** Generate embeddings for tactical patterns, then use cosine similarity to find matches with similar patterns across your video library.

### 2.4 POST /embed/v2/tasks (Asynchronous)

Creates embeddings for longer content asynchronously.

**Parameters:**
- `input_type` (string, required) — `"audio"` or `"video"`
- `model_name` (string, required) — `"marengo3.0"`
- `video.start_sec` / `video.end_sec` — Time range
- `video.segmentation.dynamic.min_duration_sec` — Min segment duration (default 4s)
- `video.embedding_option` — `["visual", "audio", "transcription"]`
- `video.embedding_scope` — `["clip", "asset"]` (clip-level and/or whole-video)

**Specs:**
- Supports up to 4 hours of video
- Files up to 4 GB (6 GB on Bedrock)
- Dynamic segmentation for intelligent clip boundaries

### 2.5 POST /indexes

Creates an index for organizing videos.

```python
index = client.indexes.create(
    index_name="manager-mentor-matches",
    models=[
        IndexesCreateRequestModelsItem(
            model_name="marengo3.0",
            model_options=["visual", "audio"],
        ),
        IndexesCreateRequestModelsItem(
            model_name="pegasus1.2",
            model_options=["visual", "audio"],
        ),
    ],
    addons=["thumbnail"],
)
```

### 2.6 POST /assets + POST /indexes/{id}/indexed-assets (New Upload Workflow)

The legacy `POST /tasks` is being replaced with a two-step workflow:
1. Upload video via `POST /assets` (direct upload up to 200 MB local / 4 GB URL, or multipart up to 4 GB)
2. Index it via `POST /indexes/{index-id}/indexed-assets`

---

## 3. ENTITY COLLECTIONS — PLAYER IDENTIFICATION

This is the most football-relevant feature. Entity Search allows identifying and locating **specific people** in videos.

### Workflow:

```python
# 1. Upload reference images of a player
asset_ids = []
for photo_url in player_photos:
    asset = client.assets.create(method="url", url=photo_url)
    asset_ids.append(asset.id)

# 2. Create a collection (e.g., per team)
collection = client.entity_collections.create(
    name="Verwood Town FC Squad",
    description="2025-26 season squad photos",
)

# 3. Create entity for each player
player_entity = client.entity_collections.entities.create(
    entity_collection_id=collection.id,
    name="Player Name - #7",
    asset_ids=asset_ids,
    metadata={"position": "Striker", "shirt_number": "7"},
)

# 4. Search for that player in match footage
results = client.search.query(
    index_id=index_id,
    search_options=["visual"],
    query_text=f"<@{player_entity.id}> making a run behind the defence",
)
```

### Limitations:
- **Free plan**: 1 collection, max 15 entities
- **Paid plans**: Multiple collections, higher entity limits
- Requires **Marengo 3.0**
- Need multiple reference images per person (different angles, lighting)
- Uses **face recognition** — requires clear face visibility

### Football-Specific Considerations:
- Works for identifying specific players by face
- Does NOT identify by shirt number (no OCR on jerseys)
- Does NOT identify by team color natively
- Best when players face the camera (celebrations, close-ups, post-match)
- Less reliable during fast-paced play when faces are distant/blurred
- Recommendation: Upload 5-10 reference photos per player from different angles

---

## 4. CAN TWELVELABS DO FOOTBALL-SPECIFIC ANALYSIS?

### What TwelveLabs CAN Do for Football:

| Capability | How | Confidence |
|-----------|-----|-----------|
| **Find specific moments** (goals, fouls, corners) | Text search: "goal scored from outside the box" | HIGH |
| **Identify specific players** (by face) | Entity search with reference photos | MEDIUM |
| **Generate match summaries** | Analyze API with custom prompt | HIGH |
| **Create highlight reels** | Search for high-action moments, get timestamps | HIGH |
| **Detect sports actions** | Marengo 3.0 "sports intelligence" for soccer | HIGH |
| **Composed search** (player photo + action) | Image + text search | MEDIUM-HIGH |
| **Cinematography understanding** | Detects zoom, pan, tracking shots | MEDIUM |
| **Transcription search** | Find moments by what commentators say | HIGH |
| **Structured JSON output** | Return formations/events as parseable data | HIGH |

### What TwelveLabs CANNOT Do:

| Capability | Why Not | Alternative |
|-----------|---------|-------------|
| **Shirt number detection** | No OCR on jerseys | Custom YOLO model (Roboflow/SoccerNet) |
| **Team color classification** | Not a native feature | OpenCV color segmentation / YOLO team classifier |
| **Player tracking (positions)** | No coordinate output | OpenCV + ByteTrack / Roboflow supervision |
| **Tactical formation detection** | No spatial awareness output | Custom CV pipeline (pitch detection + player positions) |
| **Ball tracking** | Not designed for this | YOLO ball detection model |
| **Heatmap generation** | No positional data | Homography + tracking pipeline |
| **Real-time live analysis** | API is batch-only | Edge deployment with YOLO |
| **Speed/distance metrics** | No positional tracking | GPS data or pitch calibration + tracking |

### The Critical Insight for Manager Mentor:

TwelveLabs is excellent for **semantic understanding** (what is happening, who is involved, finding moments by description) but lacks **spatial/positional analysis** (where players are, formations, tracking coordinates). The optimal architecture combines:

1. **TwelveLabs** — Semantic layer (search, analysis, highlights, player identification)
2. **OpenCV + YOLO** — Spatial layer (player detection, tracking, pitch homography, formations)
3. **Claude** — Interpretation layer (combining both into coaching insights)

---

## 5. VIDEO REQUIREMENTS

### Marengo 3.0 (Search/Embeddings):

| Spec | Requirement |
|------|------------|
| **Duration** | 4 seconds to 4 hours |
| **File size** | Up to 4 GB (6 GB on Bedrock) |
| **Resolution** | 360x360 to 5184x2160 px |
| **Aspect ratio** | Between 1:2.4 and 2.4:1 (includes 16:9, 4:3, 1:1) |
| **Formats** | All FFmpeg-supported formats (MP4, MOV, AVI, MKV, etc.) |
| **Audio formats** | WAV, MP3, FLAC |
| **Audio sync** | A/V streams must not differ by > 0.5 seconds |
| **Image requirements** | Min 128x128 px, max 5 MB (for entity/search) |
| **Text query length** | Up to 500 tokens |

### Pegasus 1.2 (Analysis/Text Generation):

| Spec | Requirement |
|------|------------|
| **Duration** | 4 seconds to 2 hours |
| **File size** | Up to 2 GB |
| **Prompt length** | Up to 2,000 tokens |

### Upload Methods:
- **Direct upload (local)**: Up to 200 MB
- **Direct upload (URL)**: Up to 4 GB
- **Multipart upload**: Up to 4 GB
- URLs must be direct links to raw video files (no YouTube/cloud sharing links)

---

## 6. RATE LIMITS & PRICING

### Pricing (Developer Plan — Pay As You Go):

| API | Cost |
|-----|------|
| **Video Indexing (Marengo)** | $0.042 / minute |
| **Infrastructure Fee** | $0.0015 / minute |
| **Search API** | $4 / 1,000 queries |
| **Analyze API (Pegasus input)** | $0.021 / minute |
| **Analyze API (output)** | $0.0075 / 1K tokens |
| **Embed API (Video)** | $0.0083 / minute |
| **Embed API (Audio)** | $0.10 / 1,000 requests |
| **Embed API (Image/Text)** | $0.07 / 1,000 requests |

### Cost Estimate for Manager Mentor:

A typical 90-minute match:
- **Index**: 90 min x $0.0435 = **$3.92**
- **Analyze** (3 calls, full match): 90 min x $0.021 x 3 + tokens = **~$6.50**
- **Search** (50 queries per match): 50 x $0.004 = **$0.20**
- **Total per match: ~$10.62**

### Free Tier:
- 600 minutes of video indexing (one-time, accumulated)
- 90-day index access
- 100 videos per index
- 50 requests/day per API
- 8 requests/minute per API

### Rate Limits (Developer Tier 1):

| API | RPD | RPM | Duration/Day | Duration/Hour |
|-----|-----|-----|-------------|--------------|
| Index | 3,000 | 60 | 3,000 min | 600 min |
| Upload | 3,000 | 60 | - | - |
| Search | 3,000 | 600 | - | - |
| Analyze | 1,000 | 60 | 3,000 min | 600 min |
| Embed (Video) | 3,000 | 25 | 3,000 min | 600 min |

Tiers auto-upgrade based on monthly spend: Tier 2 at $200/mo, Tier 3 at $400/mo.

---

## 7. LANGUAGE SUPPORT

Marengo 3.0 supports queries in **37 languages** including English, Arabic, Bengali, Chinese, Croatian, Czech, Danish, Dutch, Farsi, Filipino, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Telugu, Thai, Turkish, Ukrainian, Vietnamese, and more.

Search queries can be in any supported language regardless of the video's language.

---

## 8. NEW FEATURES (2025-2026 TIMELINE)

| Date | Feature | Relevance to MM |
|------|---------|----------------|
| **Mar 2026** | Multiple-image search (up to 10 images + text) | Could search with multiple player photos at once |
| **Mar 2026** | Multiple-image embeddings with fused modalities | Better player similarity matching |
| **Feb 2026** | Marengo 2.7 deprecated, auto-reindex to 3.0 | Must use 3.0 going forward |
| **Feb 2026** | /gist and /summarize sunset | Use /analyze for everything |
| **Jan 2026** | Enhanced rate limiting (multi-dimensional) | Plan around DPH/DPD limits |
| **Dec 2025** | Pegasus 1.2 in 23+ AWS regions | Better global availability |
| **Nov 2025** | **Marengo 3.0 GA** — sports intelligence, entity search, 4hr video, 512d embeddings | Core upgrade for MM |
| **Sep 2025** | Entity Search API launched | Player identification feature |
| **Sep 2025** | Structured JSON responses for /analyze | Machine-readable match analysis |
| **Jul 2025** | User-defined metadata on videos | Tag videos with match data |
| **May 2025** | Transcription retrieval in search responses | Get commentary text with results |

### Fine-Tuning:
- Available for **selected customers only** (contact sales)
- Supported on Marengo 2.7+
- Could train on grassroots football footage for better action recognition
- Evaluation includes mAP metrics
- Deployment within days
- **Recommendation**: Worth pursuing once we have 100+ annotated match clips

---

## 9. COMPETITOR COMPARISON

### Google Video Intelligence API

| Aspect | Details |
|--------|---------|
| **Strengths** | 20,000+ object/scene labels, object tracking with bounding boxes, person detection with attributes, shot change detection, text detection (OCR) |
| **Weaknesses** | No semantic natural language search, no entity collections, no video-to-text generation, no embeddings API |
| **Sports Use** | Can detect "person" + bounding boxes, track objects across frames. No sports-specific intelligence. |
| **Pricing** | ~$0.10/min for label detection, $0.05/min for shot detection |
| **Verdict** | Good for low-level detection but lacks high-level semantic understanding. No match analysis capability. |

### AWS Rekognition Video

| Aspect | Details |
|--------|---------|
| **Strengths** | Custom Labels (train on ~200 images), person tracking, face recognition, celebrity recognition, streaming video analysis |
| **Weaknesses** | Custom Labels requires training pipeline, no natural language search, limited to predefined label types, no video-to-text |
| **Sports Use** | Could train Custom Labels for specific sports actions. Person tracking with bounding boxes. Face recognition for players. |
| **Pricing** | ~$0.12/min stored video, $0.10/min streaming. Free tier: 2 training hrs/mo, 1 inference hr/mo |
| **Verdict** | Better for surveillance-style analysis. Custom Labels could work for sports but requires significant training effort. |

### Azure AI Video Indexer

| Aspect | Details |
|--------|---------|
| **Strengths** | 30+ AI models, custom Person models (up to 50 per account), face identification, speaker recognition, OCR, emotion detection, real-time analysis (preview), topic extraction |
| **Weaknesses** | Less flexible than TwelveLabs for custom queries, no multimodal embeddings, person model limited to 50 models per account |
| **Sports Use** | Custom Person models per sport. Could identify players by face. New real-time analysis in preview (2025). |
| **Pricing** | Pay-per-minute model, varies by analysis type |
| **Verdict** | Most feature-rich of the cloud providers for video analysis. Person models are similar to TwelveLabs entities but less flexible. Real-time preview is interesting. |

### Roboflow + YOLO (Custom CV Pipeline)

| Aspect | Details |
|--------|---------|
| **Strengths** | Full control, open source, real-time capable, spatial/positional data, bounding boxes with coordinates, custom model training, pitch keypoint detection, team color classification, ball tracking |
| **Weaknesses** | Requires training data, more engineering effort, no semantic understanding, no natural language search, no video-to-text generation |
| **Sports Use** | **Best for spatial analysis**: player positions, formation detection, pitch homography, speed/distance, heatmaps. Pre-trained football models available on Roboflow Universe. |
| **Key Models** | `football-players-detection`, `football-field-detection` (keypoints), YOLO v8/v9/v11, RF-DETR |
| **Pricing** | Roboflow: free tier (1,000 API calls/mo), Starter $249/mo. Self-hosted YOLO: free |
| **Verdict** | **Essential complement to TwelveLabs.** Provides the spatial layer that TwelveLabs lacks. |

### Comparison Matrix

| Feature | TwelveLabs | Google VI | AWS Rekognition | Azure VI | Roboflow/YOLO |
|---------|-----------|-----------|----------------|----------|--------------|
| NL search | **YES** | No | No | Partial | No |
| Player ID (face) | **YES** | No | YES | YES | No |
| Player tracking (coords) | No | YES | YES | No | **YES** |
| Shirt number OCR | No | Partial | No | Partial | **Custom** |
| Team color detection | No | No | No | No | **Custom** |
| Formation detection | No | No | No | No | **Custom** |
| Ball tracking | No | Partial | No | No | **YES** |
| Match summaries | **YES** | No | No | No | No |
| Embeddings | **YES** | No | No | No | No |
| Structured JSON output | **YES** | No | No | No | No |
| Sports intelligence | **YES** | No | No | No | Custom |
| Real-time | No | No | YES | Preview | **YES** |
| Fine-tuning | Selected | No | YES | YES | **YES** |
| Cost per 90min match | ~$10 | ~$9 | ~$11 | ~$8 | Free (self-hosted) |

---

## 10. RECOMMENDED ARCHITECTURE FOR MANAGER MENTOR

### Tier 1: TwelveLabs (Semantic Layer)
- **Index all match videos** with Marengo 3.0 + Pegasus 1.2
- **Entity collections** per team squad (reference photos of each player)
- **Analyze API** for match summaries, key events, tactical descriptions
- **Search API** for finding specific moments ("counter-attack leading to goal")
- **Embed API** for cross-match similarity (find similar tactical patterns across matches)

### Tier 2: OpenCV + YOLO (Spatial Layer)
- **Player detection** with pre-trained football YOLO model
- **Team classification** by jersey color (K-means clustering on detected players)
- **Pitch homography** using keypoint detection to map to 2D top-down view
- **Player tracking** with ByteTrack/BoTSORT for consistent IDs across frames
- **Formation detection** from 2D positions (classify 4-4-2, 4-3-3, etc.)
- **Ball tracking** for possession and passing analysis

### Tier 3: Claude (Interpretation Layer)
- Combine TwelveLabs semantic analysis + OpenCV spatial data
- Generate coaching recommendations in natural language
- Provide formation comparisons and tactical suggestions
- Create training session plans based on match analysis

### Data Flow:
```
Upload -> Mux (host) -> TwelveLabs (index + analyze)
                     -> OpenCV/YOLO (track + detect)
                     -> Claude (interpret both)
                     -> Coach Dashboard (display)
```

---

## 11. IMMEDIATE ACTION ITEMS

1. **Verify our index uses Marengo 3.0** (2.7 being deprecated mid-March 2026 — may already be auto-migrated)
2. **Create entity collections** for Verwood Town FC squad with reference photos
3. **Test /analyze with structured JSON** for match analysis output
4. **Build OpenCV pipeline** for the spatial analysis TwelveLabs cannot provide
5. **Explore fine-tuning** for grassroots football footage (contact sales@twelvelabs.io)
6. **Budget**: At ~$10/match, plan for pricing at scale (100 matches/season = ~$1,000)
