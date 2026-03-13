# Video Upload & Processing Research
> **Date**: 2026-03-08
> **Purpose**: Understand VEO/Hudl/Trace/Catapult video file sizes, upload processes, and best practices for handling multi-GB video uploads in Manager Mentor
> **Status**: Complete

---

## Table of Contents
1. [VEO Video File Sizes & Specs](#1-veo-video-file-sizes--specs)
2. [VEO Upload & Processing Pipeline](#2-veo-upload--processing-pipeline)
3. [Competitor Upload Approaches](#3-competitor-upload-approaches)
4. [Best Practices for Multi-GB Video Uploads](#4-best-practices-for-multi-gb-video-uploads)
5. [Upload Strategy Comparison](#5-upload-strategy-comparison)
6. [Recommended Architecture for Manager Mentor](#6-recommended-architecture-for-manager-mentor)

---

## 1. VEO Video File Sizes & Specs

### Camera Generations & Storage

| Camera | Internal Storage | Games per Camera | Notes |
|--------|-----------------|------------------|-------|
| **Veo Cam 1** | 60 GB | ~4 full matches | Oldest generation |
| **Veo Cam 2** | 120 GB | ~8 full matches | Previous generation |
| **Veo Cam 3** | 120 GB (estimated) | ~8 full matches | Current generation, improved upload speed |

### Recording Specifications

| Spec | Veo Cam 3 | Notes |
|------|-----------|-------|
| **Capture Resolution** | Dual 4K (3840x2160) per lens | Two fisheye lenses stitched into panoramic |
| **Output Resolution** | 1080p HD (delivered to users) | 4K captured, downscaled after processing |
| **Frame Rate** | 30 FPS | Fixed |
| **HDR** | Yes (4K HDR labelled on lens) | Wide dynamic range for outdoor conditions |
| **Field of View** | ~180 degrees panoramic | Two lenses stitched |
| **Battery Life** | ~7 hours continuous recording | |
| **Codec** | Likely H.264/H.265 | Based on industry standard for camera recording |

### Estimated File Sizes

Based on cross-referencing multiple data points:

| Duration | Estimated File Size | Source/Method |
|----------|-------------------|---------------|
| **90-min match (VEO)** | **~12-16 GB** | Derived from: upload at 20 Mbps takes 1.5-2 hours = ~13.5-18 GB; Veo 1 stores ~4 games in 60 GB = ~15 GB each |
| **70-min match (2x35 halves)** | ~10-12 GB | Soccer Stripes reported ~2 hours upload on hotspot, ~30-40 min on Google Fiber |
| **Veo Go (phone recording)** | ~6 GB per phone per 1.5 hours | Confirmed by VEO Help Center: "a 1.5-hour recording takes up about 6 GB of storage on each camera phone" |

### Calculated Bitrate

Working backwards from file sizes:
- 15 GB for 90 minutes = ~22 Mbps average bitrate
- This is consistent with dual-4K capture compressed to high-quality H.264/H.265
- The raw panoramic capture (dual 4K stitched) would be ~7680x2160 at 30fps before processing
- VEO likely records at ~20-25 Mbps bitrate internally on the camera

### Key Insight for Manager Mentor
**Users uploading standard camera footage (single camera, 1080p, 30fps) would have files of ~4-8 GB per 90 minutes** depending on bitrate:
- 1080p @ 8 Mbps H.264 = ~5.4 GB per 90 min
- 1080p @ 12 Mbps H.264 = ~8.1 GB per 90 min
- 4K @ 20 Mbps H.265 = ~13.5 GB per 90 min
- 4K @ 35 Mbps H.264 = ~23.6 GB per 90 min

---

## 2. VEO Upload & Processing Pipeline

### Upload Process (Camera to Cloud)

1. **Recording stored on camera** - Video is saved to internal SSD during recording
2. **Connection required** - Camera connects via Wi-Fi or Ethernet (USB-C dongle for Cam 2/3)
3. **Upload initiated** - Can be automatic (on connection) or manual (via Veo Camera App)
4. **Progress tracking** - Percentage shown in Veo Camera App, also visible at app.veo.co under "Recordings"
5. **Resumable** - Users can pause uploads (unplug camera for another game), and upload resumes from where it left off when reconnected
6. **Auto-deletion** - Once fully uploaded to cloud, recording is automatically removed from camera storage

### Upload Speeds (Real World Data)

| Connection | 90-min Match Upload Time | Effective Speed |
|------------|------------------------|-----------------|
| 20 Mbps Wi-Fi | 1.5-2 hours | ~15-20 Mbps utilized |
| Hotel Wi-Fi | 5-13 hours | Often fails entirely (captive portal issue) |
| Google Fiber (1 Gbps) | 30-40 minutes | VEO bottlenecks to ~50-60 Mbps |
| 500 Mbps connection | ~3 hours | Users complain VEO throttles to ~10-15 Mbps |

### Critical Pain Point: VEO Throttles Upload Speed
Multiple users report that VEO's upload server is the bottleneck, NOT their internet connection. Users with 500 Mbps upload speeds still see 3+ hour uploads. One user noted: "uploading a recording to YouTube of a similar size takes considerably less time." This is a **massive competitive opportunity** for Manager Mentor.

### Cloud Processing Pipeline (After Upload)

| Stage | Duration | What Happens |
|-------|----------|-------------- |
| **Preparing** | 5-15 minutes | Initial file validation and preparation |
| **Processing** | 1-2 hours | AI analysis: stitching, auto-follow cam generation, event detection, clip creation |
| **Follow-cam** | Available first | AI-tracked panoramic-to-zoomed view |
| **AI Clips** | Available during processing | Auto-detected highlights |
| **Detected Events** | Hours after upload | Goals, shots, corners etc. |
| **Veo Analytics** | Last to complete | Full statistical analysis |

### Total Time: Recording End to Video Available
- **Best case**: 2-3 hours (fast internet, short game)
- **Typical case**: 4-6 hours
- **Tournament/travel**: 24-48 hours (hotel Wi-Fi issues, multiple games queued)
- **Worst reported**: 47+ hours

### Veo Go (Phone-based) Upload Process
- Two phones record one half of the pitch each
- After the match, both video files are uploaded to the Veo platform
- VEO's AI automatically stitches the two perspectives into one full match video
- VEO recommends uploading both halves as soon as the game ends
- Processing creates two video types after both halves uploaded

---

## 3. Competitor Upload Approaches

### Hudl

**Scale**: 200,000+ teams, 6M+ users, 100+ PB of stored video

**Architecture** (confirmed from AWS case study):
- **Cloud**: 100% AWS
- **Storage**: Amazon S3 (100+ petabytes)
- **Upload acceleration**: Amazon S3 Transfer Acceleration (20%+ speed improvement)
- **Encoding**: 2,000+ servers spun up on Friday nights for video encoding alone
- **CDN**: Amazon CloudFront for delivery
- **Caching**: Amazon ElastiCache (Redis) for real-time data feeds
- **Data Warehouse**: Amazon Redshift
- **Peak load**: 39 hours of video uploaded every minute during football season

**Upload Process**:
- Web uploader (browser-based, no software install needed)
- Also has Hudl Mercury desktop app (being deprecated in favor of web uploader)
- Users copy video files from camera to computer, then upload via browser
- Typical upload: 3-4 hours for a full game
- Hudl supports multiple video formats and auto-transcodes

**Encoding/Compression**:
- Recently migrated from pure H.264 (x264) to Visionular Aurora4 (H.264) and Aurora5 (H.265)
- Achieved 50% faster encoding, 17-19% quality improvement (VMAF), 30% higher encoding FPS
- This reduced their 100+ PB storage costs significantly
- Encoding ladder: multiple bitrates/resolutions for adaptive streaming

### Trace

**File Size**: ~16 GB per standard match (2 x 45-minute halves) - confirmed by Trace Help Center

**Upload Process**:
- Camera connects via Ethernet cable (hardwired only, no Wi-Fi)
- Camera must stay plugged into power during entire upload
- Estimated upload time: 3 hours 49 minutes at 10 Mbps
- Minimum recommended: 5 Mbps upload speed
- **Resumable**: Users can unplug mid-upload, record another game, plug back in and upload resumes
- Processing: Up to 12 hours after upload completes

**Key Limitations**:
- Ethernet-only means no hotel uploads (same problem as VEO)
- 12-hour processing time is very long
- No live streaming capability

### Catapult Pro Video

**Architecture**:
- Cloud-based platform (AWS-hosted, confirmed from AWS IoT blog)
- Designed for elite/professional teams (not consumer)
- Supports multi-angle video capture from mobile devices
- Live streaming from multiple mobile devices
- Raw files uploaded directly to cloud, auto-processed
- Up to 30 raw files can be stored on a device at once
- Integrates with GPS tracking data (Catapult Vector/PlayerTek)
- Imports data from Hudl SportsCode, Second Spectrum, Tracab, NBA Hawkeye

**Upload Pattern**:
- Professional setup: dedicated analysts handle upload
- Mobile app: "Upload raw files directly to the cloud - file automatically processed in the web"
- Focus is on data integration (GPS + video sync) rather than raw video upload speed

### Summary Comparison

| Platform | File Size (90 min) | Upload Method | Upload Time | Processing Time | Resumable |
|----------|-------------------|---------------|-------------|-----------------|-----------|
| **VEO** | ~12-16 GB | Wi-Fi / Ethernet | 1.5-13 hours | 1-2 hours | Yes |
| **Trace** | ~16 GB | Ethernet only | ~3.5-4 hours @ 10 Mbps | Up to 12 hours | Yes |
| **Hudl** | Varies (user camera) | Browser upload | 3-4 hours typical | Minutes-hours | Partial |
| **Catapult** | Varies | Cloud direct / App | Professional setup | Cloud processing | Yes |

---

## 4. Best Practices for Multi-GB Video Uploads

### The Core Challenge
A 90-minute football match at 1080p generates 4-16 GB of video. This creates several problems:
- Browser HTTP request timeouts (most default to 2-5 minutes)
- Network interruptions cause full re-upload
- Server memory exhaustion if buffering entire file
- Progress tracking is opaque to users
- Mobile networks are unreliable

### Industry Best Practices

#### 1. Never Upload Through Your Backend Server
- Use **pre-signed URLs** to upload directly from browser/client to object storage (S3, GCS, Azure Blob)
- Backend only generates the signed URL and tracks metadata
- This avoids: backend memory issues, double bandwidth costs, server CPU bottleneck
- YouTube, Vimeo, Cloudflare Stream, and all major platforms do this

#### 2. Always Use Chunked/Multipart Upload
- Split files into 5-100 MB chunks on the client side
- Upload chunks in parallel (3-6 concurrent uploads typical)
- Each chunk gets its own pre-signed URL
- Failed chunks can be retried individually without re-uploading the whole file
- AWS S3 requires multipart upload for files over 5 GB, recommends it for files over 100 MB

#### 3. Make Uploads Resumable
- Track which chunks have been successfully uploaded
- On reconnection, only upload remaining chunks
- Store upload state in localStorage (browser) or local DB (mobile)
- The TUS protocol was designed specifically for this

#### 4. Stream Processing on the Server
- Never load the entire file into memory
- Use streaming parsers (e.g., `streaming-form-data` for Python)
- Process chunks as they arrive
- FastAPI's `request.stream()` allows processing byte chunks without full memory load

#### 5. Show Granular Progress
- Per-chunk progress tracking
- Overall upload percentage
- Estimated time remaining
- Upload speed indicator
- This is critical for user experience with multi-hour uploads

---

## 5. Upload Strategy Comparison

### Strategy 1: TUS Protocol (Resumable Upload Protocol)

**What it is**: Open protocol specifically designed for resumable file uploads (tus.io)

**How it works**:
1. Client sends POST to create upload resource (includes file size metadata)
2. Server responds with upload URL
3. Client sends PATCH requests with file chunks + `Upload-Offset` header
4. Server tracks offset; client can resume from last confirmed offset
5. On completion, server triggers processing

**Pros**:
- Purpose-built for resumable uploads
- Official implementations in every language (Python, JS, Go, Java, etc.)
- FastAPI implementations available: `fastapi-tusd`, `tuspyserver`
- Cloudflare Stream requires TUS for files > 200 MB
- Handles network failures gracefully
- Simple protocol (HTTP-based)

**Cons**:
- Uploads go through your server (not direct to S3 by default)
- Need to proxy to S3 or use tusd with S3 backend
- Additional server to maintain (tusd reference server)

**FastAPI Implementation** (`tuspyserver`):
```python
from tuspyserver import create_tus_router
from fastapi import FastAPI

app = FastAPI()
app.include_router(
    create_tus_router(
        files_dir="./uploads",
        on_upload_complete=process_video,
    )
)
```

### Strategy 2: S3 Multipart Upload with Pre-signed URLs

**What it is**: Upload directly from browser to S3 using pre-signed URLs for each part

**How it works**:
1. Client requests upload initiation from backend
2. Backend calls `CreateMultipartUpload` on S3, returns `UploadId`
3. Client requests pre-signed URLs for each part (5 MB - 100 MB each)
4. Client uploads parts directly to S3 in parallel using pre-signed URLs
5. Client reports ETags for each completed part to backend
6. Backend calls `CompleteMultipartUpload` on S3

**Pros**:
- **Zero load on your backend** (data goes direct to S3)
- Massive parallelism (upload 6+ parts simultaneously)
- Built-in to AWS S3 (no extra servers)
- Can combine with S3 Transfer Acceleration (20%+ speed boost)
- Scales infinitely (S3 handles the load)
- Failed parts retry individually
- AWS test results: 485 MB file baseline 72s, with multipart + acceleration = 20s (72% faster)

**Cons**:
- More complex client-side code
- Pre-signed URLs expire (default 15 min, configurable)
- Minimum part size 5 MB (S3 requirement)
- Need to track part ETags and complete upload
- Requires AWS SDK on client side or custom implementation

**Architecture Pattern**:
```
Browser                    Backend (FastAPI)              AWS S3
  |                            |                            |
  |--POST /upload/init-------->|                            |
  |                            |--CreateMultipartUpload---->|
  |<--{uploadId, presignedURLs}|                            |
  |                            |                            |
  |--PUT part1 (direct)--------|--------------------------->|
  |--PUT part2 (direct)--------|--------------------------->|
  |--PUT part3 (direct)--------|--------------------------->|
  |  (parallel uploads)        |                            |
  |                            |                            |
  |--POST /upload/complete---->|                            |
  |  {uploadId, parts[]}       |--CompleteMultipartUpload-->|
  |<--{success, videoId}       |                            |
```

### Strategy 3: Chunked Upload Through Backend (FastAPI)

**What it is**: Client splits file into chunks, sends each as a separate POST to your FastAPI backend

**How it works**:
1. Client splits file into chunks (e.g., 5-50 MB)
2. Each chunk sent as separate POST with chunk number + total chunks metadata
3. Backend saves chunks to disk as they arrive
4. After final chunk, backend reassembles file and begins processing

**FastAPI Example**:
```python
@app.post("/uploads")
async def upload_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    chunk_number: int = Form(0),
    total_chunks: int = Form(1),
):
    chunk_dir = f"./temp/{name}"
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = f"{chunk_dir}/{chunk_number}"

    async with aiofiles.open(chunk_path, 'wb') as f:
        while content := await file.read(1024 * 1024):
            await f.write(content)

    if chunk_number == total_chunks - 1:
        await reassemble_and_process(name, chunk_dir, total_chunks)

    return {"chunk": chunk_number, "status": "received"}
```

**Pros**:
- Simple to implement
- Full control over the upload process
- Can add authentication/validation per chunk
- No external dependencies (no S3 required initially)

**Cons**:
- All data flows through your backend (bandwidth cost, CPU load)
- Must handle reassembly logic
- Resumability requires custom tracking
- Doesn't scale as well as direct-to-S3

### Strategy 4: FastAPI Streaming Upload (request.stream())

**What it is**: Use Starlette's streaming request body to process file data as it arrives, avoiding memory issues

**How it works**:
1. Client sends entire file as request body (no multipart form)
2. Backend reads the request body as an async stream
3. Data is written to disk (or piped to S3) chunk by chunk
4. Never loads full file into memory

```python
@app.post('/upload')
async def upload(request: Request):
    filename = request.headers['filename']
    async with aiofiles.open(filepath, 'wb') as f:
        async for chunk in request.stream():
            await f.write(chunk)
    return {"message": f"Uploaded: {filename}"}
```

**Pros**:
- Memory-efficient (never loads full file)
- Simpler than chunked upload
- Fast (benchmarks show it's the fastest FastAPI upload method)

**Cons**:
- No built-in resumability
- Single HTTP connection (if it drops, restart from zero)
- Still flows through backend
- Not suitable for unreliable networks

### Strategy 5: Client-Side Compression Before Upload

**What it is**: Compress/transcode video in the browser before uploading

**Technologies**: FFmpeg.wasm (WebAssembly), MediaRecorder API, Web Codecs API

**Pros**:
- Dramatically reduces upload size (e.g., 16 GB -> 4 GB)
- Reduces upload time proportionally
- Standardizes input format for processing pipeline

**Cons**:
- Very CPU intensive on client device
- FFmpeg.wasm is slow (10-30x slower than native)
- Web Codecs API has limited browser support
- Bad UX: user waits for compression THEN upload
- Most users' devices may not handle it well

**Verdict**: Not recommended for initial implementation. Better to accept original files and transcode server-side.

### Strategy Comparison Matrix

| Strategy | Resumable | Scalable | Backend Load | Complexity | Best For |
|----------|-----------|----------|-------------|------------|----------|
| **TUS Protocol** | Excellent | Medium | High | Low | MVP / simple deployments |
| **S3 Multipart + Pre-signed** | Good | Excellent | None | Medium-High | Production at scale |
| **Chunked via Backend** | Good | Low | High | Medium | Prototyping |
| **Streaming (request.stream)** | None | Medium | High | Low | Small files only |
| **Client Compression** | N/A | N/A | None | Very High | Future optimization |

---

## 6. Recommended Architecture for Manager Mentor

### Phase 1: MVP (Start Here)

**Strategy: TUS Protocol via FastAPI**

Why: Fastest to implement, resumable out of the box, well-tested libraries.

```
Client (Browser)
    |
    |-- TUS upload (chunked, resumable) -->  FastAPI + tuspyserver
    |                                            |
    |                                            |--> Save to local disk
    |                                            |--> Trigger processing pipeline
    |                                            |--> Move to S3 after processing
```

**Implementation**:
- Use `tuspyserver` or `fastapi-tusd` package
- Set chunk size to 10-25 MB
- Use `on_upload_complete` hook to trigger video processing
- Client: use `tus-js-client` library (official, well-maintained)
- Add progress bar, speed indicator, ETA in the UI

**Estimated effort**: 1-2 days for basic upload, 1 day for progress UI

### Phase 2: Production Scale

**Strategy: S3 Multipart Upload with Pre-signed URLs + Transfer Acceleration**

Why: Zero backend load, infinite scale, fastest uploads globally.

```
Client (Browser)
    |
    |-- POST /api/upload/init ---------->  FastAPI Backend
    |<-- {uploadId, presignedURLs[]}         |
    |                                        |--> CreateMultipartUpload (S3)
    |                                        |--> Generate pre-signed URLs
    |
    |-- PUT part1 ---------------------->  S3 (Transfer Acceleration)
    |-- PUT part2 ---------------------->  S3 (via nearest edge)
    |-- PUT part3 ---------------------->  S3 (parallel)
    |   ...
    |
    |-- POST /api/upload/complete ------>  FastAPI Backend
    |                                        |--> CompleteMultipartUpload (S3)
    |                                        |--> Trigger Lambda/processing
    |<-- {videoId, status: processing}
    |
    |                                     S3 Event Notification
    |                                        |--> Lambda / SQS
    |                                        |--> Start AI processing pipeline
```

**Key Design Decisions**:
- Part size: 25 MB (good balance of parallelism vs overhead)
- Parallel uploads: 4-6 concurrent (browser connection limit)
- Pre-signed URL expiry: 1 hour (generous for slow connections)
- Transfer Acceleration: enabled (20%+ speed boost, ~$0.04/GB cost)
- Upload state persisted in localStorage for resumability across page refreshes

### How We Beat VEO on Upload Speed

| VEO Pain Point | Manager Mentor Advantage |
|---------------|-------------------------|
| Upload throttled to ~15 Mbps regardless of connection | Direct-to-S3 with Transfer Acceleration, no throttle |
| 1.5-13 hours for a game | Target: 15-30 min on decent broadband |
| No upload during recording | Users upload from any device, any time |
| Hotel Wi-Fi captive portal blocks camera | Browser-based upload works on any network |
| Must wait for processing before viewing | Progressive processing: show raw video immediately, enhance over time |
| 24-48 hour tournament delay | Target: playback within minutes, AI analysis within 1-2 hours |

### File Size Budget for Manager Mentor

Since users will upload from standard cameras (GoPro, iPhone, DSLR, etc.) rather than VEO's proprietary camera:

| Source | Typical Resolution | Bitrate | 90-min Size |
|--------|-------------------|---------|-------------|
| iPhone 15/16 (1080p) | 1920x1080 | 10-15 Mbps | 6.8-10 GB |
| iPhone 15/16 (4K) | 3840x2160 | 30-50 Mbps | 20-34 GB |
| GoPro Hero 12 (1080p) | 1920x1080 | 12-20 Mbps | 8-13.5 GB |
| GoPro Hero 12 (4K) | 3840x2160 | 60-100 Mbps | 40-67 GB |
| DSLR/Mirrorless (1080p) | 1920x1080 | 15-28 Mbps | 10-19 GB |
| Tablet on tripod | 1920x1080 | 8-12 Mbps | 5.4-8 GB |

**Design for**: 5 GB minimum, 20 GB typical, 50 GB maximum per upload

### Storage Cost Estimates (AWS S3)

| Volume | Monthly Storage | Transfer Out | Total Monthly |
|--------|----------------|-------------|---------------|
| 100 matches/month @ 10 GB avg | 1 TB = ~$23 | 1 TB = ~$90 | ~$113 |
| 500 matches/month @ 10 GB avg | 5 TB = ~$115 | 5 TB = ~$450 | ~$565 |
| 1000 matches/month @ 10 GB avg | 10 TB = ~$230 | 10 TB = ~$900 | ~$1,130 |

**Cost optimization**: Transcode to H.265 after upload (50% size reduction), move to S3 Glacier after 90 days, use CloudFront for delivery caching.

---

## Key Takeaways

1. **VEO files are ~12-16 GB per match** (dual 4K capture, ~22 Mbps bitrate). Standard user uploads will be **5-20 GB**.

2. **VEO's biggest pain point is upload speed** -- they throttle uploads server-side, causing 1.5-13 hour waits. This is our single biggest competitive advantage.

3. **All competitors use cloud processing** with significant delays (1-12 hours). Offering immediate raw playback + progressive AI enhancement would be a differentiator.

4. **S3 Multipart Upload with Pre-signed URLs** is the industry standard for production-scale video upload (used by Hudl, YouTube, Vimeo, etc.).

5. **TUS protocol** is the best choice for MVP -- simple, resumable, well-supported in Python/FastAPI.

6. **Never proxy video data through your backend** at scale -- use direct-to-S3 uploads with pre-signed URLs.

7. **Hudl processes 39 hours of video per minute** at peak, using 2,000+ encoding servers on AWS. Our architecture must be designed to scale similarly from day one.

8. **Trace requires Ethernet-only uploads**, VEO struggles with hotel Wi-Fi captive portals. Browser-based upload from any device on any network is a massive UX advantage.
