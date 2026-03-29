# Mux Capabilities Report for Manager Mentor (Football Analyzer)

**Date**: 2026-03-28
**Purpose**: Comprehensive technical audit of every Mux feature available for football video analysis
**Current Integration**: `backend/app/services/mux_service.py` — basic upload, asset creation, clip URLs, thumbnails, GIFs

---

## 1. FEATURE-BY-FEATURE BREAKDOWN

### 1.1 Instant Clipping (Time-Based URLs)

**What it does**: Extract any segment of a video without re-encoding. Pure URL manipulation.

**How it works**:
```
# HLS clip — plays only seconds 45 to 60
https://stream.mux.com/{PLAYBACK_ID}.m3u8?asset_start_time=45&asset_end_time=60

# Thumbnail at exact moment
https://image.mux.com/{PLAYBACK_ID}/thumbnail.jpg?time=47.5

# Animated GIF of the moment
https://image.mux.com/{PLAYBACK_ID}/animated.gif?start=45&end=55

# Storyboard respecting clip boundaries
https://image.mux.com/{PLAYBACK_ID}/storyboard.vtt?format=webp&asset_start_time=45&asset_end_time=60
```

**Parameters**:
- `asset_start_time` / `asset_end_time` — seconds (float), relative to asset start
- `program_start_time` / `program_end_time` — epoch integers for live stream clips
- Both start and end are optional (omit start = from beginning, omit end = to end)

**Football use case**: When AI says "good pressing trigger at 23:15", we generate a clip URL instantly. Zero processing time. Zero extra storage cost.

**Current status**: Already implemented in `mux_service.py` via `get_clip_url()`.

**Security**: For signed playback, embed `asset_start_time` and `asset_end_time` as JWT claims. This prevents users from manipulating the URL to access the full video when they should only see clips.

---

### 1.2 Storyboards / Sprite Sheets (Timeline Hover Previews)

**What it does**: Auto-generates a grid of thumbnails from evenly-spaced frames. When a user hovers over the scrub bar, they see a preview of that moment.

**URLs**:
```
# Storyboard image (sprite sheet)
https://image.mux.com/{PLAYBACK_ID}/storyboard.jpg
https://image.mux.com/{PLAYBACK_ID}/storyboard.png
https://image.mux.com/{PLAYBACK_ID}/storyboard.webp

# WebVTT metadata (maps time ranges to sprite coordinates)
https://image.mux.com/{PLAYBACK_ID}/storyboard.vtt

# JSON metadata (alternative to VTT)
https://image.mux.com/{PLAYBACK_ID}/storyboard.json
```

**Formats**: jpg, png, webp

**Clipping support**: Storyboards respect `asset_start_time` and `asset_end_time` params, so clips get their own mini-storyboard.

**Mux Player integration**: Built-in. Just use `<mux-player>` and it auto-loads storyboards. For custom players, pass the VTT as a metadata text track.

**Football use case**: Coach scrubs through a 90-minute match and sees hover previews to quickly find the moment they want. Critical for usability.

**Implementation for Mux Player**:
```html
<mux-player
  playback-id="{PLAYBACK_ID}"
  storyboard-src="https://image.mux.com/{PLAYBACK_ID}/storyboard.vtt?format=webp"
></mux-player>
```

For clips:
```html
<mux-player
  playback-id="{PLAYBACK_ID}"
  extra-source-params="asset_start_time=45&asset_end_time=90"
  storyboard-src="https://image.mux.com/{PLAYBACK_ID}/storyboard.vtt?format=webp&asset_start_time=45&asset_end_time=90"
></mux-player>
```

---

### 1.3 Thumbnail Generation

**What it does**: Extract a still frame from any point in the video with transformations.

**URL format**:
```
https://image.mux.com/{PLAYBACK_ID}/thumbnail.{jpg|png|webp}?time={seconds}
```

**Parameters**:
| Param | Type | Description |
|-------|------|-------------|
| `time` | float | Timestamp in seconds (default: middle of video) |
| `width` | int | Pixel width (default: original) |
| `height` | int | Pixel height (default: original) |
| `fit_mode` | string | `preserve`, `stretch`, `crop`, `smartcrop`, `pad` |
| `flip_h` | bool | Mirror horizontally |
| `flip_v` | bool | Mirror vertically |
| `rotate` | int | 90, 180, or 270 degrees |

**Football use case**:
- Match card thumbnails showing a key moment
- Formation snapshot at kickoff (e.g., `?time=0&width=640&height=360`)
- AI references "defensive shape at 34:20" — show a thumbnail alongside the text
- `smartcrop` can auto-focus on the main action

**Current status**: Implemented as `get_thumbnail_at()` in `mux_service.py` (basic).

---

### 1.4 Animated GIFs / WebP

**What it does**: Generate animated previews from a video segment. No server-side processing needed.

**URL format**:
```
https://image.mux.com/{PLAYBACK_ID}/animated.{gif|webp}?start={s}&end={s}
```

**Parameters**:
| Param | Type | Description |
|-------|------|-------------|
| `start` | float | Start time in seconds (default: 0) |
| `end` | float | End time in seconds (default: start + 5) |
| `width` | int | Max 640px (default: 320) |
| `height` | int | Max 640px (aspect-ratio preserved) |
| `fps` | int | Frame rate 1-30 (default: 15) |

**Constraints**: Max 10 seconds duration, minimum 250ms.

**Football use case**:
- Highlight reel previews — each "moment" gets an animated GIF card
- Share on social media / WhatsApp group (coaches love sharing GIFs of goals)
- AI insight cards: "Great through-ball at 67:32" with a looping GIF preview
- Use WebP over GIF for better compression (smaller files, same quality)

**Current status**: Implemented as `get_gif()` in `mux_service.py`.

**Enhancement needed**: Add WebP support (currently only GIF). WebP is ~50% smaller.

---

### 1.5 Subtitles / Captions

**What it does**: Auto-generate captions from speech, or supply SRT/VTT files.

**Auto-generated captions**:
- Uses speech recognition + ML during asset processing
- Generates a single caption track based on spoken language
- Also produces full **transcripts** for further use

**Manual caption tracks**:
- Upload SRT or VTT files as additional tracks on an asset
- Multiple languages supported

**Football use case**:
- Auto-caption coach commentary if they narrate while filming
- Could overlay tactical commentary generated by AI as subtitle tracks
- Accessibility compliance

**API**: Add `"auto_generated_captions": true` to asset creation, or use the `generate_subtitles` endpoint.

**Player**: `default_subtitles_lang` param controls which language shows by default.

---

### 1.6 Chapters / Cue Points

**Chapters**: Visually split the timeline into named sections. Users see chapter titles and can click to jump.

**Cue Points**: Associate custom metadata (any JSON-serializable value) with time ranges. Defined with `startTime`, optional `endTime`, and a `value`.

**Football use case** (huge for us):
- Auto-generate chapters: "First Half", "Second Half", "Injury Break"
- AI-generated chapters: "Goal 1 (23:15)", "Counter-attack sequence (45:30)", "Set piece (67:00)"
- Cue points for every AI-detected event — press triggers, transitions, fouls
- Coach clicks chapter to jump straight to the tactical moment

**Implementation**: Mux Player supports chapters via WebVTT chapter tracks. AI generates the VTT, we pass it to the player:

```html
<mux-player playback-id="{ID}">
  <track kind="chapters" src="/api/matches/{id}/chapters.vtt" default />
</mux-player>
```

**AI chapter generation**: Mux has a blog/example on AI-generated chapters using their player. We can generate these from TwelveLabs analysis results.

---

### 1.7 Player Customization

**Mux Player** (`@mux/mux-player-react` — already in our frontend):

**Appearance**:
- `accent-color` / `accentColor` — brand color for controls (e.g., `"#ea580c"`)
- Full CSS parts API — style every element individually
- CSS variables for showing/hiding specific controls
- Themes support (primaryColor, secondaryColor, border)

**Controls & Features**:
| Feature | Attribute | Our Use |
|---------|-----------|---------|
| Picture-in-Picture | `pip` | Coach watches while reading analysis |
| Playback Rate | `playbackRate` + `playback-rate-control` | **Slow motion** for reviewing key moments |
| Chapters | `<track kind="chapters">` | AI-generated match timeline |
| Hotkeys | `hotkeys` | Custom keyboard shortcuts |
| Skip forward/back | `skip-forward-time`, `skip-back-time` | Quick 10s skip through footage |
| Fullscreen | `fullscreen` | Standard |
| Chromecast / AirPlay | `cast`, `airplay` | Cast to TV in changing room |
| Loop | `loop` | Loop a specific clip for review |
| Start time | `startTime` | Jump to AI-referenced moment |
| Poster | `poster` | Custom thumbnail |
| Storyboard | `storyboard-src` | Timeline hover previews |

**Events**: `play`, `pause`, `ended`, `seeked`, `timeupdate`, `error`, `ready`, `fullscreen-change`, `pip-change`

**Methods**: `play()`, `pause()`, `seekTo(time)`, `enterPip()`, `exitPip()`, `enterFullscreen()`

---

### 1.8 Mux Data (Viewer Analytics)

**What it does**: Automatic QoE (Quality of Experience) monitoring and viewer analytics.

**Setup**: Just pass `env-key` to Mux Player. Analytics are collected automatically.

**Metrics available**:
- Video startup time
- Rebuffering rate and duration
- Playback failure rate
- Viewer engagement (% watched, drop-off points)
- Concurrent viewers
- Geographic distribution
- Device/browser breakdown
- Custom dimensions via metadata (`video_id`, `video_title`, `viewer_user_id`)

**APIs**:
- `list_data_metrics` — list all available metrics
- `get_overall_values_data_metrics` — aggregate values
- `get_timeseries_data_metrics` — time-series data
- `get_breakdown_monitoring_data_metrics` — breakdown by dimension
- `list_data_video_views` — individual view records
- `retrieve_data_video_views` — single view details
- `list_data_errors` — playback errors
- `list_data_incidents` — detected incidents

**Real-time monitoring**:
- `list_metrics_data_real_time` — live metrics
- `retrieve_breakdown_data_real_time` — live breakdowns
- `retrieve_timeseries_data_real_time` — live time-series

**Annotations**: Create markers on your data timeline (e.g., "deployed v2.1", "match day") via `create_data_annotations`.

**Football use case**:
- Track which matches coaches actually watch
- See where they drop off (do they watch full match or skip to highlights?)
- Monitor video quality on rural 4G connections (grassroots coaches often review on phones)
- Identify most-replayed moments (engagement data = valuable insight)

---

### 1.9 Webhooks

**Key events for our pipeline**:

| Event | When | Our Action |
|-------|------|------------|
| `video.upload.asset_created` | Upload complete, asset created | Store asset_id, begin polling |
| `video.asset.created` | Asset record created | Log |
| `video.asset.ready` | Encoding done, playback ready | Trigger TwelveLabs indexing |
| `video.asset.errored` | Encoding failed | Notify coach, retry |
| `video.asset.static_renditions.ready` | MP4 downloads available | Update download links |
| `video.asset.track.ready` | Caption/subtitle track ready | Enable captions in UI |
| `video.live_stream.active` | Live stream started | N/A (future) |
| `video.live_stream.idle` | Live stream ended | N/A (future) |

**Current gap**: We are polling for asset status. Should switch to webhooks for reliability and efficiency.

---

### 1.10 Video Quality Levels (formerly "Encoding Tiers")

Renamed in 2025 from "encoding tiers" to "video quality levels":

| Level | Encoding Cost | Quality | Max Resolution | Live Support | Best For |
|-------|--------------|---------|----------------|--------------|----------|
| **Basic** | FREE | Reduced ladder, lower target quality | Up to 4K | No | Simple video, cost-sensitive |
| **Plus** | $0.0075/min (list) | AI per-title encoding, high quality | Up to 4K (1080p live) | Yes | Standard use |
| **Premium** | ~1.5x Plus | Same AI tech, tuned for highest quality | Up to 4K (1080p live) | Yes | Premium content |

**Current status**: We use `"baseline"` (now called Basic) — appropriate for grassroots phone footage.

**Recommendation**: Stay on Basic. Grassroots footage from phones does not benefit from premium encoding. The quality delta is minimal when source material is already 1080p phone video. Saves significant encoding cost.

---

### 1.11 Static Renditions (MP4 Downloads)

**What it does**: Creates downloadable MP4 or M4A files alongside HLS streaming.

**Options** (specified at asset creation or updated later):
- `"highest"` — MP4 up to 4K resolution
- `"audio-only"` — M4A audio extraction
- `"capped-1080p"` — MP4 capped at 1080p (via legacy `mp4_support`)
- Can request multiple: `["highest", "audio-only"]`

**Download URL format**:
```
https://stream.mux.com/{PLAYBACK_ID}/{filename}
```
Where filename comes from `static_renditions.files[].name`.

**Football use case**:
- Download clip MP4s for offline review
- Extract audio for voice-note analysis
- Provide MP4 URL to TwelveLabs for indexing (already using this)
- Share clips via WhatsApp/email as downloadable files

**Current status**: We check for static renditions in `get_asset()` and extract MP4 URL. Need to explicitly request them at creation time.

**Enhancement needed**: Add `"static_renditions": [{"resolution": "highest"}]` to asset creation params.

---

### 1.12 Signed URLs / Access Control

**Three playback policies**:
| Policy | URL Format | Use Case |
|--------|-----------|----------|
| `public` | `https://stream.mux.com/{PLAYBACK_ID}` | Open access |
| `signed` | `.../{PLAYBACK_ID}?token={JWT}` | Token-gated access |
| `drm` | DRM-protected | Premium content protection |

**JWT signing**:
- Create signing keys via API (`create_system_signing_keys`)
- Generate JWTs server-side with claims: `kid`, `exp`, `aud` (type: `v` for video, `t` for thumbnail, `s` for storyboard, `g` for GIF)
- Embed clip parameters (`asset_start_time`, `asset_end_time`) in JWT to lock down clip boundaries

**Playback restrictions**: Can restrict by referrer domain and user agent.

**Football use case**:
- Free tier: public playback IDs
- Paid tier: signed URLs so only authenticated coaches can view
- Clip sharing: generate time-limited signed JWT for specific clip window

**Current status**: Using `public` policy. Should upgrade to `signed` before launch for paid tiers.

---

### 1.13 Upload Limits

- **Max duration**: 12 hours per upload
- **Max file size**: No published hard limit (supports very large files via chunked upload)
- **Formats**: Wide range of video formats accepted
- **Free plan**: 10 on-demand assets, any duration
- **Direct upload**: Chunked upload via `mux-uploader` web component or API-generated upload URLs

**Football context**: A full 90-minute match at 1080p from a phone is typically 2-8 GB. Well within Mux's capabilities.

---

### 1.14 Playback Resolution Control

**HLS manifest parameters**:
- `max_resolution` — Cap maximum rendition (270p to 2160p)
- `min_resolution` — Set floor for minimum rendition
- `rendition_order` — `desc` to prefer higher quality first

**Football use case**: Cap at 720p for free tier users, allow 1080p for paid. Saves delivery bandwidth.

---

## 2. FOOTBALL ANALYSIS USE CASES — IMPLEMENTATION GUIDE

### 2.1 Showing Specific Moments When AI References a Timestamp

**Current flow**: AI says "pressing trigger at 23:15" as plain text.

**Enhanced flow**:
1. Parse timestamp from AI response
2. Generate instant clip URL: `stream.mux.com/{ID}.m3u8?asset_start_time=1390&asset_end_time=1400`
3. Generate thumbnail: `image.mux.com/{ID}/thumbnail.webp?time=1395&width=480`
4. Generate GIF preview: `image.mux.com/{ID}/animated.webp?start=1390&end=1400`
5. Render inline: thumbnail card that expands to clip player on click

**Implementation**:
```tsx
// When AI mentions a timestamp, render this component
<MomentCard
  playbackId={match.playback_id}
  startTime={1390}
  endTime={1400}
  label="Pressing trigger"
/>

// MomentCard shows GIF preview, click expands to:
<MuxPlayer
  playbackId={match.playback_id}
  extraSourceParams={`asset_start_time=${start}&asset_end_time=${end}`}
  storyboardSrc={`https://image.mux.com/${playbackId}/storyboard.vtt?format=webp&asset_start_time=${start}&asset_end_time=${end}`}
/>
```

### 2.2 Creating Highlight Reels (Clip + Concatenate)

**Mux does NOT natively concatenate clips into a single new asset.** Two approaches:

**Option A — Virtual Highlight Reel (Recommended)**:
- Store a list of `{start, end, label}` timestamps in the database
- Frontend plays clips sequentially using the Mux Player
- On clip end, auto-seek to next clip's start time
- Feels like a continuous reel, zero re-encoding cost

```tsx
const highlights = [
  { start: 45, end: 55, label: "Counter-attack" },
  { start: 120, end: 135, label: "Goal" },
  { start: 300, end: 310, label: "Great save" },
];
// Play each as instant clip, chain via onEnded event
```

**Option B — Server-Side Concatenation** (if downloadable reel needed):
- Download MP4 static renditions for each clip time range
- Use FFmpeg server-side to concatenate
- Upload result as new Mux asset
- More expensive (encoding + storage), only for "export highlight reel" feature

**Recommendation**: Option A for in-app viewing (free), Option B only for shareable export.

### 2.3 Side-by-Side Comparison of Different Matches

**Mux supports multiple simultaneous players**. No special API needed.

```tsx
<div className="grid grid-cols-2 gap-4">
  <MuxPlayer
    playbackId={match1.playback_id}
    extraSourceParams={`asset_start_time=${clip1.start}&asset_end_time=${clip1.end}`}
    accentColor="#3b82f6"
  />
  <MuxPlayer
    playbackId={match2.playback_id}
    extraSourceParams={`asset_start_time=${clip2.start}&asset_end_time=${clip2.end}`}
    accentColor="#ef4444"
  />
</div>
```

**Sync playback**: Use `timeupdate` event on primary player to `seekTo()` on secondary player. Both players share the same playback position.

**Football use case**: "Compare how we defended corners in Match 3 vs Match 7"

### 2.4 Slow-Motion Playback of Key Moments

**Mux Player supports `playbackRate`** — set to values below 1.0 for slow motion.

```tsx
<MuxPlayer
  playbackId={match.playback_id}
  playbackRate={0.25}  // Quarter speed
  extraSourceParams={`asset_start_time=${start}&asset_end_time=${end}`}
/>
```

**Available rates**: Any float. Common: 0.25x, 0.5x, 0.75x, 1x, 1.5x, 2x

**Enable user control**: `playback-rate-control={true}` shows a speed selector in the player UI.

**Football use case**: Slow-mo review of tackles, offside decisions, technique analysis.

### 2.5 Picture-in-Picture for Tactical Overlay

**Native PiP support** in Mux Player:
```tsx
<MuxPlayer
  playbackId={match.playback_id}
  pip={true}  // Enable PiP button
/>
```

**Tactical overlay approach**:
1. Coach opens a clip in PiP mode (video floats in corner)
2. Main screen shows tactical board / formation diagram
3. Coach can draw arrows/annotations on the tactical board while watching the clip

**Alternative — Canvas overlay**:
- Position a `<canvas>` element over the `<mux-player>`
- Draw tactical lines, arrows, player markers on the canvas
- Video plays underneath with pointer-events passing through to the player

---

## 3. PRICING ANALYSIS FOR GRASSROOTS

### 3.1 Current Pricing (2026, post price drops)

| Component | Rate | Notes |
|-----------|------|-------|
| **Encoding (Basic)** | FREE | No charge for basic quality |
| **Encoding (Plus)** | ~$0.006/min | After 22% reduction |
| **Storage (720p)** | $0.0012/min | Per minute stored |
| **Storage (1080p)** | $0.0015/min | Per minute stored |
| **Delivery** | $0.00096/min | First 100K min/month FREE |
| **Mux Data** | Included with Player | Free tier available |

### 3.2 Cost Model: Single Grassroots Team (1 match/week)

**Assumptions**:
- 1 match per week, 90 minutes each
- 1080p phone footage
- Basic encoding (free)
- 10 views per match (coach + a few players/parents)
- Retain last 20 matches (~6 months)

| Line Item | Calculation | Monthly Cost |
|-----------|-------------|-------------|
| Encoding | 4 matches x 90 min x $0.00 | $0.00 |
| Storage | 20 matches x 90 min x $0.0015 | $2.70 |
| Delivery | 4 matches x 90 min x 10 views x $0.00096 | $3.46 |
| **Total** | | **~$6.16/month** |

With the free 100K delivery minutes: 4 x 90 x 10 = 3,600 minutes, well under 100K. **Delivery is effectively free.**

**Revised total with free delivery: ~$2.70/month per team.**

### 3.3 Cost Model: 100 Teams (Growth Target)

| Line Item | Calculation | Monthly Cost |
|-----------|-------------|-------------|
| Encoding | Free (Basic tier) | $0.00 |
| Storage | 100 teams x 20 matches x 90 min x $0.0015 | $270 |
| Delivery | 100 x 4 x 90 x 10 x $0.00096 = 360K min (260K billable) | $249.60 |
| **Total** | | **~$520/month** |

At $9.99/team/month, revenue = $999/month. **Healthy 48% margin on video costs alone.**

### 3.4 Cost Optimization Strategies

1. **Use Basic encoding** — free, good enough for phone footage
2. **Cold storage** — Mux auto-discounts assets not being watched (up to 60% off storage)
3. **Cap resolution** at 720p for free tier — reduces storage and delivery costs
4. **Instant clipping over new assets** — clips cost nothing, no re-encoding
5. **Use WebP over GIF** — smaller files, less bandwidth
6. **Storyboard caching** — storyboards are CDN-cached, minimal cost
7. **Delete old matches** — auto-archive after season ends to reduce storage

---

## 4. GAPS IN CURRENT IMPLEMENTATION

| # | Gap | Priority | Effort |
|---|-----|----------|--------|
| 1 | No webhook integration (polling instead) | HIGH | Medium |
| 2 | No storyboard/hover preview in player | HIGH | Low |
| 3 | No chapters/cue points from AI analysis | HIGH | Medium |
| 4 | No animated WebP support (only GIF) | LOW | Trivial |
| 5 | No Mux Data analytics tracking | MEDIUM | Low |
| 6 | No signed URLs for paid tier | MEDIUM | Medium |
| 7 | No static renditions requested at creation | MEDIUM | Trivial |
| 8 | No playback rate control in UI | MEDIUM | Trivial |
| 9 | No PiP button enabled | LOW | Trivial |
| 10 | Using `encoding_tier: "baseline"` (old name) — should be `video_quality: "basic"` | LOW | Trivial |
| 11 | No resolution capping per user tier | LOW | Low |
| 12 | No auto-generated captions | LOW | Low |

---

## 5. RECOMMENDED IMPLEMENTATION ORDER

### Phase 1 — Quick Wins (1-2 days)
- [ ] Enable storyboard hover previews on MuxPlayer component
- [ ] Add `playback-rate-control` and `pip` to player
- [ ] Request static renditions at asset creation time
- [ ] Add WebP animated preview support to `mux_service.py`
- [ ] Update `encoding_tier: "baseline"` to `video_quality: "basic"`

### Phase 2 — AI-Powered Timeline (3-5 days)
- [ ] Generate chapter VTT files from TwelveLabs analysis results
- [ ] Pass chapters to Mux Player as `<track kind="chapters">`
- [ ] Build `MomentCard` component (thumbnail + GIF preview + click-to-play)
- [ ] Parse AI timestamps and auto-link to clip player

### Phase 3 — Infrastructure (3-5 days)
- [ ] Implement webhook endpoint for `video.asset.ready` and `video.asset.errored`
- [ ] Switch from polling to webhook-driven pipeline
- [ ] Add Mux Data `env-key` to player for analytics
- [ ] Build basic analytics dashboard (most-watched moments, engagement)

### Phase 4 — Monetization & Security (2-3 days)
- [ ] Implement signed playback URLs for paid tier
- [ ] Resolution capping (720p free, 1080p paid)
- [ ] Referrer restrictions for embed protection
- [ ] Auto-caption generation for accessibility

### Phase 5 — Advanced Features (future)
- [ ] Virtual highlight reel player (chained clips)
- [ ] Side-by-side match comparison view
- [ ] Tactical overlay canvas on top of Mux Player
- [ ] Export highlight reel as downloadable MP4 (FFmpeg concatenation)
- [ ] Live stream support for match day (Plus tier required)

---

## 6. MUX MCP TOOLS AVAILABLE

We have direct API access to Mux via the MCP server. Key tools:

**Asset Management**: `create_video_assets`, `retrieve_video_assets`, `update_video_assets`, `list_video_assets`
**Uploads**: `create_video_uploads`, `retrieve_video_uploads`, `cancel_video_uploads`
**Playback**: `create_playback_id_video_assets`, `hls_video_playback`, `thumbnail_video_playback`, `animated_video_playback`, `storyboard_video_playback`, `storyboard_meta_video_playback`, `storyboard_vtt_video_playback`, `static_rendition_video_playback`, `transcript_video_playback`, `track_video_playback`
**Static Renditions**: `create_static_rendition_video_assets`, `delete_static_rendition_video_assets`
**Tracks**: `create_track_video_assets`, `delete_track_video_assets`
**Subtitles**: `generate_subtitles_video_assets`
**Live Streams**: Full CRUD + simulcast targets
**Data/Analytics**: `list_data_metrics`, `get_overall_values_data_metrics`, `get_timeseries_data_metrics`, `list_data_video_views`, `retrieve_data_video_views`, `list_data_errors`
**Real-Time Monitoring**: `list_metrics_data_real_time`, `retrieve_breakdown_data_real_time`, `retrieve_timeseries_data_real_time`
**Signing Keys**: `create_system_signing_keys`, `list_system_signing_keys`
**Annotations**: `create_data_annotations`, `list_data_annotations`, `update_data_annotations`
**Restrictions**: `create_video_playback_restrictions`, `update_referrer_video_playback_restrictions`
**Transcription**: `create_video_transcription_vocabularies` (custom vocab for football terms)

---

## 7. KEY TECHNICAL NOTES

- **No server-side clip concatenation** in Mux — handle in frontend (virtual reel) or FFmpeg
- **Storyboards are auto-generated** — no action needed, just reference the URL
- **Thumbnails are CDN-cached** — extremely fast and cheap
- **Animated previews max 10 seconds** — sufficient for football moments
- **Encoding tier rename**: `baseline` still works but `basic` is the new name; `smart` is now `plus`; new tier `premium` exists
- **Custom transcription vocabulary**: Can add football-specific terms (e.g., "gegenpressing", "inverted wingback") for better auto-captions
- **Cold storage kicks in automatically** for unwatched assets — no config needed

---

## Sources

- [Mux Instant Clipping](https://www.mux.com/docs/guides/create-instant-clips)
- [Create Timeline Hover Previews](https://www.mux.com/docs/guides/create-timeline-hover-previews)
- [Get Images from Video](https://www.mux.com/docs/guides/get-images-from-a-video)
- [Enable Static MP4 Renditions](https://www.mux.com/docs/guides/enable-static-mp4-renditions)
- [Video Quality Levels](https://www.mux.com/docs/guides/use-video-quality-levels)
- [Secure Video Playback](https://www.mux.com/docs/guides/secure-video-playback)
- [Modify Playback Behavior](https://www.mux.com/docs/guides/modify-playback-behavior)
- [Listen for Webhooks](https://www.mux.com/docs/core/listen-for-webhooks)
- [Webhook Reference](https://www.mux.com/docs/webhook-reference)
- [Mux Player Advanced Usage](https://www.mux.com/docs/guides/player-advanced-usage)
- [Player Customize Look and Feel](https://www.mux.com/docs/guides/player-customize-look-and-feel)
- [AI-Generated Chapters](https://www.mux.com/blog/ai-generated-chapters-for-your-videos-with-mux-player)
- [Add Subtitles to Videos](https://www.mux.com/docs/guides/add-subtitles-to-your-videos)
- [Mux Video Pricing](https://www.mux.com/docs/pricing/video)
- [Mux Pricing Calculator](https://www.mux.com/pricing/calculator)
- [Estimating Video Costs](https://www.mux.com/docs/pricing/estimating-video-costs)
- [Advanced Static Renditions](https://www.mux.com/blog/advanced-static-renditions)
- [Control Playback Resolution](https://www.mux.com/docs/guides/control-playback-resolution)
- [Mux Player API Reference](https://www.mux.com/docs/guides/player-api-reference/html)
- [Mux Data (Video Performance Analytics)](https://data.mux.com)
- [Download Your Videos](https://www.mux.com/docs/guides/download-your-videos)
- [Video Streaming Pricing Comparison (2026)](https://www.buildmvpfast.com/api-costs/video)
