# Football Analyzer - Real-Time Game Management Specification

## Overview
This document specifies the real-time game management system for live match analysis using VEO camera streams. The system provides coaches with immediate tactical feedback and alerts during matches.

---

## 1. VEO Camera Integration

### 1.1 Stream Input Methods

| Method | Protocol | Latency | Use Case |
|--------|----------|---------|----------|
| **RTSP Stream** | rtsp:// | 1-3 sec | Direct camera feed |
| **HLS Stream** | http(s):// m3u8 | 5-10 sec | Cloud-based viewing |
| **WebRTC** | webrtc:// | <1 sec | Ultra-low latency |

### 1.2 VEO Camera Compatibility

VEO cameras support live streaming via:
- **VEO Live**: HLS stream accessible via VEO's API
- **Direct RTSP**: Available on some VEO models for local network
- **VEO Share Links**: Can extract HLS URL for processing

### 1.3 Stream Configuration

```python
@dataclass
class LiveStreamConfig:
    stream_url: str          # RTSP/HLS URL
    stream_type: str         # "rtsp", "hls", "webrtc"
    target_fps: float = 2.0  # Processing FPS (not display FPS)
    buffer_seconds: float = 2.0  # Buffer for network jitter
    reconnect_attempts: int = 5
    reconnect_delay_ms: int = 2000
```

---

## 2. Real-Time Processing Pipeline

### 2.1 Processing Flow

```
VEO Stream → Frame Buffer → Detection (YOLO) → Tracking → Event Detection → Alert Generation
                                    ↓                              ↓
                              Team Classification            Stats Update
                                    ↓                              ↓
                              Ball Detection  ←――――――――→  WebSocket Push
```

### 2.2 Latency Budget (Target: <5 seconds end-to-end)

| Stage | Target Latency |
|-------|---------------|
| Stream acquisition | 1-2 sec |
| Frame decode | 50ms |
| YOLO detection | 100-200ms |
| Tracking + Events | 50ms |
| Alert generation | 20ms |
| WebSocket push | 10ms |
| **Total** | **~2-3 seconds** |

### 2.3 Processing Modes

| Mode | FPS | Detections | Events | Alerts |
|------|-----|------------|--------|--------|
| **Full Analysis** | 2-3 | All players | All events | All |
| **Tactical Only** | 1 | Key players | Possession changes | Pressing/Shape |
| **Alerts Only** | 0.5 | Minimal | Critical only | High priority |

---

## 3. Real-Time Alerts

### 3.1 Alert Categories

#### Immediate Alerts (Push within 2 seconds)
| Alert | Trigger | Action Suggested |
|-------|---------|------------------|
| **Pressing Opportunity** | 2+ defenders near ball carrier | "PRESS NOW - #7 isolated" |
| **Counter-Attack** | Turnover in defensive third | "COUNTER - Wide right" |
| **Defensive Gap** | >15m gap in defensive line | "COVER - Gap between #4 and #5" |
| **Offside Trap Broken** | Attacker behind line | "DROP BACK - Offside broken" |

#### Tactical Alerts (Every 30-60 seconds)
| Alert | Trigger | Information |
|-------|---------|-------------|
| **Formation Shift** | Detected formation change | "Opponent changed to 4-4-2" |
| **Player Fatigue** | Speed drop >20% | "#8 showing fatigue signs" |
| **Pressing Intensity** | PPDA change | "Opponent pressing higher" |
| **Width Analysis** | Team shape change | "Stretch play - they're narrow" |

#### Strategic Alerts (Half-time / Every 10 min)
| Alert | Information |
|-------|-------------|
| **Possession Summary** | "62% possession, mostly in middle third" |
| **Danger Zone** | "Vulnerable to left-wing crosses" |
| **Key Matchup** | "#10 winning duels vs #6" |

### 3.2 Alert Data Structure

```python
@dataclass
class LiveAlert:
    alert_id: str
    priority: str  # "immediate", "tactical", "strategic"
    category: str  # "pressing", "defensive", "attacking", "fatigue"
    timestamp_ms: int

    # Content
    title: str           # "PRESSING OPPORTUNITY"
    message: str         # "Press #7 - isolated on right wing"
    action: str          # "Trigger high press"

    # Visual
    highlight_players: List[int]  # Track IDs to highlight
    highlight_zone: Optional[str]  # "left_wing", "center", etc.

    # Timing
    duration_seconds: int = 10  # How long alert stays visible
    expires_at_ms: int         # Auto-dismiss time

    # Audio
    play_sound: bool = True
    sound_type: str = "alert"  # "alert", "warning", "info"
```

---

## 4. Live Dashboard Features

### 4.1 Main Display Areas

```
┌─────────────────────────────────────────────────────────────────┐
│                        LIVE FEED                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │              [Video with player tracking overlay]        │   │
│  │                                                          │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ ALERT BOX    │                                       │   │
│  │  │ PRESS NOW!   │                                       │   │
│  │  └──────────────┘                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  POSSESSION  │  │    SCORE     │  │   LIVE EVENTS        │  │
│  │  HOME  AWAY  │  │   2 - 1      │  │   12' Pass - #7→#10  │  │
│  │  58%   42%   │  │              │  │   14' Shot - #9      │  │
│  │  ▓▓▓▓▒▒▒     │  │  45:23       │  │   15' Corner - HOME  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    TACTICAL MINI-MAP                        │ │
│  │    [2D pitch view with real-time player positions]         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Dashboard Components

| Component | Update Frequency | Data Source |
|-----------|-----------------|-------------|
| Video feed | Real-time | Stream |
| Player overlay | 2-3 FPS | Detection |
| Alert box | On trigger | Alert system |
| Possession bar | 5 seconds | Stats |
| Score/Time | 1 second | Manual/detected |
| Event feed | On event | Event detection |
| Mini-map | 1 second | Tracking |

### 4.3 Coach Interaction

| Action | Function |
|--------|----------|
| **Tap alert** | Dismiss / mark as addressed |
| **Tap player** | Quick stats popup |
| **Swipe** | Switch between views |
| **Voice command** | "Show pressing stats" |
| **Quick notes** | Record observation with timestamp |

---

## 5. WebSocket Protocol

### 5.1 Connection

```
ws://server:8000/ws/live-coaching/{match_id}
```

### 5.2 Message Types

#### Server → Client

```json
// Frame update (2-3 FPS)
{
    "type": "frame_update",
    "timestamp_ms": 2700000,
    "players": [...],
    "ball": {...},
    "possession": "home"
}

// Alert
{
    "type": "alert",
    "alert": {
        "alert_id": "abc123",
        "priority": "immediate",
        "title": "PRESSING OPPORTUNITY",
        "message": "Press #7 now",
        "highlight_players": [7],
        "play_sound": true
    }
}

// Stats update (every 5 sec)
{
    "type": "stats_update",
    "stats": {
        "home_possession": 58.2,
        "away_possession": 41.8,
        "home_passes": 124,
        "away_passes": 87
    }
}

// Event
{
    "type": "event",
    "event": {
        "type": "shot",
        "player_id": 9,
        "team": "home",
        "on_target": true
    }
}
```

#### Client → Server

```json
// Dismiss alert
{
    "type": "dismiss_alert",
    "alert_id": "abc123"
}

// Add note
{
    "type": "add_note",
    "timestamp_ms": 2700000,
    "note": "Good pressing from #8"
}

// Request stats
{
    "type": "request_stats",
    "stat_type": "passing"
}

// Update score (manual)
{
    "type": "update_score",
    "home": 2,
    "away": 1
}
```

---

## 6. Implementation Phases

### Phase 1: Stream Integration (Week 1)
1. RTSP/HLS stream reader with reconnection
2. Frame buffer for consistent processing
3. Basic WebSocket live feed
4. Simple detection overlay

### Phase 2: Real-Time Events (Week 2)
1. Low-latency event detection
2. Possession tracking
3. Basic alerts (pressing opportunities)
4. Stats accumulation

### Phase 3: Alert System (Week 3)
1. Full alert categories
2. Alert prioritization
3. Sound notifications
4. Alert history

### Phase 4: Dashboard (Week 4)
1. Live video with overlay
2. Stats panels
3. Mini-map
4. Event feed
5. Coach interaction

### Phase 5: Polish & Optimization
1. Latency optimization
2. Mobile responsiveness
3. Offline fallback
4. Performance tuning

---

## 7. Technical Requirements

### 7.1 Server Requirements

| Resource | Requirement |
|----------|-------------|
| CPU | 4+ cores (8 recommended) |
| RAM | 8GB minimum |
| GPU | Optional (CUDA for faster YOLO) |
| Network | 10Mbps+ stable connection |

### 7.2 Client Requirements

| Platform | Requirement |
|----------|-------------|
| Browser | Chrome 90+, Safari 14+, Firefox 88+ |
| Mobile | iOS 14+, Android 10+ |
| Network | 5Mbps+ for video stream |

### 7.3 Dependencies

```python
# Stream handling
opencv-python>=4.5.0  # RTSP/video processing
av>=9.0.0             # HLS decoding
aiortc>=1.3.0         # WebRTC (optional)

# Real-time
websockets>=10.0
asyncio
aiohttp

# Existing
ultralytics           # YOLO
numpy
scikit-learn
```

---

## 8. API Endpoints

### 8.1 Stream Management

```
POST /api/live/start
Body: { "stream_url": "rtsp://...", "stream_type": "rtsp" }
Response: { "session_id": "...", "websocket_url": "ws://..." }

POST /api/live/stop
Body: { "session_id": "..." }

GET /api/live/status
Response: { "active": true, "fps": 2.1, "latency_ms": 2300 }
```

### 8.2 Configuration

```
POST /api/live/config
Body: {
    "alert_settings": {
        "pressing_threshold": 2,
        "sound_enabled": true,
        "immediate_alerts": true,
        "tactical_alerts": true
    },
    "display_settings": {
        "show_tracking": true,
        "show_minimap": true,
        "overlay_opacity": 0.7
    }
}
```

### 8.3 Manual Input

```
POST /api/live/score
Body: { "home": 2, "away": 1 }

POST /api/live/substitution
Body: { "team": "home", "player_out": 7, "player_in": 14 }

POST /api/live/note
Body: { "note": "Good pressing", "timestamp_ms": 2700000 }
```

---

## 9. Error Handling

### 9.1 Stream Issues

| Issue | Detection | Recovery |
|-------|-----------|----------|
| Stream disconnect | No frames for 3 sec | Auto-reconnect (5 attempts) |
| High latency | Latency > 10 sec | Switch to alerts-only mode |
| Frame corruption | Decode error | Skip frame, continue |

### 9.2 Processing Issues

| Issue | Detection | Fallback |
|-------|-----------|----------|
| Detection failure | No players detected | Use last known positions |
| Tracking loss | ID jumps | Reset tracker, re-identify |
| Alert overload | >10 alerts/min | Throttle to high priority |

---

## 10. Future Enhancements

1. **Multi-camera support**: Merge views from multiple VEO cameras
2. **AI Commentary**: Generate spoken tactical commentary
3. **Opponent learning**: Build opponent profiles during match
4. **Substitution advisor**: Suggest optimal subs based on fatigue/matchups
5. **Set piece detection**: Alert when opponent is setting up set piece
6. **Weather integration**: Adjust analysis for conditions
7. **Referee positioning**: Track referee for offside analysis
