# Football Match Analyzer

AI-powered football match analysis and real-time coaching system using computer vision.

## Features

- **Player Detection & Tracking**: YOLOv8 + ByteTrack for consistent player identification
- **Team Classification**: Automatic jersey color detection and team assignment
- **Ball Tracking**: Multi-method ball detection (YOLO + motion + color)
- **Event Detection**: Passes, shots, tackles, interceptions
- **Real-time Analytics**: Distance, sprints, possession, pass completion
- **AI Coaching Alerts**: Pressing opportunities, formation drift, space exploitation
- **Post-match Reports**: xG, heatmaps, pass networks, player ratings

## System Requirements

### Hardware
- **CPU**: Intel i7 or equivalent (for post-match processing)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: Optional for local inference (NVIDIA with CUDA)
- **Storage**: 50GB+ for video files

### For Real-time Analysis
- Cloud GPU service (RunPod, Lambda Labs, AWS) - see [CLOUD_INFERENCE_SETUP.md](CLOUD_INFERENCE_SETUP.md)

### Software
- Python 3.10+
- Node.js 18+
- FFmpeg (for video processing)

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### 3. Access Dashboard

Open http://localhost:3000 in your browser.

## Project Structure

```
football-analyzer/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── services/
│   │   ├── video_ingestion.py  # Video/RTMP handling
│   │   ├── detection.py        # YOLO player detection
│   │   ├── tracking.py         # ByteTrack integration
│   │   ├── pitch_mapping.py    # Homography transforms
│   │   ├── ball_detection.py   # Ball tracking
│   │   ├── event_detection.py  # Event classification
│   │   └── analytics.py        # Metrics calculation
│   ├── ai/
│   │   ├── tactical_analyzer.py # Pattern recognition
│   │   ├── recommendation.py    # Coaching suggestions
│   │   └── alerts.py           # Alert management
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── utils/
│   │   └── geometry.py         # Math utilities
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main application
│   │   ├── components/
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── TacticalBoard.tsx
│   │   │   ├── StatsPanel.tsx
│   │   │   ├── AlertFeed.tsx
│   │   │   ├── VideoUpload.tsx
│   │   │   └── MatchSetup.tsx
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts
│   │   └── types.ts
│   ├── package.json
│   └── tailwind.config.js
├── data/
│   ├── uploads/                # Uploaded videos
│   ├── frames/                 # Extracted frames
│   └── models/                 # ML models
├── CLOUD_INFERENCE_SETUP.md
└── README.md
```

## Configuration

Create a `.env` file in the backend directory:

```env
# Cloud Inference (for live mode)
CLOUD_INFERENCE_ENABLED=false
CLOUD_INFERENCE_URL=https://your-cloud-endpoint
CLOUD_API_KEY=your_api_key

# Processing
LIVE_FPS=10
ANALYSIS_FPS=25
DETECTION_CONFIDENCE=0.5

# Pitch dimensions (standard)
PITCH_LENGTH=105
PITCH_WIDTH=68
```

## Usage

### Post-Match Analysis (CPU)

1. Upload your match video (MP4, AVI, MOV, MKV)
2. Set up team names and jersey colors
3. Select "Post-match" processing mode
4. Analysis runs overnight on CPU
5. View results in dashboard next day

### Live Analysis (Cloud GPU)

1. Set up cloud GPU endpoint (see [CLOUD_INFERENCE_SETUP.md](CLOUD_INFERENCE_SETUP.md))
2. Configure CLOUD_INFERENCE_URL in .env
3. Connect VEO Cam 3 RTMP stream
4. Select "Live" processing mode
5. Real-time coaching alerts appear in dashboard

## API Endpoints

### Video
- `POST /api/video/upload` - Upload video file
- `POST /api/video/{id}/process` - Start processing
- `GET /api/video/{id}/status` - Get processing status

### Match
- `POST /api/match/create` - Create new match
- `GET /api/match/{id}` - Get match state
- `POST /api/match/{id}/calibrate` - Calibrate pitch

### Analytics
- `GET /api/match/{id}/analytics` - Get analytics
- `GET /api/match/{id}/heatmap/{player_id}` - Player heatmap
- `GET /api/match/{id}/pass-network` - Pass network

### WebSocket
- `ws://localhost:8000/ws/match/{id}` - Real-time updates

## Coaching Alerts

The system generates three types of alerts:

1. **Immediate** (< 5 seconds)
   - Pressing opportunities
   - Counter-attack risks
   - Dangerous situations

2. **Tactical** (30-60 seconds)
   - Formation drift warnings
   - Space to exploit
   - Substitution suggestions

3. **Strategic** (half-time)
   - Pattern analysis
   - Tactical adjustments
   - Performance insights

## Technology Stack

### Backend
- FastAPI (REST API + WebSocket)
- PyTorch + Ultralytics (YOLOv8)
- OpenCV (video processing)
- NumPy/Pandas (analytics)
- SQLite (data storage)

### Frontend
- React 18 + TypeScript
- Tailwind CSS
- Recharts (visualizations)
- Lucide React (icons)

## VEO Cam 3 Integration

The system supports RTMP streaming from VEO Cam 3:

1. Configure VEO to broadcast RTMP
2. Use the stream URL in the app
3. Real-time frames are sent to cloud GPU
4. Results displayed with ~200-300ms latency

## Limitations

- Ball detection accuracy depends on video quality
- Team classification requires visible jersey colors
- Real-time mode requires cloud GPU (costs ~$1-2/match)
- Pitch calibration needed for each camera angle

## Future Improvements

- [ ] Fine-tuned ball detection model
- [ ] Player re-identification across camera cuts
- [ ] Advanced xG model
- [ ] Tactical pattern library
- [ ] Mobile app for touchline use
- [ ] Multi-camera support

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [SoccerNet](https://www.soccer-net.org/) for football AI research
- VEO Technologies for camera hardware
