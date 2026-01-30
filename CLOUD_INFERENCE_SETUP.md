# Cloud GPU Inference Setup Guide

This guide explains how to set up cloud GPU inference for real-time match analysis.

## Why Cloud GPU?

Your Intel i7-8650U with integrated graphics cannot run deep learning inference fast enough for real-time analysis:
- **Local CPU**: ~0.5-1 FPS (too slow for live coaching)
- **Cloud GPU**: 15-30 FPS (smooth real-time analysis)

## Recommended Services

### Option 1: RunPod (Recommended for beginners)

**Cost**: ~$0.20-0.50/hour for inference workloads

1. **Create Account**
   - Sign up at [runpod.io](https://runpod.io)
   - Add credits ($10 minimum)

2. **Deploy Serverless Endpoint**
   ```bash
   # Use the pre-built YOLO inference template
   # Or deploy custom container (see below)
   ```

3. **Configure in App**
   ```bash
   # Set environment variables
   CLOUD_INFERENCE_ENABLED=true
   CLOUD_INFERENCE_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run
   CLOUD_API_KEY=your_runpod_api_key
   ```

### Option 2: Lambda Labs

**Cost**: ~$0.50/hour for RTX A4000

1. **Create Account**
   - Sign up at [lambdalabs.com](https://lambdalabs.com)

2. **Launch Instance**
   - Select GPU instance (RTX A4000 recommended)
   - Use PyTorch image

3. **Deploy Inference Server**
   ```bash
   # SSH into instance
   ssh ubuntu@your-instance-ip

   # Clone and run inference server
   git clone https://github.com/your-repo/football-analyzer
   cd football-analyzer/inference-server
   pip install -r requirements.txt
   python server.py --port 8001
   ```

### Option 3: AWS EC2 (g4dn instances)

**Cost**: ~$0.50-1.00/hour for g4dn.xlarge

1. **Launch Instance**
   - AMI: Deep Learning AMI (Ubuntu)
   - Instance: g4dn.xlarge (1x T4 GPU)

2. **Configure Security Group**
   - Allow inbound on port 8001 (inference API)

3. **Deploy Server**
   ```bash
   # Install dependencies
   pip install ultralytics fastapi uvicorn

   # Run inference server
   python inference_server.py
   ```

---

## Inference Server Code

Create this file on your cloud GPU server:

```python
# inference_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

# Load model once at startup
model = YOLO("yolov8s.pt")
model.to("cuda")  # Use GPU

class InferenceRequest(BaseModel):
    image: str  # Base64 encoded

class Detection(BaseModel):
    bbox: dict
    position: dict
    confidence: float
    class_id: int

class InferenceResponse(BaseModel):
    detections: list[Detection]
    processing_time_ms: float

@app.post("/detect", response_model=InferenceResponse)
async def detect(request: InferenceRequest):
    import time
    start = time.time()

    # Decode image
    img_data = base64.b64decode(request.image)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    results = model(frame, conf=0.5, classes=[0, 32])  # Person + sports ball

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(Detection(
                bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(box.conf[0])},
                position={"x": (x1+x2)//2, "y": (y1+y2)//2},
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0])
            ))

    processing_time = (time.time() - start) * 1000

    return InferenceResponse(
        detections=detections,
        processing_time_ms=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

## Docker Deployment (Optional)

For easier deployment, use this Dockerfile:

```dockerfile
FROM ultralytics/ultralytics:latest-python

WORKDIR /app

COPY inference_server.py .

EXPOSE 8001

CMD ["python", "inference_server.py"]
```

Build and deploy:
```bash
docker build -t football-inference .
docker run --gpus all -p 8001:8001 football-inference
```

---

## Cost Optimization Tips

1. **Use Spot/Preemptible Instances**
   - AWS Spot: 60-70% cheaper
   - RunPod Spot: 50% cheaper

2. **Auto-shutdown**
   - Stop instances when not in use
   - Use scheduled scaling

3. **Batch Processing**
   - Send multiple frames per request
   - Reduces API overhead

4. **Model Optimization**
   - Use YOLOv8n (nano) for faster inference
   - Enable TensorRT for NVIDIA GPUs

---

## Testing Your Setup

1. **Test Connection**
   ```bash
   curl -X POST http://your-cloud-server:8001/detect \
     -H "Content-Type: application/json" \
     -d '{"image": "BASE64_ENCODED_TEST_IMAGE"}'
   ```

2. **Measure Latency**
   ```python
   import httpx
   import time

   start = time.time()
   response = httpx.post(
       "http://your-server:8001/detect",
       json={"image": test_image_b64}
   )
   latency = (time.time() - start) * 1000
   print(f"Round-trip latency: {latency:.0f}ms")
   ```

   Target: < 300ms round-trip for smooth coaching

---

## Hybrid Mode (Recommended)

For cost efficiency, use this approach:

1. **Live Matches**: Cloud GPU (~$1-2 per 90min match)
2. **Post-Match Analysis**: Local CPU overnight (free)

Configure in `.env`:
```bash
# For live matches
CLOUD_INFERENCE_ENABLED=true

# For post-match (set before starting)
CLOUD_INFERENCE_ENABLED=false
```

---

## Troubleshooting

### High Latency (>500ms)
- Check network connection
- Use server in same region
- Enable connection pooling

### Out of Memory
- Reduce batch size
- Use smaller model (yolov8n)
- Check for memory leaks

### Connection Refused
- Check firewall rules
- Verify server is running
- Check port configuration
