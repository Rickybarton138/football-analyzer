"""
Off-Screen Player Position Prediction for VEO Camera Footage.

VEO cameras track the ball, so only ~60% of the pitch is visible at any time.
This module predicts positions of off-screen players to maintain a full 22-player
model using Kalman filtering, formation templates, and role-zone constraints.

Pipeline: YOLO -> ByteTrack -> OffScreenPredictor -> formation_detector / tactical_intelligence
Input: N detected players (~12-16) per frame
Output: 22 players (11+11) with confidence scores + visibility state
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from copy import deepcopy

from scipy.optimize import linear_sum_assignment
from services.predictive_tracker import KalmanFilter2D
from config import settings


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class VisibilityState(Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"
    FORMATION_HOLD = "formation_hold"


@dataclass
class PlayerPrediction:
    slot_id: int                    # 1-11 within team
    team: str                       # 'home' or 'away'
    jersey_number: Optional[int] = None
    x: float = 50.0                 # 0-100 normalized pitch coords
    y: float = 50.0
    state: VisibilityState = VisibilityState.PREDICTED
    confidence: float = 0.5
    last_detected_frame: int = 0
    frames_since_detected: int = 0
    role: str = "CM"
    track_id: Optional[int] = None


@dataclass
class CameraFrustum:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0


# ---------------------------------------------------------------------------
# Formation Templates — 11 positions each, (x, y) in 0-100 coords
# Home team attacks RIGHT (higher x). Away mirrors automatically.
# ---------------------------------------------------------------------------

FORMATION_TEMPLATES: Dict[str, List[Tuple[str, float, float]]] = {
    "4-4-2": [
        ("GK",  5, 50),
        ("RB", 25, 80), ("CB", 25, 60), ("CB", 25, 40), ("LB", 25, 20),
        ("RM", 50, 85), ("CM", 45, 60), ("CM", 45, 40), ("LM", 50, 15),
        ("ST", 75, 60), ("ST", 75, 40),
    ],
    "4-3-3": [
        ("GK",  5, 50),
        ("RB", 25, 80), ("CB", 25, 60), ("CB", 25, 40), ("LB", 25, 20),
        ("CM", 45, 70), ("CDM", 40, 50), ("CM", 45, 30),
        ("RW", 75, 85), ("ST", 80, 50), ("LW", 75, 15),
    ],
    "4-2-3-1": [
        ("GK",  5, 50),
        ("RB", 25, 80), ("CB", 25, 60), ("CB", 25, 40), ("LB", 25, 20),
        ("CDM", 38, 60), ("CDM", 38, 40),
        ("RM", 55, 80), ("CAM", 60, 50), ("LM", 55, 20),
        ("ST", 80, 50),
    ],
    "3-5-2": [
        ("GK",  5, 50),
        ("CB", 25, 70), ("CB", 25, 50), ("CB", 25, 30),
        ("RWB", 45, 90), ("CM", 42, 65), ("CDM", 38, 50), ("CM", 42, 35), ("LWB", 45, 10),
        ("ST", 75, 60), ("ST", 75, 40),
    ],
    "5-3-2": [
        ("GK",  5, 50),
        ("RWB", 30, 90), ("CB", 22, 70), ("CB", 22, 50), ("CB", 22, 30), ("LWB", 30, 10),
        ("CM", 45, 65), ("CM", 42, 50), ("CM", 45, 35),
        ("ST", 75, 60), ("ST", 75, 40),
    ],
}

# Default fallback
DEFAULT_FORMATION = "4-4-2"


# ---------------------------------------------------------------------------
# Role Zone Constraints — (x_min, x_max, y_min, y_max) for home team
# Away team mirrors x: x -> 100-x
# ---------------------------------------------------------------------------

ROLE_ZONES: Dict[str, Tuple[float, float, float, float]] = {
    "GK":   (0, 15, 25, 75),
    "CB":   (10, 50, 15, 85),
    "LB":   (10, 55, 0, 40),
    "RB":   (10, 55, 60, 100),
    "LWB":  (10, 55, 0, 40),
    "RWB":  (10, 55, 60, 100),
    "CDM":  (25, 70, 10, 90),
    "CM":   (25, 70, 10, 90),
    "CAM":  (35, 80, 15, 85),
    "LM":   (25, 85, 0, 45),
    "RM":   (25, 85, 55, 100),
    "LW":   (25, 85, 0, 45),
    "RW":   (25, 85, 55, 100),
    "ST":   (40, 100, 15, 85),
    "CF":   (40, 100, 15, 85),
}


def _soft_clamp(value: float, low: float, high: float, steepness: float = 0.3) -> float:
    """Sigmoid-based soft clamp to keep value within [low, high]."""
    mid = (low + high) / 2
    half_range = (high - low) / 2
    if half_range <= 0:
        return mid
    # Rescale to [-1, 1] range
    normalized = (value - mid) / half_range
    # Soft sigmoid clamp
    clamped = normalized / (1 + abs(normalized) * steepness)
    return mid + clamped * half_range


def _mirror_x(x: float) -> float:
    """Mirror x-coordinate for away team."""
    return 100.0 - x


# ---------------------------------------------------------------------------
# Slot — internal state for one of the 22 players
# ---------------------------------------------------------------------------

@dataclass
class _Slot:
    slot_id: int
    team: str
    role: str
    template_x: float
    template_y: float
    kalman: KalmanFilter2D = field(default_factory=KalmanFilter2D)
    jersey_number: Optional[int] = None
    track_id: Optional[int] = None
    state: VisibilityState = VisibilityState.PREDICTED
    confidence: float = 0.5
    last_detected_frame: int = 0
    frames_since_detected: int = 999
    initialized: bool = False

    def __post_init__(self):
        self.kalman = KalmanFilter2D()
        # Tune Kalman for 0-100 coordinate space
        self.kalman.Q = np.diag([0.5, 0.5, 0.3, 0.3])
        self.kalman.R = np.diag([2.0, 2.0])
        self.kalman.P = np.diag([10.0, 10.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# OffScreenPredictor — main class
# ---------------------------------------------------------------------------

class OffScreenPredictor:
    """
    Maintains 22 PlayerPrediction slots (11 home + 11 away).
    Each slot has a KalmanFilter2D for position/velocity estimation.

    On each frame:
    1. Estimate camera frustum from detected players
    2. Match detections to slots via Hungarian algorithm
    3. Update matched slots with Kalman measurement update
    4. Predict unmatched slots with velocity decay + formation attractor
    5. Handle re-entry when players return to frame
    """

    def __init__(self):
        self.slots: Dict[str, List[_Slot]] = {"home": [], "away": []}
        self._formation_name: str = DEFAULT_FORMATION
        self._initialized = False
        self._frame_count = 0

    def _init_slots(self, formation_name: Optional[str] = None):
        """Initialize 22 slots from formation template."""
        fname = formation_name or self._formation_name
        if fname not in FORMATION_TEMPLATES:
            fname = DEFAULT_FORMATION
        self._formation_name = fname
        template = FORMATION_TEMPLATES[fname]

        self.slots["home"] = []
        self.slots["away"] = []

        for i, (role, x, y) in enumerate(template):
            # Home slot
            home_slot = _Slot(
                slot_id=i + 1, team="home", role=role,
                template_x=x, template_y=y,
            )
            home_slot.kalman.initialize(x, y)
            home_slot.initialized = True
            self.slots["home"].append(home_slot)

            # Away slot — mirror x
            away_slot = _Slot(
                slot_id=i + 1, team="away", role=role,
                template_x=_mirror_x(x), template_y=y,
            )
            away_slot.kalman.initialize(_mirror_x(x), y)
            away_slot.initialized = True
            self.slots["away"].append(away_slot)

        self._initialized = True

    def _update_formation(self, formation_name: str):
        """Update formation template positions without resetting Kalman state."""
        if formation_name not in FORMATION_TEMPLATES:
            return
        if formation_name == self._formation_name:
            return
        self._formation_name = formation_name
        template = FORMATION_TEMPLATES[formation_name]

        for team in ("home", "away"):
            for i, (role, x, y) in enumerate(template):
                if i < len(self.slots[team]):
                    slot = self.slots[team][i]
                    slot.role = role
                    if team == "away":
                        slot.template_x = _mirror_x(x)
                    else:
                        slot.template_x = x
                    slot.template_y = y

    def _estimate_frustum(self, detections: List[Dict]) -> CameraFrustum:
        """Estimate visible pitch area from detected player positions."""
        if not detections:
            return CameraFrustum()

        xs = [d["x"] for d in detections]
        ys = [d["y"] for d in detections]
        margin = settings.PREDICTION_FRUSTUM_MARGIN

        return CameraFrustum(
            x_min=max(0, min(xs) - margin),
            x_max=min(100, max(xs) + margin),
            y_min=max(0, min(ys) - margin),
            y_max=min(100, max(ys) + margin),
        )

    def _is_in_frustum(self, x: float, y: float, frustum: CameraFrustum) -> bool:
        """Check if a position is within the visible camera area."""
        return (frustum.x_min <= x <= frustum.x_max and
                frustum.y_min <= y <= frustum.y_max)

    def process_frame(
        self,
        frame_number: int,
        timestamp_ms: int,
        detections: List[Dict],
        formation_name: Optional[str] = None,
    ) -> List[PlayerPrediction]:
        """
        Process a frame and return 22 player predictions.

        Args:
            frame_number: Current frame number
            timestamp_ms: Timestamp in milliseconds
            detections: List of dicts with keys:
                x (float 0-100), y (float 0-100), team ('home'/'away'),
                jersey_number (Optional[int]), track_id (Optional[int])
            formation_name: Current detected formation name (e.g. "4-4-2")

        Returns:
            List of 22 PlayerPrediction objects (11 home + 11 away)
        """
        self._frame_count = frame_number

        # Initialize slots on first call or formation change
        if not self._initialized:
            self._init_slots(formation_name)
        elif formation_name:
            self._update_formation(formation_name)

        # Estimate camera frustum
        frustum = self._estimate_frustum(detections)

        # Split detections by team
        home_dets = [d for d in detections if d.get("team") == "home"]
        away_dets = [d for d in detections if d.get("team") == "away"]

        # Match and update each team
        self._match_and_update("home", home_dets, frame_number, frustum)
        self._match_and_update("away", away_dets, frame_number, frustum)

        # Build output
        return self._build_output()

    def _match_and_update(
        self,
        team: str,
        detections: List[Dict],
        frame_number: int,
        frustum: CameraFrustum,
    ):
        """Match detections to slots and update/predict each slot."""
        slots = self.slots[team]
        n_slots = len(slots)
        n_dets = len(detections)

        # --- Step 1: Hungarian matching ---
        matched_slot_indices: set = set()
        matched_det_indices: set = set()

        if n_dets > 0 and n_slots > 0:
            cost = np.full((n_dets, n_slots), 1000.0)

            for di, det in enumerate(detections):
                dx, dy = det["x"], det["y"]
                d_jersey = det.get("jersey_number")

                for si, slot in enumerate(slots):
                    # Predicted position from Kalman
                    sx, sy = slot.kalman.position

                    # Euclidean distance
                    dist = math.sqrt((dx - sx) ** 2 + (dy - sy) ** 2)

                    # Jersey number priority: exact match gets cost 0
                    if d_jersey and slot.jersey_number and d_jersey == slot.jersey_number:
                        dist = 0.0

                    cost[di, si] = dist

            row_idx, col_idx = linear_sum_assignment(cost)

            for di, si in zip(row_idx, col_idx):
                if cost[di, si] < settings.PREDICTION_REENTRY_THRESHOLD * 5:
                    matched_slot_indices.add(si)
                    matched_det_indices.add(di)

                    slot = slots[si]
                    det = detections[di]

                    # Re-entry: if too far, re-initialize Kalman
                    if cost[di, si] > settings.PREDICTION_REENTRY_THRESHOLD:
                        slot.kalman.initialize(det["x"], det["y"])
                    else:
                        slot.kalman.predict(dt=1.0)
                        slot.kalman.update(np.array([det["x"], det["y"]]))

                    slot.state = VisibilityState.DETECTED
                    slot.confidence = 1.0
                    slot.last_detected_frame = frame_number
                    slot.frames_since_detected = 0
                    slot.track_id = det.get("track_id")
                    if det.get("jersey_number"):
                        slot.jersey_number = det["jersey_number"]

        # --- Step 2: Handle unmatched detections ---
        # Assign to closest unmatched slot with reasonable distance
        for di in range(n_dets):
            if di in matched_det_indices:
                continue
            det = detections[di]
            best_si = None
            best_dist = settings.PREDICTION_REENTRY_THRESHOLD * 5

            for si in range(n_slots):
                if si in matched_slot_indices:
                    continue
                slot = slots[si]
                sx, sy = slot.kalman.position
                dist = math.sqrt((det["x"] - sx) ** 2 + (det["y"] - sy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_si = si

            if best_si is not None:
                matched_slot_indices.add(best_si)
                slot = slots[best_si]
                slot.kalman.initialize(det["x"], det["y"])
                slot.state = VisibilityState.DETECTED
                slot.confidence = 1.0
                slot.last_detected_frame = frame_number
                slot.frames_since_detected = 0
                slot.track_id = det.get("track_id")
                if det.get("jersey_number"):
                    slot.jersey_number = det["jersey_number"]

        # --- Step 3: Predict unmatched slots ---
        for si in range(n_slots):
            if si in matched_slot_indices:
                continue

            slot = slots[si]
            slot.frames_since_detected = frame_number - slot.last_detected_frame

            # Kalman predict
            state = slot.kalman.predict(dt=1.0)
            kx, ky = state[0], state[1]

            # Velocity decay
            decay_rate = settings.PREDICTION_VELOCITY_DECAY_RATE
            frames_since = slot.frames_since_detected
            vel_factor = max(0.0, 1.0 - decay_rate * frames_since)
            slot.kalman.x[2] *= vel_factor
            slot.kalman.x[3] *= vel_factor

            # Formation attractor blend
            blend_frames = settings.PREDICTION_FORMATION_BLEND_FRAMES
            blend = min(1.0, frames_since / blend_frames)
            kx = kx * (1.0 - blend) + slot.template_x * blend
            ky = ky * (1.0 - blend) + slot.template_y * blend

            # Role zone soft clamp
            zone = ROLE_ZONES.get(slot.role)
            if zone:
                zx_min, zx_max, zy_min, zy_max = zone
                if slot.team == "away":
                    zx_min, zx_max = _mirror_x(zx_max), _mirror_x(zx_min)
                kx = _soft_clamp(kx, zx_min, zx_max)
                ky = _soft_clamp(ky, zy_min, zy_max)

            # Hard clamp to pitch bounds
            kx = max(0.0, min(100.0, kx))
            ky = max(0.0, min(100.0, ky))

            # Write back position (not velocity — decay already applied)
            slot.kalman.x[0] = kx
            slot.kalman.x[1] = ky

            # Confidence decay: max(min_conf, 0.95 * exp(-decay * seconds))
            seconds_since = frames_since / 30.0  # assume ~30 fps analysis
            conf_decay = settings.PREDICTION_CONFIDENCE_DECAY
            min_conf = settings.PREDICTION_MIN_CONFIDENCE
            slot.confidence = max(min_conf, 0.95 * math.exp(-conf_decay * seconds_since))

            # State transition
            if frames_since < 90:  # < 3 seconds
                slot.state = VisibilityState.PREDICTED
            else:
                slot.state = VisibilityState.FORMATION_HOLD

    def _build_output(self) -> List[PlayerPrediction]:
        """Build 22 PlayerPrediction objects from current slot state."""
        output = []
        for team in ("home", "away"):
            for slot in self.slots[team]:
                px, py = slot.kalman.position
                output.append(PlayerPrediction(
                    slot_id=slot.slot_id,
                    team=slot.team,
                    jersey_number=slot.jersey_number,
                    x=px,
                    y=py,
                    state=slot.state,
                    confidence=slot.confidence,
                    last_detected_frame=slot.last_detected_frame,
                    frames_since_detected=slot.frames_since_detected,
                    role=slot.role,
                    track_id=slot.track_id,
                ))
        return output

    def get_team_predictions(self, team: str) -> List[PlayerPrediction]:
        """Get predictions for a single team."""
        return [p for p in self._build_output() if p.team == team]

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        all_preds = self._build_output()
        detected = sum(1 for p in all_preds if p.state == VisibilityState.DETECTED)
        predicted = sum(1 for p in all_preds if p.state == VisibilityState.PREDICTED)
        formation_hold = sum(1 for p in all_preds if p.state == VisibilityState.FORMATION_HOLD)
        avg_conf = sum(p.confidence for p in all_preds) / max(1, len(all_preds))
        return {
            "total": len(all_preds),
            "detected": detected,
            "predicted": predicted,
            "formation_hold": formation_hold,
            "avg_confidence": round(avg_conf, 3),
        }


# Singleton instance
offscreen_predictor = OffScreenPredictor()
