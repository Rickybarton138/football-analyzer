# Football Analyzer - Event Detection Specification

## Overview
This document specifies the event detection and statistics system for the Football Match Analyzer.

---

## 1. Event Detection Rules

### 1.1 Ball Transfer Events

#### Pass
- **Detection**: Ball moves from Player A to Player B on the **same team**
- **Success**: Ball received by teammate
- **Failure**: Ball intercepted by opponent
- **Attributes**:
  - Direction: Forward / Sideways / Backward (relative to attacking goal)
  - Pitch Third: Defensive / Middle / Attacking (horizontal zone where pass originated)
  - Player: Track ID of passer

#### Tackle
- **Detection**: Ball transfers from Team A to Team B while players are in close proximity
- **Proximity Threshold**: ~60 pixels (configurable)
- **Success**: Tackler's team gains possession
- **Failure**: Opponent retains possession (foul scenario)
- **Attributes**:
  - Tackler: Player who gained possession
  - Tackled: Player who lost possession
  - Pitch Third: Zone where tackle occurred

#### Header
- **Detection**: Ball contact occurs at upper portion of player bounding box (top 30%)
- **Success**: Same team retains possession OR goal scored
- **Failure**: Opponent gains possession
- **Attributes**:
  - Player: Track ID
  - Outcome: Clearance / Pass / Shot

#### Shot
- **Detection**: Ball moves from attacking player toward opponent's goal
- **On Target**: Ball saved by GK OR Goal scored
- **Off Target**: Ball goes behind goal line but not in goal
- **Goal**: Ball crosses goal line between posts
- **Attributes**:
  - Player: Shooter track ID
  - Distance: Estimated distance from goal
  - Pitch Third: Where shot was taken from

---

### 1.2 Set Piece Events

| Event | Detection | Stationary Duration | Location |
|-------|-----------|---------------------|----------|
| **Goal Kick** | Ball still on 6-yard line | 3+ seconds | In front of goal |
| **Kick-off / Restart** | Ball still on center spot | 3+ seconds | Center circle |
| **Free Kick** | Ball still outside both 18-yard boxes | 3+ seconds | On pitch |
| **Corner** | Ball still in corner arc | 3+ seconds | Corner of pitch |
| **Throw-in** | Ball enters pitch from outside sideline | Immediate | Sideline |

---

### 1.3 Match Periods

- **First Half**: Starts at first kick-off, ends when extended stoppage detected (>3 minutes no play)
- **Second Half**: Starts at second kick-off, ends at final whistle or video end
- **Half Detection**: Look for kick-off event after extended break

---

## 2. Team Statistics

### 2.1 Pitch Zones

The pitch is divided into a 3x3 grid:

```
+------------------+------------------+------------------+
|   DEFENSIVE      |     MIDDLE       |    ATTACKING     |
|     LEFT         |      LEFT        |      LEFT        |
+------------------+------------------+------------------+
|   DEFENSIVE      |     MIDDLE       |    ATTACKING     |
|    CENTER        |     CENTER       |     CENTER       |
+------------------+------------------+------------------+
|   DEFENSIVE      |     MIDDLE       |    ATTACKING     |
|     RIGHT        |      RIGHT       |      RIGHT       |
+------------------+------------------+------------------+
        ^                  ^                   ^
   Own Goal Line      Halfway Line      Opponent Goal Line
```

**Horizontal Thirds** (for statistics):
- Defensive Third: Own goal to 1/3 pitch
- Middle Third: 1/3 to 2/3 pitch
- Attacking Third: 2/3 to opponent goal

**Vertical Thirds** (for heatmaps/positioning):
- Left: Left sideline to 1/3 width
- Center: 1/3 to 2/3 width
- Right: 2/3 to right sideline

---

### 2.2 Required Team Statistics

#### Possession
- **Total Possession %**: Time each team has the ball
- **Possession by Third**: % in Defensive / Middle / Attacking thirds

#### Passing
| Statistic | Description |
|-----------|-------------|
| Total Passes | Count of all pass attempts |
| Successful Passes | Passes received by teammate |
| Failed Passes | Passes intercepted or out of play |
| Pass Accuracy % | Successful / Total * 100 |
| Forward Passes | Passes toward opponent goal |
| Sideways Passes | Passes perpendicular to goal |
| Backward Passes | Passes toward own goal |
| Passes by Third | Breakdown by Defensive/Middle/Attacking |
| Longest Pass Sequence | Most consecutive successful passes |

#### Shooting
| Statistic | Description |
|-----------|-------------|
| Total Shots | All shot attempts |
| Shots on Target | Saved by GK or Goal |
| Shots off Target | Missed the goal |
| Shot Accuracy % | On Target / Total * 100 |
| Goals | Shots that crossed goal line |

#### Other
| Statistic | Description |
|-----------|-------------|
| Tackles | Successful ball wins |
| Headers | Aerial duels |
| Corners | Corner kicks awarded |
| Free Kicks | Free kicks awarded |
| Throw-ins | Throw-ins taken |

---

## 3. Technical Parameters

### 3.1 Configurable Thresholds

```python
# Proximity thresholds (pixels)
TACKLE_PROXIMITY = 60          # Max distance for tackle detection
POSSESSION_PROXIMITY = 40      # Max distance for ball possession
HEADER_ZONE_RATIO = 0.30       # Top 30% of bbox = header zone

# Stationary detection
STATIONARY_THRESHOLD = 5       # Max pixels movement
STATIONARY_DURATION = 2.0      # Seconds ball must be still

# Pitch zones (normalized 0-100)
DEFENSIVE_THIRD_END = 33.3
MIDDLE_THIRD_END = 66.6
LEFT_THIRD_END = 33.3
RIGHT_THIRD_START = 66.6

# Shot detection
GOAL_ZONE_Y_MIN = 35           # Goal vertical range (normalized)
GOAL_ZONE_Y_MAX = 65
SHOT_DIRECTION_THRESHOLD = 0.7 # Ball must be moving 70%+ toward goal
```

### 3.2 Team Identification

**Method**: User-assisted color clustering
1. During upload, user selects "Home" and "Away" jersey colors from palette
2. System clusters detected jersey colors to nearest selection
3. Goalkeepers identified by `is_goalkeeper` flag + different color from outfield

**Fallback**: Auto-cluster two most common non-green colors (grass exclusion)

---

## 4. Data Structures

### 4.1 Event Record

```python
@dataclass
class MatchEvent:
    event_id: str
    event_type: str  # pass, tackle, shot, goal, header, corner, etc.
    timestamp_ms: int
    frame_number: int
    period: int  # 1 = first half, 2 = second half

    # Location
    pitch_x: float  # 0-100, 0 = own goal line
    pitch_y: float  # 0-100, 0 = left sideline
    pitch_third: str  # defensive, middle, attacking

    # Players involved
    player_id: int  # Primary player (passer, shooter, tackler)
    player_team: str  # home, away
    target_player_id: Optional[int]  # Recipient, tackled player
    target_team: Optional[str]

    # Outcome
    success: bool
    outcome: str  # goal, save, miss, completed, intercepted, etc.

    # Pass-specific
    direction: Optional[str]  # forward, sideways, backward

    # Shot-specific
    on_target: Optional[bool]
```

### 4.2 Team Statistics Record

```python
@dataclass
class TeamStats:
    team: str  # home, away
    period: int  # 0 = full match, 1 = first half, 2 = second half

    # Possession
    possession_pct: float
    possession_defensive_third_pct: float
    possession_middle_third_pct: float
    possession_attacking_third_pct: float

    # Passing
    passes_total: int
    passes_successful: int
    passes_failed: int
    pass_accuracy_pct: float
    passes_forward: int
    passes_sideways: int
    passes_backward: int
    passes_defensive_third: int
    passes_middle_third: int
    passes_attacking_third: int
    longest_pass_sequence: int

    # Shooting
    shots_total: int
    shots_on_target: int
    shots_off_target: int
    shot_accuracy_pct: float
    goals: int

    # Other
    tackles: int
    headers: int
    corners: int
    free_kicks: int
    throw_ins: int
```

---

## 5. Implementation Priority

### Phase 1: Core Detection
1. Team classification (jersey colors)
2. Ball possession tracking
3. Pass detection (basic)
4. Shot/Goal detection

### Phase 2: Advanced Events
1. Tackle detection
2. Header detection
3. Pass direction classification
4. Set piece detection

### Phase 3: Statistics
1. Possession calculation
2. Pass statistics by zone
3. Shot statistics
4. Longest pass sequence

### Phase 4: UI Integration
1. Team stats comparison view
2. Event timeline
3. Player event lists
4. Clip generation from events

---

## 6. Questions Resolved

| Question | Answer |
|----------|--------|
| Pitch calibration | Use homography if available, else estimate from frame |
| Tackle proximity | 60 pixels (~2-3m on pitch) |
| Header zone | Top 30% of player bounding box |
| Stationary threshold | <5 pixels movement over 2-3 seconds |
| Team identification | User picks colors at upload, auto-cluster as fallback |
