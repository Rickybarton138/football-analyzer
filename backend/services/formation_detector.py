"""
Formation Detection Service

Detects team formations (4-4-2, 4-3-3, 3-5-2, etc.) from player positions.
Uses clustering and position analysis to identify defensive lines.

VEO Camera Perspective (SIDELINE VIEW):
- Camera is positioned at the SIDE of the pitch near the halfway line
- Video X-axis (left to right) = LENGTH of pitch (goal to goal)
- Video Y-axis (top to bottom) = WIDTH of pitch (near touchline to far touchline)
- Left side of video = one goal (e.g., home team's goal)
- Right side of video = other goal (e.g., away team's goal)
- Top of video = far touchline
- Bottom of video = near touchline (camera side)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans


@dataclass
class FormationSnapshot:
    """A detected formation at a specific moment."""
    timestamp: float
    frame_number: int
    team: str
    formation: str  # e.g., "4-4-2", "4-3-3"
    confidence: float
    player_positions: List[Tuple[float, float]]  # (pitch_x, pitch_y) in normalized coords
    lines: List[int]  # e.g., [4, 4, 2] for 4-4-2
    line_depths: List[float]  # Average Y position of each line


@dataclass
class FormationStats:
    """Formation statistics over the match."""
    team: str
    primary_formation: str
    formation_counts: Dict[str, int]
    avg_defensive_line_height: float  # 0-100, where 100 is opponent goal
    avg_team_width: float  # pixels
    avg_compactness: float  # average distance between players
    formation_changes: int
    avg_line_depths: Dict[str, float]  # defense, midfield, attack line positions


COMMON_FORMATIONS = [
    "4-4-2", "4-3-3", "4-2-3-1", "4-1-4-1",
    "3-5-2", "3-4-3", "5-3-2", "5-4-1",
    "4-5-1", "4-4-1-1", "4-3-2-1", "3-4-1-2"
]

# Video dimensions
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


class FormationDetector:
    """
    Detects team formations from player positions.

    VEO Camera Understanding (SIDELINE VIEW):
    - Camera is at the SIDE of the pitch near halfway line
    - Video X-axis (horizontal, 0-1920) = LENGTH of pitch (goal to goal)
    - Video Y-axis (vertical, 0-1080) = WIDTH of pitch (far touchline to near touchline)
    - Left of video (X=0) = one goal (typically home team defends this)
    - Right of video (X=1920) = other goal (home team attacks this)

    For Formation Detection:
    - Lines are determined by X-position (depth on pitch)
    - Width spread is determined by Y-position
    - Home team: GK near X=0, attacks toward X=1920
    - Away team: GK near X=1920, attacks toward X=0

    Algorithm:
    1. Identify team direction based on average X position
    2. Find goalkeeper (player furthest back toward their goal)
    3. Cluster remaining outfield players by X-position (depth)
    4. Count players in each depth band (defense -> midfield -> attack)
    5. Match to known formations
    """

    def __init__(self):
        self.home_formations: List[FormationSnapshot] = []
        self.away_formations: List[FormationSnapshot] = []
        self.home_stats: Optional[FormationStats] = None
        self.away_stats: Optional[FormationStats] = None

        # Detection parameters
        self.min_players_for_detection = 7  # Need at least 7 outfield players
        self.line_cluster_threshold_pct = 8  # % of pitch width for line grouping
        self.line_cluster_threshold = 8  # For legacy code compatibility

    def reset(self):
        """Reset detector for new analysis."""
        self.home_formations = []
        self.away_formations = []
        self.home_stats = None
        self.away_stats = None

    def _transform_to_pitch_coords(self, video_x: float, video_y: float,
                                   team: str, home_attacks_right: bool = True) -> Tuple[float, float]:
        """
        Transform video pixel coordinates to normalized pitch coordinates.

        For VEO SIDELINE camera:
        - Video X (0-1920) maps to pitch length (goal to goal)
        - Video Y (0-1080) maps to pitch width (touchline to touchline)

        Returns:
            (depth_pct, width_pct) where:
            - depth_pct: 0-100 from team's own goal to opponent's goal
            - width_pct: 0-100 from left touchline to right touchline
        """
        # Normalize to 0-100
        video_x_pct = (video_x / VIDEO_WIDTH) * 100
        video_y_pct = (video_y / VIDEO_HEIGHT) * 100

        # Width position is straightforward (Y in video = width on pitch)
        # Note: Y=0 is top of video (far touchline), Y=1080 is bottom (near touchline/camera)
        width_pct = video_y_pct

        # Depth depends on which team and attack direction
        if team == 'home':
            if home_attacks_right:
                # Home attacks right: X=0 is their goal, X=1920 is opponent goal
                depth_pct = video_x_pct  # 0 = own goal, 100 = opponent goal
            else:
                depth_pct = 100 - video_x_pct
        else:  # away team
            if home_attacks_right:
                # Away attacks left: X=1920 is their goal, X=0 is opponent goal
                depth_pct = 100 - video_x_pct  # Invert for away team
            else:
                depth_pct = video_x_pct

        return depth_pct, width_pct

    def detect_formation(self, players: List[Tuple[float, float]],
                         team: str, home_attacks_right: bool = True) -> Optional[Tuple[str, List[int], float, List[float]]]:
        """
        Detect formation from player positions using depth-based clustering.

        For SIDELINE camera view:
        - X-position in video = depth on pitch (distance from goal line)
        - Players are clustered by their X-position into defensive lines

        Args:
            players: List of (video_x, video_y) positions for team players
            team: 'home' or 'away'
            home_attacks_right: True if home team attacks towards right of video

        Returns:
            Tuple of (formation_string, lines, confidence, line_depths) or None
        """
        if len(players) < self.min_players_for_detection:
            return None

        # Transform all positions to pitch coordinates (depth, width)
        pitch_positions = [
            self._transform_to_pitch_coords(p[0], p[1], team, home_attacks_right)
            for p in players
        ]

        # Sort by depth (0 = own goal, 100 = opponent goal)
        sorted_by_depth = sorted(pitch_positions, key=lambda p: p[0])

        # Identify goalkeeper: player closest to own goal (lowest depth)
        # Only remove if significantly behind others (typical GK position)
        if len(sorted_by_depth) > 1:
            gk_depth = sorted_by_depth[0][0]
            next_player_depth = sorted_by_depth[1][0]

            # GK should be at least 10% of pitch behind the defensive line
            if next_player_depth - gk_depth > 10:
                outfield = sorted_by_depth[1:]  # Remove GK
            else:
                # Still likely the deepest player is GK even if close to defenders
                outfield = sorted_by_depth[1:]
        else:
            return None

        if len(outfield) < 6:
            return None

        # Cluster outfield players by depth into lines
        depths = np.array([p[0] for p in outfield]).reshape(-1, 1)

        # Use K-means to find optimal number of lines (2-4)
        best_formation = None
        best_score = -1

        for n_lines in [2, 3, 4]:
            if len(outfield) < n_lines:
                continue

            try:
                kmeans = KMeans(n_clusters=n_lines, random_state=42, n_init=10)
                labels = kmeans.fit_predict(depths)
                centers = kmeans.cluster_centers_.flatten()

                # Count players in each cluster, sort by depth (defense first)
                sorted_indices = np.argsort(centers)
                lines = []
                line_depths = []

                for idx in sorted_indices:
                    count = int(np.sum(labels == idx))
                    lines.append(count)
                    line_depths.append(float(centers[idx]))

                # Validate formation
                formation = "-".join(str(n) for n in lines)
                total_players = sum(lines)

                # Score the formation
                score = 0

                # Bonus for matching known formations
                if formation in COMMON_FORMATIONS:
                    score += 50
                elif self._is_similar_formation(formation):
                    score += 30

                # Bonus for having ~10 outfield players
                if total_players == 10:
                    score += 20
                elif total_players >= 9:
                    score += 10
                else:
                    score -= abs(total_players - 10) * 5

                # Bonus for balanced lines (defense should have 3-5, not 1-2)
                if lines[0] >= 3:  # Defensive line should have at least 3
                    score += 10

                # Penalize very unbalanced lines
                max_line = max(lines)
                min_line = min(lines)
                if max_line - min_line > 4:
                    score -= 15

                # Check line separation (lines should be distinct)
                if len(line_depths) > 1:
                    min_gap = min(line_depths[i+1] - line_depths[i] for i in range(len(line_depths)-1))
                    if min_gap > 8:  # Lines should be at least 8% of pitch apart
                        score += 10

                if score > best_score:
                    best_score = score
                    best_formation = (formation, lines, line_depths)

            except Exception:
                continue

        if best_formation is None:
            return None

        formation, lines, line_depths = best_formation

        # Calculate confidence
        confidence = 0.85 if formation in COMMON_FORMATIONS else 0.6
        if sum(lines) >= 9:
            confidence += 0.1
        if len(lines) >= 3:
            confidence += 0.05

        return formation, lines, min(confidence, 1.0), line_depths

    def _is_similar_formation(self, formation: str) -> bool:
        """Check if formation is similar to a known one (within 1 player per line)."""
        try:
            parts = [int(x) for x in formation.split('-')]
            for known in COMMON_FORMATIONS:
                known_parts = [int(x) for x in known.split('-')]
                if len(parts) == len(known_parts):
                    diff = sum(abs(a - b) for a, b in zip(parts, known_parts))
                    if diff <= 2:
                        return True
        except:
            pass
        return False

    def _cluster_into_lines(self, y_positions: List[float]) -> Tuple[List[int], List[float]]:
        """
        Cluster Y-positions (depth) into defensive lines.

        Returns tuple of (player counts per line, average depth per line).
        Lines ordered from defense to attack.
        """
        if not y_positions:
            return [], []

        positions = np.array(y_positions)

        # Use gap-based clustering
        sorted_pos = np.sort(positions)
        lines = []
        line_positions = []
        current_line = [sorted_pos[0]]

        for i in range(1, len(sorted_pos)):
            gap = sorted_pos[i] - sorted_pos[i-1]

            if gap > self.line_cluster_threshold:
                lines.append(len(current_line))
                line_positions.append(np.mean(current_line))
                current_line = [sorted_pos[i]]
            else:
                current_line.append(sorted_pos[i])

        if current_line:
            lines.append(len(current_line))
            line_positions.append(np.mean(current_line))

        # Normalize to 2-4 lines
        lines, line_positions = self._normalize_lines_with_depths(lines, line_positions)

        return lines, line_positions

    def _normalize_lines_with_depths(self, lines: List[int],
                                      depths: List[float]) -> Tuple[List[int], List[float]]:
        """Normalize lines to 2-4 while preserving depth information."""
        if not lines:
            return [], []

        # Merge lines with only 1 player
        normalized_lines = []
        normalized_depths = []

        i = 0
        while i < len(lines):
            if lines[i] == 1:
                if normalized_lines:
                    # Merge with previous
                    normalized_lines[-1] += 1
                    normalized_depths[-1] = (normalized_depths[-1] + depths[i]) / 2
                elif i + 1 < len(lines):
                    # Merge with next
                    lines[i + 1] += 1
                    depths[i + 1] = (depths[i] + depths[i + 1]) / 2
                else:
                    normalized_lines.append(lines[i])
                    normalized_depths.append(depths[i])
            else:
                normalized_lines.append(lines[i])
                normalized_depths.append(depths[i])
            i += 1

        # Merge if more than 4 lines
        while len(normalized_lines) > 4:
            min_sum = float('inf')
            merge_idx = 0
            for i in range(len(normalized_lines) - 1):
                if normalized_lines[i] + normalized_lines[i+1] < min_sum:
                    min_sum = normalized_lines[i] + normalized_lines[i+1]
                    merge_idx = i

            total = normalized_lines[merge_idx] + normalized_lines[merge_idx + 1]
            avg_depth = (normalized_depths[merge_idx] + normalized_depths[merge_idx + 1]) / 2
            normalized_lines[merge_idx] = total
            normalized_depths[merge_idx] = avg_depth
            normalized_lines.pop(merge_idx + 1)
            normalized_depths.pop(merge_idx + 1)

        return normalized_lines, normalized_depths

    def _normalize_lines(self, lines: List[int]) -> List[int]:
        """
        Normalize detected lines to standard formations.

        Merge very small lines or split large ones.
        """
        if not lines:
            return []

        # Merge lines with only 1 player into adjacent lines
        normalized = []
        i = 0
        while i < len(lines):
            if lines[i] == 1 and normalized:
                # Merge with previous line
                normalized[-1] += 1
            elif lines[i] == 1 and i + 1 < len(lines):
                # Merge with next line
                lines[i + 1] += 1
            else:
                normalized.append(lines[i])
            i += 1

        # Ensure we have 2-4 lines (defense, midfield, attack)
        while len(normalized) > 4:
            # Merge two smallest adjacent lines
            min_sum = float('inf')
            merge_idx = 0
            for i in range(len(normalized) - 1):
                if normalized[i] + normalized[i+1] < min_sum:
                    min_sum = normalized[i] + normalized[i+1]
                    merge_idx = i
            normalized[merge_idx] += normalized[merge_idx + 1]
            normalized.pop(merge_idx + 1)

        return normalized

    def process_frame(self, frame_data: dict) -> Dict[str, Optional[FormationSnapshot]]:
        """
        Process a frame and detect formations for both teams.

        For SIDELINE camera (VEO):
        - Video X = pitch length (goal to goal)
        - Video Y = pitch width (touchline to touchline)
        - Home team attacks RIGHT (toward high X values)
        - Away team attacks LEFT (toward low X values)

        Args:
            frame_data: Frame analysis data with detections

        Returns:
            Dictionary with 'home' and 'away' FormationSnapshots (or None)
        """
        frame_num = frame_data.get('frame_number', 0)
        timestamp = frame_data.get('timestamp', 0.0)
        detections = frame_data.get('detections', [])

        # Separate players by team
        home_players = []
        away_players = []

        for det in detections:
            team = det.get('team')
            bbox = det.get('bbox', [0, 0, 0, 0])
            # Use center-bottom of bbox as player position (feet position)
            center_x = (bbox[0] + bbox[2]) / 2
            bottom_y = bbox[3]
            pos = (center_x, bottom_y)

            if team == 'home':
                home_players.append(pos)
            elif team == 'away':
                away_players.append(pos)

        results = {'home': None, 'away': None}

        # Detect home formation (home attacks right in video)
        if len(home_players) >= self.min_players_for_detection:
            result = self.detect_formation(home_players, 'home', home_attacks_right=True)
            if result:
                formation, lines, confidence, line_depths = result
                snapshot = FormationSnapshot(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    team='home',
                    formation=formation,
                    confidence=confidence,
                    player_positions=home_players,
                    lines=lines,
                    line_depths=line_depths
                )
                self.home_formations.append(snapshot)
                results['home'] = snapshot

        # Detect away formation (away attacks left in video)
        if len(away_players) >= self.min_players_for_detection:
            result = self.detect_formation(away_players, 'away', home_attacks_right=True)
            if result:
                formation, lines, confidence, line_depths = result
                snapshot = FormationSnapshot(
                    timestamp=timestamp,
                    frame_number=frame_num,
                    team='away',
                    formation=formation,
                    confidence=confidence,
                    player_positions=away_players,
                    lines=lines,
                    line_depths=line_depths
                )
                self.away_formations.append(snapshot)
                results['away'] = snapshot

        return results

    def _calculate_team_metrics(self, positions: List[Tuple[float, float]],
                                attacking_right: bool) -> Dict:
        """Calculate tactical metrics from player positions."""
        if len(positions) < 3:
            return {}

        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # Team width (lateral spread)
        width = max(y_coords) - min(y_coords)

        # Defensive line height (x position of deepest defender, excluding GK)
        sorted_x = sorted(x_coords)
        if attacking_right:
            def_line = sorted_x[1] if len(sorted_x) > 1 else sorted_x[0]  # Exclude GK
        else:
            def_line = sorted_x[-2] if len(sorted_x) > 1 else sorted_x[-1]

        # Normalize defensive line to 0-100 scale
        def_line_normalized = (def_line / 1920) * 100

        # Compactness (average distance between all players)
        distances = []
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distances.append(dist)

        compactness = np.mean(distances) if distances else 0

        return {
            'width': width,
            'defensive_line': def_line_normalized,
            'compactness': compactness
        }

    def calculate_stats(self) -> Dict[str, FormationStats]:
        """Calculate formation statistics for both teams."""
        stats = {}

        for team, formations in [('home', self.home_formations), ('away', self.away_formations)]:
            if not formations:
                continue

            # Count formations
            formation_counts = Counter(f.formation for f in formations)
            primary = formation_counts.most_common(1)[0][0] if formation_counts else "Unknown"

            # Calculate average metrics
            all_positions = [f.player_positions for f in formations]
            attacking_right = (team == 'home')

            metrics_list = [
                self._calculate_team_metrics(pos, attacking_right)
                for pos in all_positions if pos
            ]

            avg_def_line = np.mean([m['defensive_line'] for m in metrics_list if m]) if metrics_list else 50
            avg_width = np.mean([m['width'] for m in metrics_list if m]) if metrics_list else 0
            avg_compact = np.mean([m['compactness'] for m in metrics_list if m]) if metrics_list else 0

            # Calculate average line depths from formations with line_depths
            avg_line_depths = {}
            formations_with_depths = [f for f in formations if hasattr(f, 'line_depths') and f.line_depths]
            if formations_with_depths:
                # Get the most common number of lines
                num_lines = Counter(len(f.line_depths) for f in formations_with_depths).most_common(1)[0][0]
                matching_formations = [f for f in formations_with_depths if len(f.line_depths) == num_lines]

                if matching_formations:
                    line_names = ['defense', 'midfield', 'attack'][:num_lines]
                    if num_lines == 4:
                        line_names = ['defense', 'def_mid', 'att_mid', 'attack']

                    for i, name in enumerate(line_names):
                        depths = [f.line_depths[i] for f in matching_formations]
                        avg_line_depths[name] = round(np.mean(depths), 1)

            # Count formation changes
            changes = 0
            prev_formation = None
            for f in formations:
                if prev_formation and f.formation != prev_formation:
                    changes += 1
                prev_formation = f.formation

            team_stats = FormationStats(
                team=team,
                primary_formation=primary,
                formation_counts=dict(formation_counts),
                avg_defensive_line_height=round(avg_def_line, 1),
                avg_team_width=round(avg_width, 1),
                avg_compactness=round(avg_compact, 1),
                formation_changes=changes,
                avg_line_depths=avg_line_depths
            )

            if team == 'home':
                self.home_stats = team_stats
            else:
                self.away_stats = team_stats

            stats[team] = team_stats

        return stats

    def get_formation_timeline(self) -> Dict[str, List[Dict]]:
        """Get formation changes over time."""
        timeline = {'home': [], 'away': []}

        for team, formations in [('home', self.home_formations), ('away', self.away_formations)]:
            prev_formation = None
            for f in formations:
                if f.formation != prev_formation:
                    timeline[team].append({
                        'timestamp': f.timestamp,
                        'frame': f.frame_number,
                        'formation': f.formation,
                        'confidence': f.confidence
                    })
                    prev_formation = f.formation

        return timeline

    def analyze_from_frames(self, frame_analyses: List[dict]) -> Dict:
        """
        Analyze formations from frame data.

        Args:
            frame_analyses: List of frame analysis dictionaries

        Returns:
            Dictionary with formation statistics
        """
        self.reset()

        # Sample frames (every 30th frame = 1 per second at 30fps)
        sampled_frames = frame_analyses[::30] if len(frame_analyses) > 100 else frame_analyses

        for frame_data in sampled_frames:
            self.process_frame(frame_data)

        stats = self.calculate_stats()
        timeline = self.get_formation_timeline()

        # Default stats for teams without data
        default_stats = FormationStats('unknown', 'Unknown', {}, 50, 0, 0, 0, {})

        home_stats = stats.get('home', default_stats)
        away_stats = stats.get('away', default_stats)

        return {
            'home': {
                'primary_formation': home_stats.primary_formation,
                'formation_counts': home_stats.formation_counts,
                'avg_defensive_line': home_stats.avg_defensive_line_height,
                'avg_compactness': home_stats.avg_compactness,
                'avg_team_width': home_stats.avg_team_width,
                'formation_changes': home_stats.formation_changes,
                'avg_line_depths': home_stats.avg_line_depths,
                'timeline': timeline.get('home', [])
            },
            'away': {
                'primary_formation': away_stats.primary_formation,
                'formation_counts': away_stats.formation_counts,
                'avg_defensive_line': away_stats.avg_defensive_line_height,
                'avg_compactness': away_stats.avg_compactness,
                'avg_team_width': away_stats.avg_team_width,
                'formation_changes': away_stats.formation_changes,
                'avg_line_depths': away_stats.avg_line_depths,
                'timeline': timeline.get('away', [])
            }
        }


# Singleton instance
formation_detector = FormationDetector()
