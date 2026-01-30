"""
Player Identity Management - Database models and schemas for persistent player tracking.

This module provides player re-identification (ReID) capabilities by maintaining
player identities across track fragment changes using:
1. Positional role assignment (e.g., Left Back in 4-4-2)
2. Visual appearance features
3. Manual labeling by coaches
"""
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np


class PlayerIdentity(BaseModel):
    """Persistent player identity that survives track_id changes."""

    player_id: str = Field(..., description="Unique persistent player ID (UUID)")
    team: str = Field(..., description="Team: 'home' or 'away'")
    jersey_number: Optional[int] = Field(None, description="Jersey number if known")
    positional_role: Optional[str] = Field(None, description="Tactical role: LB, CB, RB, LM, CM, RM, ST, GK, etc.")

    # Visual features for ReID
    appearance_features: Optional[List[float]] = Field(None, description="Visual feature vector for ReID")
    jersey_color_bgr: Optional[List[int]] = Field(None, description="Dominant jersey color [B, G, R]")

    # Tracking history
    track_id_history: List[int] = Field(default_factory=list, description="All track_ids associated with this player")
    first_seen_frame: int = Field(..., description="Frame number where player first appeared")
    last_seen_frame: int = Field(..., description="Frame number where player last appeared")

    # Position tracking (normalized pitch coordinates)
    typical_x_min: float = Field(0.0, description="Minimum X position on pitch (0-1)")
    typical_x_max: float = Field(1.0, description="Maximum X position on pitch (0-1)")
    typical_y_min: float = Field(0.0, description="Minimum Y position on pitch (0-1)")
    typical_y_max: float = Field(1.0, description="Maximum Y position on pitch (0-1)")

    # Manual labeling
    manually_labeled: bool = Field(False, description="Was this player manually labeled by coach?")
    label_confidence: float = Field(0.5, description="Confidence in identity (0-1)")

    # Ball possession tracking
    ball_touches: int = Field(0, description="Number of times player touched the ball")
    frames_with_ball: int = Field(0, description="Number of frames where player possessed the ball")

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "550e8400-e29b-41d4-a716-446655440000",
                "team": "home",
                "jersey_number": 10,
                "positional_role": "CAM",
                "track_id_history": [12, 45, 67, 89],
                "first_seen_frame": 0,
                "last_seen_frame": 5400,
                "typical_x_min": 0.3,
                "typical_x_max": 0.7,
                "typical_y_min": 0.4,
                "typical_y_max": 0.8,
                "manually_labeled": True,
                "label_confidence": 1.0
            }
        }


class FormationRole(BaseModel):
    """Positional role within a tactical formation."""

    role: str = Field(..., description="Role name: GK, LB, CB, RB, LM, CM, RM, CAM, ST, etc.")
    zone_x_min: float = Field(..., description="Expected zone X min (0-1)")
    zone_x_max: float = Field(..., description="Expected zone X max (0-1)")
    zone_y_min: float = Field(..., description="Expected zone Y min (0-1)")
    zone_y_max: float = Field(..., description="Expected zone Y max (0-1)")

    def contains_position(self, x: float, y: float, tolerance: float = 0.25) -> bool:
        """Check if position is within this role's zone (with tolerance)."""
        return (
            self.zone_x_min - tolerance <= x <= self.zone_x_max + tolerance and
            self.zone_y_min - tolerance <= y <= self.zone_y_max + tolerance
        )

    class Config:
        json_schema_extra = {
            "example": {
                "role": "LB",
                "zone_x_min": 0.0,
                "zone_x_max": 0.25,
                "zone_y_min": 0.25,
                "zone_y_max": 0.5
            }
        }


class FormationTemplate(BaseModel):
    """Formation template defining expected player positions."""

    formation_name: str = Field(..., description="Formation name: 4-4-2, 4-3-3, etc.")
    roles: List[FormationRole] = Field(..., description="List of positional roles")

    @staticmethod
    def get_formation_442() -> "FormationTemplate":
        """Standard 4-4-2 formation template."""
        return FormationTemplate(
            formation_name="4-4-2",
            roles=[
                # Goalkeeper
                FormationRole(role="GK", zone_x_min=0.0, zone_x_max=0.1, zone_y_min=0.3, zone_y_max=0.7),
                # Defense (4)
                FormationRole(role="LB", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.0, zone_y_max=0.25),
                FormationRole(role="CB_L", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.25, zone_y_max=0.5),
                FormationRole(role="CB_R", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.5, zone_y_max=0.75),
                FormationRole(role="RB", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.75, zone_y_max=1.0),
                # Midfield (4)
                FormationRole(role="LM", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.0, zone_y_max=0.3),
                FormationRole(role="CM_L", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.3, zone_y_max=0.5),
                FormationRole(role="CM_R", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.5, zone_y_max=0.7),
                FormationRole(role="RM", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.7, zone_y_max=1.0),
                # Attack (2)
                FormationRole(role="ST_L", zone_x_min=0.6, zone_x_max=0.85, zone_y_min=0.2, zone_y_max=0.5),
                FormationRole(role="ST_R", zone_x_min=0.6, zone_x_max=0.85, zone_y_min=0.5, zone_y_max=0.8),
            ]
        )

    @staticmethod
    def get_formation_433() -> "FormationTemplate":
        """Standard 4-3-3 formation template."""
        return FormationTemplate(
            formation_name="4-3-3",
            roles=[
                # Goalkeeper
                FormationRole(role="GK", zone_x_min=0.0, zone_x_max=0.1, zone_y_min=0.3, zone_y_max=0.7),
                # Defense (4)
                FormationRole(role="LB", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.0, zone_y_max=0.25),
                FormationRole(role="CB_L", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.25, zone_y_max=0.5),
                FormationRole(role="CB_R", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.5, zone_y_max=0.75),
                FormationRole(role="RB", zone_x_min=0.15, zone_x_max=0.3, zone_y_min=0.75, zone_y_max=1.0),
                # Midfield (3)
                FormationRole(role="CM_L", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.15, zone_y_max=0.4),
                FormationRole(role="CDM", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.4, zone_y_max=0.6),
                FormationRole(role="CM_R", zone_x_min=0.35, zone_x_max=0.55, zone_y_min=0.6, zone_y_max=0.85),
                # Attack (3)
                FormationRole(role="LW", zone_x_min=0.6, zone_x_max=0.85, zone_y_min=0.0, zone_y_max=0.3),
                FormationRole(role="ST", zone_x_min=0.6, zone_x_max=0.85, zone_y_min=0.35, zone_y_max=0.65),
                FormationRole(role="RW", zone_x_min=0.6, zone_x_max=0.85, zone_y_min=0.7, zone_y_max=1.0),
            ]
        )


class PlayerLabelRequest(BaseModel):
    """Request to manually label a player."""

    video_id: str = Field(..., description="Video/match ID")
    frame_number: int = Field(..., description="Frame number where player was labeled")
    track_id: int = Field(..., description="Current track_id of player")
    jersey_number: Optional[int] = Field(None, description="Jersey number (if visible)")
    positional_role: Optional[str] = Field(None, description="Positional role (e.g., LB, CM, ST)")
    team: str = Field(..., description="Team: 'home' or 'away'")

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "28593ff1-ba43-457a-a41e-1fc737052323",
                "frame_number": 150,
                "track_id": 12,
                "jersey_number": 10,
                "positional_role": "CAM",
                "team": "home"
            }
        }


class PlayerIdentityDatabase:
    """In-memory database for player identities (could be replaced with SQLite/PostgreSQL)."""

    def __init__(self):
        # video_id -> list of PlayerIdentity objects
        self._identities: Dict[str, List[PlayerIdentity]] = {}
        # "video_id_team" -> formation_name (e.g., "abc123_home" -> "4-4-2")
        self._formations: Dict[str, str] = {}

    def get_or_create_identity(
        self,
        video_id: str,
        team: str,
        track_id: int,
        frame_number: int,
        pitch_x: float,
        pitch_y: float,
        jersey_color: Optional[List[int]] = None
    ) -> PlayerIdentity:
        """
        Get existing player identity from the kickoff master list.
        Maps new track_ids to the closest existing master player.
        """
        if video_id not in self._identities:
            self._identities[video_id] = []

        identities = self._identities[video_id]

        # PRIORITY 1: Try to find existing identity by track_id (fast path)
        for identity in identities:
            if track_id in identity.track_id_history and identity.team == team:
                identity.last_seen_frame = frame_number
                return identity

        # PRIORITY 2: If kickoff master list exists, map to closest master player
        team_players = [p for p in identities if p.team == team and p.label_confidence >= 0.9]

        if len(team_players) > 0:
            # Find closest master player by position
            import math
            closest_player = None
            min_distance = float('inf')

            for player in team_players:
                # Calculate center of player's typical position
                player_x = (player.typical_x_min + player.typical_x_max) / 2
                player_y = (player.typical_y_min + player.typical_y_max) / 2

                # Calculate distance
                distance = math.sqrt((pitch_x - player_x)**2 + (pitch_y - player_y)**2)

                if distance < min_distance:
                    min_distance = distance
                    closest_player = player

            # Map this track_id to the closest master player
            # Always map to closest player when master list exists (no distance threshold)
            if closest_player:
                if track_id not in closest_player.track_id_history:
                    closest_player.track_id_history.append(track_id)
                closest_player.last_seen_frame = frame_number
                # Update position bounds
                closest_player.typical_x_min = min(closest_player.typical_x_min, pitch_x)
                closest_player.typical_x_max = max(closest_player.typical_x_max, pitch_x)
                closest_player.typical_y_min = min(closest_player.typical_y_min, pitch_y)
                closest_player.typical_y_max = max(closest_player.typical_y_max, pitch_y)
                return closest_player

        # PRIORITY 3: Fallback to role-based matching (ONLY for matches without kickoff initialization)
        # Check if kickoff master list exists - if so, DON'T create new players
        has_master_list = any(p.team == team and p.label_confidence >= 0.9 for p in identities)

        if has_master_list:
            # Kickoff master list exists but we couldn't match - this shouldn't happen
            # Log warning and skip this detection
            print(f"[WARNING] Could not match track_id {track_id} to any master player at ({pitch_x:.2f}, {pitch_y:.2f})")
            # Return the closest player anyway (this shouldn't happen with the logic above)
            team_players = [p for p in identities if p.team == team and p.label_confidence >= 0.9]
            if team_players:
                return min(team_players, key=lambda p:
                    ((pitch_x - (p.typical_x_min + p.typical_x_max)/2)**2 +
                     (pitch_y - (p.typical_y_min + p.typical_y_max)/2)**2)**0.5)

        # No master list - use formation-based matching and creation
        formation_key = f"{video_id}_{team}"
        formation_name = self._formations.get(formation_key)
        if formation_name:
            formation = self._get_formation_template(formation_name)
            if formation:
                for role in formation.roles:
                    if role.contains_position(pitch_x, pitch_y):
                        for identity in identities:
                            if identity.team == team and identity.positional_role == role.role:
                                if track_id not in identity.track_id_history:
                                    identity.track_id_history.append(track_id)
                                identity.last_seen_frame = frame_number
                                identity.typical_x_min = min(identity.typical_x_min, pitch_x)
                                identity.typical_x_max = max(identity.typical_x_max, pitch_x)
                                identity.typical_y_min = min(identity.typical_y_min, pitch_y)
                                identity.typical_y_max = max(identity.typical_y_max, pitch_y)
                                return identity

                        # Create new player for this role (only if no kickoff initialization)
                        import uuid
                        new_identity = PlayerIdentity(
                            player_id=str(uuid.uuid4()),
                            team=team,
                            positional_role=role.role,
                            track_id_history=[track_id],
                            first_seen_frame=frame_number,
                            last_seen_frame=frame_number,
                            typical_x_min=pitch_x,
                            typical_x_max=pitch_x,
                            typical_y_min=pitch_y,
                            typical_y_max=pitch_y,
                            jersey_color_bgr=jersey_color,
                            label_confidence=0.7
                        )
                        identities.append(new_identity)
                        return new_identity

        # LAST RESORT: Create unassigned identity (shouldn't happen with kickoff initialization)
        import uuid
        new_identity = PlayerIdentity(
            player_id=str(uuid.uuid4()),
            team=team,
            positional_role=None,
            track_id_history=[track_id],
            first_seen_frame=frame_number,
            last_seen_frame=frame_number,
            typical_x_min=pitch_x,
            typical_x_max=pitch_x,
            typical_y_min=pitch_y,
            typical_y_max=pitch_y,
            jersey_color_bgr=jersey_color,
            label_confidence=0.3
        )
        identities.append(new_identity)
        return new_identity

    def set_formation(self, video_id: str, formation_name: str):
        """Set formation for a match."""
        self._formations[video_id] = formation_name

    def manual_label_player(
        self,
        video_id: str,
        track_id: int,
        jersey_number: Optional[int],
        positional_role: Optional[str],
        team: str,
        frame_number: int
    ) -> PlayerIdentity:
        """Manually label a player."""
        if video_id not in self._identities:
            self._identities[video_id] = []

        identities = self._identities[video_id]

        # Find existing identity with this track_id
        for identity in identities:
            if track_id in identity.track_id_history and identity.team == team:
                # Update with manual labels
                if jersey_number:
                    identity.jersey_number = jersey_number
                if positional_role:
                    identity.positional_role = positional_role
                identity.manually_labeled = True
                identity.label_confidence = 1.0
                return identity

        # Create new identity with manual labels
        import uuid
        new_identity = PlayerIdentity(
            player_id=str(uuid.uuid4()),
            team=team,
            jersey_number=jersey_number,
            positional_role=positional_role,
            track_id_history=[track_id],
            first_seen_frame=frame_number,
            last_seen_frame=frame_number,
            manually_labeled=True,
            label_confidence=1.0
        )
        identities.append(new_identity)
        return new_identity

    def get_all_identities(self, video_id: str) -> List[PlayerIdentity]:
        """Get all player identities for a match."""
        return self._identities.get(video_id, [])

    def get_identity_by_track_id(self, video_id: str, track_id: int) -> Optional[PlayerIdentity]:
        """Get player identity by current track_id."""
        identities = self._identities.get(video_id, [])
        for identity in identities:
            if track_id in identity.track_id_history:
                return identity
        return None

    def _get_formation_template(self, formation_name: str) -> Optional[FormationTemplate]:
        """Get formation template by name."""
        if formation_name == "4-4-2":
            return FormationTemplate.get_formation_442()
        elif formation_name == "4-3-3":
            return FormationTemplate.get_formation_433()
        # Add more formations as needed
        return None

    def record_ball_possession(self, video_id: str, player_id: str, is_new_touch: bool = False):
        """Record ball possession for a player."""
        identities = self._identities.get(video_id, [])
        for identity in identities:
            if identity.player_id == player_id:
                identity.frames_with_ball += 1
                if is_new_touch:
                    identity.ball_touches += 1
                break

    def initialize_from_kickoff(self, video_id: str, kickoff_detections: List[Dict], formation_name: str = "4-4-2"):
        """
        Initialize the 22 master player identities from kickoff frame when all players are visible.
        This creates the definitive roster that will be tracked throughout the match.

        Args:
            video_id: Match video ID
            kickoff_detections: List of player detections from kickoff frame (should have ~22 players)
            formation_name: Formation to use for positional role assignment
        """
        if video_id in self._identities and len(self._identities[video_id]) > 0:
            print(f"[KICKOFF] Players already initialized for {video_id}, skipping")
            return

        self._identities[video_id] = []

        # Separate by team
        home_detections = [d for d in kickoff_detections if d.get('team') == 'home']
        away_detections = [d for d in kickoff_detections if d.get('team') == 'away']

        print(f"[KICKOFF] Initial detection: {len(home_detections)} home, {len(away_detections)} away")

        # Get formation template
        formation = self._get_formation_template(formation_name)
        if not formation:
            print(f"[KICKOFF] Warning: Unknown formation {formation_name}, using default positions")

        # Standard shirt numbering by position
        role_to_number = {
            'GK': 1,
            'RB': 2, 'CB_R': 3, 'CB_L': 4, 'LB': 5,
            'CM_R': 6, 'RM': 7, 'CM_L': 8, 'ST_R': 9, 'ST_L': 10, 'LM': 11
        }

        # Filter to select best 11 players per team (remove referee/linesmen)
        def select_best_11(detections, team_name, formation):
            """
            Select 11 players that best match formation positions.
            Uses greedy assignment: each formation position picks its closest unassigned player.
            """
            if len(detections) <= 11:
                return detections

            # Get formation roles (11 positions)
            if not formation or len(formation.roles) != 11:
                # Fallback: select by X-coordinate spread to get diverse positions
                sorted_dets = sorted(detections, key=lambda d: d.get('pitch_x', 0))
                return sorted_dets[:11]

            selected = []
            remaining = list(detections)

            # For each formation position, find the closest unassigned player
            for formation_role in formation.roles:
                if not remaining:
                    break

                # Calculate role center
                role_x = (formation_role.zone_x_min + formation_role.zone_x_max) / 2
                role_y = (formation_role.zone_y_min + formation_role.zone_y_max) / 2

                # Find closest player to this role
                best_det = None
                best_distance = float('inf')

                for det in remaining:
                    # Normalize coordinates to 0-1
                    det_x = det.get('pitch_x', 0) / 100.0
                    det_y = det.get('pitch_y', 0) / 100.0

                    # Flip X coordinate for away team (they defend opposite goal)
                    if team_name == 'away':
                        det_x = 1.0 - det_x

                    # Calculate distance to role center
                    import math
                    distance = math.sqrt((det_x - role_x)**2 + (det_y - role_y)**2)

                    if distance < best_distance:
                        best_distance = distance
                        best_det = det

                if best_det:
                    # Tag this detection with its assigned role
                    best_det['assigned_role'] = formation_role.role
                    selected.append(best_det)
                    remaining.remove(best_det)

            return selected

        home_detections = select_best_11(home_detections, 'home', formation)
        away_detections = select_best_11(away_detections, 'away', formation)

        print(f"[KICKOFF] Filtered to 11 players per team: {len(home_detections)} home, {len(away_detections)} away")
        print(f"[KICKOFF] Home pitch_x range: {min(d['pitch_x'] for d in home_detections):.1f} - {max(d['pitch_x'] for d in home_detections):.1f}")
        print(f"[KICKOFF] Away pitch_x range: {min(d['pitch_x'] for d in away_detections):.1f} - {max(d['pitch_x'] for d in away_detections):.1f}")

        for team, detections in [('home', home_detections), ('away', away_detections)]:
            # Use pre-assigned roles from select_best_11()
            role_counts = {}
            for detection in detections:
                pitch_x = detection.get('pitch_x', 0) / 100.0  # Normalize to 0-1
                pitch_y = detection.get('pitch_y', 0) / 100.0

                # Flip coordinates for away team (for position bounds)
                if team == 'away':
                    pitch_x = 1.0 - pitch_x

                # Use the role assigned during selection (greedy matching)
                role = detection.get('assigned_role')

                # Track role assignments for debugging
                role_counts[role] = role_counts.get(role, 0) + 1

                # Create player identity
                import uuid
                track_id = detection.get('track_id', 0)
                jersey_number = role_to_number.get(role, 0) if role else 0

                player_identity = PlayerIdentity(
                    player_id=str(uuid.uuid4()),
                    team=team,
                    jersey_number=jersey_number if jersey_number > 0 else None,
                    positional_role=role,
                    track_id_history=[track_id],
                    first_seen_frame=detection.get('frame_number', 0),
                    last_seen_frame=detection.get('frame_number', 0),
                    typical_x_min=pitch_x,
                    typical_x_max=pitch_x,
                    typical_y_min=pitch_y,
                    typical_y_max=pitch_y,
                    jersey_color_bgr=detection.get('jersey_color'),
                    label_confidence=1.0  # High confidence - these are the definitive 22 players
                )
                self._identities[video_id].append(player_identity)

            # Log role distribution for this team
            print(f"[KICKOFF] {team.capitalize()} role distribution: {role_counts}")

        print(f"[KICKOFF] Initialized {len(self._identities[video_id])} master player identities")


# Global player identity database
player_identity_db = PlayerIdentityDatabase()
