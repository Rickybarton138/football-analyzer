"""
Fix detection file by applying spatial deduplication and proper team classification.
This script processes the raw detection file to fix player counting issues.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def filter_grass_colors(colors):
    """Filter out grass, brown, and low-saturation colors - only keep clear jersey colors."""
    filtered = []
    for color in colors:
        b, g, r = color[0], color[1], color[2]

        # Calculate saturation
        max_ch = max(r, g, b)
        min_ch = min(r, g, b)
        saturation = (max_ch - min_ch) / max(max_ch, 1) if max_ch > 0 else 0

        # Only accept colors with clear characteristics:

        # Red jerseys (R clearly dominant)
        is_red = (r > max(g, b) * 1.4 and r > 70)

        # Blue jerseys (B clearly dominant)
        is_blue = (b > max(r, g) * 1.3 and b > 70)

        # White jerseys (all high, low saturation OK)
        is_white = (min_ch > 160 and max_ch > 200)

        # Black/dark jerseys (all low, but uniform)
        is_black = (max_ch < 60 and saturation < 0.3)

        # Yellow jerseys (R and G high, B low)
        is_yellow = (min(r, g) > 140 and b < 100)

        # Green jerseys (G dominant, but MUCH higher than grass)
        is_green_jersey = (g > max(r, b) * 1.5 and g > 120 and saturation > 0.4)

        # Orange jerseys (R high, G medium, B low)
        is_orange = (r > 140 and 70 < g < 150 and b < 80)

        # Purple/magenta jerseys (R and B high, G low)
        is_purple = (min(r, b) > 100 and g < min(r, b) * 0.7)

        if is_red or is_blue or is_white or is_black or is_yellow or is_green_jersey or is_orange or is_purple:
            filtered.append(color)

    return filtered

def detect_team_colors(detections, sample_frames=100):
    """Detect team colors from sample frames, excluding goalkeepers."""
    all_colors = []
    frame_width = 1920  # Assume standard HD

    for det in detections[:sample_frames]:
        for player in det.get('players', []):
            jersey_color = player.get('jersey_color')
            if jersey_color:
                # Skip if looks like goalkeeper (near edges)
                bbox = player.get('bbox', {})
                if isinstance(bbox, dict):
                    x1, x2 = bbox.get('x1', 0), bbox.get('x2', 0)
                else:
                    x1, x2 = bbox[0] if len(bbox) > 0 else 0, bbox[2] if len(bbox) > 2 else 0

                center_x = (x1 + x2) / 2
                near_left = center_x < frame_width * 0.15
                near_right = center_x > frame_width * 0.85

                # Skip goalkeepers
                if near_left or near_right:
                    continue

                all_colors.append(jersey_color)

    # Filter colors
    filtered_colors = filter_grass_colors(all_colors)

    if len(filtered_colors) < 20:
        print(f"Warning: Only {len(filtered_colors)} valid colors found")
        return None, None

    # Cluster into 2 teams
    colors_array = np.array(filtered_colors)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(colors_array)

    # Determine which cluster has more members - that's likely the dominant team
    labels = kmeans.labels_
    counts = np.bincount(labels)

    # Sort by count to get larger cluster first
    sorted_indices = np.argsort(-counts)

    home_color = kmeans.cluster_centers_[sorted_indices[0]]
    away_color = kmeans.cluster_centers_[sorted_indices[1]]

    return home_color, away_color

def _bgr_to_lab(bgr_color):
    """Convert a single BGR color to CIELAB."""
    import cv2
    bgr_pixel = np.array(bgr_color, dtype=np.uint8).reshape(1, 1, 3)
    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
    return lab_pixel.reshape(3).astype(float)

def classify_team(jersey_color, home_color, away_color):
    """Classify player team based on jersey color using CIELAB distance."""
    if jersey_color is None or home_color is None or away_color is None:
        return 'unknown'

    color_lab = _bgr_to_lab(jersey_color)
    home_lab = _bgr_to_lab(home_color)
    away_lab = _bgr_to_lab(away_color)

    home_dist = np.linalg.norm(color_lab - home_lab)
    away_dist = np.linalg.norm(color_lab - away_lab)

    min_dist = min(home_dist, away_dist)
    max_dist = max(home_dist, away_dist)

    # Too far from both teams in LAB space
    if min_dist > 40:
        return 'unknown'

    # Colors too ambiguous between teams
    if max_dist > 0 and min_dist / max_dist > 0.85:
        return 'unknown'

    return 'home' if home_dist < away_dist else 'away'

def deduplicate_frame(players, grid_size=60, min_confidence=0.45):
    """Apply spatial deduplication and confidence filtering."""
    spatial_grid = {}

    for player in players:
        # Get confidence
        bbox = player.get('bbox', {})
        if isinstance(bbox, dict):
            confidence = bbox.get('confidence', 0)
            x1, y1, x2, y2 = bbox.get('x1', 0), bbox.get('y1', 0), bbox.get('x2', 0), bbox.get('y2', 0)
        else:
            confidence = 0.5
            x1, y1, x2, y2 = bbox if len(bbox) == 4 else [0, 0, 0, 0]

        # Skip low confidence
        if confidence < min_confidence:
            continue

        # Calculate grid cell
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        grid_x = int(center_x / grid_size)
        grid_y = int(center_y / grid_size)
        grid_key = (grid_x, grid_y)

        # Keep highest confidence in each grid cell
        if grid_key not in spatial_grid or confidence > spatial_grid[grid_key].get('confidence', 0):
            player['confidence'] = confidence
            player['center_x'] = center_x
            player['center_y'] = center_y
            spatial_grid[grid_key] = player

    return list(spatial_grid.values())

def fix_detection_file(input_path, output_path, grid_size=60, min_confidence=0.45):
    """Fix detection file with proper deduplication and team classification."""
    print(f"Loading detection file: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    detections = data.get('detections', [])
    print(f"Total frames: {len(detections)}")

    # Detect team colors
    print("Detecting team colors...")
    home_color, away_color = detect_team_colors(detections, sample_frames=100)

    if home_color is not None and away_color is not None:
        print(f"Team colors detected:")
        print(f"  Home (BGR): {home_color.astype(int).tolist()} -> RGB: {[int(home_color[2]), int(home_color[1]), int(home_color[0])]}")
        print(f"  Away (BGR): {away_color.astype(int).tolist()} -> RGB: {[int(away_color[2]), int(away_color[1]), int(away_color[0])]}")
    else:
        print("WARNING: Could not detect team colors, using existing classifications")

    # Process each frame
    print(f"Processing frames with grid_size={grid_size}, min_confidence={min_confidence}...")

    total_players_before = 0
    total_players_after = 0
    home_counts = []
    away_counts = []

    for i, det in enumerate(detections):
        players = det.get('players', [])
        total_players_before += len(players)

        # Deduplicate
        deduped_players = deduplicate_frame(players, grid_size, min_confidence)

        # Re-classify teams
        home_count = 0
        away_count = 0
        unknown_count = 0

        for player in deduped_players:
            if home_color is not None and away_color is not None:
                jersey_color = player.get('jersey_color')
                player['team'] = classify_team(jersey_color, home_color, away_color)

            team = player.get('team', 'unknown')
            if team == 'home':
                home_count += 1
            elif team == 'away':
                away_count += 1
            else:
                unknown_count += 1

        # Instead of force-balancing teams (which corrupts color-based classification),
        # mark low-confidence excess players as unknown
        home_players = [p for p in deduped_players if p.get('team') == 'home']
        away_players = [p for p in deduped_players if p.get('team') == 'away']
        unknown_players = [p for p in deduped_players if p.get('team') == 'unknown']

        # Sort by confidence
        home_players.sort(key=lambda p: p.get('confidence', 0), reverse=True)
        away_players.sort(key=lambda p: p.get('confidence', 0), reverse=True)

        # Cap each team at max_per_team, but mark excess as unknown instead of
        # force-reassigning to the opposite team
        max_per_team = 12

        if len(home_players) > max_per_team:
            excess = home_players[max_per_team:]
            for p in excess:
                p['team'] = 'unknown'
            unknown_players.extend(excess)
            home_players = home_players[:max_per_team]

        if len(away_players) > max_per_team:
            excess = away_players[max_per_team:]
            for p in excess:
                p['team'] = 'unknown'
            unknown_players.extend(excess)
            away_players = away_players[:max_per_team]

        home_count = len(home_players)
        away_count = len(away_players)

        deduped_players = home_players + away_players + unknown_players

        det['players'] = deduped_players
        total_players_after += len(deduped_players)
        home_counts.append(home_count)
        away_counts.append(away_count)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(detections)} frames...")

    # Save fixed file
    print(f"\nSaving fixed detection file to: {output_path}")

    # Add metadata
    if 'metadata' not in data:
        data['metadata'] = {}

    data['metadata']['fixed'] = True
    data['metadata']['grid_size'] = grid_size
    data['metadata']['min_confidence'] = min_confidence
    data['metadata']['team_colors'] = {
        'home_bgr': home_color.astype(int).tolist() if home_color is not None else None,
        'away_bgr': away_color.astype(int).tolist() if away_color is not None else None
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

    # Print statistics
    avg_home = sum(home_counts) / len(home_counts)
    avg_away = sum(away_counts) / len(away_counts)

    print("\n=== Results ===")
    print(f"Total detections before: {total_players_before}")
    print(f"Total detections after:  {total_players_after}")
    print(f"Reduction: {100 * (1 - total_players_after/total_players_before):.1f}%")
    print(f"\nAverage players per frame:")
    print(f"  Home: {avg_home:.1f} (min: {min(home_counts)}, max: {max(home_counts)})")
    print(f"  Away: {avg_away:.1f} (min: {min(away_counts)}, max: {max(away_counts)})")
    print(f"  Total: {avg_home + avg_away:.1f}")
    print(f"\nFixed detection file saved successfully!")

if __name__ == "__main__":
    # Find most recent detection file
    frames_dir = Path("c:/Users/info/football-analyzer/data/frames")
    detection_files = list(frames_dir.glob("*_detections.json"))

    if not detection_files:
        print("ERROR: No detection files found")
        sys.exit(1)

    # Get most recent
    input_file = max(detection_files, key=lambda p: p.stat().st_mtime)
    output_file = input_file  # Overwrite the original

    # Backup original
    backup_file = input_file.parent / f"{input_file.stem}_backup.json"
    print(f"Creating backup: {backup_file}")
    import shutil
    shutil.copy(input_file, backup_file)

    # Fix the file
    fix_detection_file(
        input_path=str(input_file),
        output_path=str(output_file),
        grid_size=60,          # Spatial deduplication grid size
        min_confidence=0.45    # Minimum confidence threshold
    )
