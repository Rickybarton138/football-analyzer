"""
Geometry Utilities

Helper functions for distance calculations, coordinate transformations,
and geometric operations.
"""
import numpy as np
from typing import List, Tuple, Optional
from models.schemas import Position, PixelPosition


def calculate_distance(pos1: Position, pos2: Position) -> float:
    """
    Calculate Euclidean distance between two pitch positions.

    Args:
        pos1: First position in meters
        pos2: Second position in meters

    Returns:
        Distance in meters
    """
    return np.sqrt((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2)


def calculate_speed(
    pos1: Position,
    pos2: Position,
    time_delta_ms: int
) -> float:
    """
    Calculate speed between two positions.

    Args:
        pos1: Starting position
        pos2: Ending position
        time_delta_ms: Time difference in milliseconds

    Returns:
        Speed in km/h
    """
    distance_m = calculate_distance(pos1, pos2)
    time_s = time_delta_ms / 1000
    if time_s == 0:
        return 0.0
    speed_ms = distance_m / time_s
    return speed_ms * 3.6  # Convert m/s to km/h


def calculate_angle(pos1: Position, pos2: Position) -> float:
    """
    Calculate angle from pos1 to pos2.

    Args:
        pos1: Origin position
        pos2: Target position

    Returns:
        Angle in radians (0 = right, pi/2 = up)
    """
    return np.arctan2(pos2.y - pos1.y, pos2.x - pos1.x)


def rotate_point(
    point: Position,
    center: Position,
    angle_rad: float
) -> Position:
    """
    Rotate a point around a center.

    Args:
        point: Point to rotate
        center: Center of rotation
        angle_rad: Angle in radians

    Returns:
        Rotated position
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate to origin
    tx = point.x - center.x
    ty = point.y - center.y

    # Rotate
    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a

    # Translate back
    return Position(x=rx + center.x, y=ry + center.y)


def point_in_rectangle(
    point: Position,
    rect_min: Position,
    rect_max: Position
) -> bool:
    """
    Check if point is inside rectangle.

    Args:
        point: Point to check
        rect_min: Bottom-left corner of rectangle
        rect_max: Top-right corner of rectangle

    Returns:
        True if point is inside rectangle
    """
    return (
        rect_min.x <= point.x <= rect_max.x and
        rect_min.y <= point.y <= rect_max.y
    )


def point_in_circle(
    point: Position,
    center: Position,
    radius: float
) -> bool:
    """
    Check if point is inside circle.

    Args:
        point: Point to check
        center: Center of circle
        radius: Radius of circle

    Returns:
        True if point is inside circle
    """
    return calculate_distance(point, center) <= radius


def line_intersection(
    p1: Position,
    p2: Position,
    p3: Position,
    p4: Position
) -> Optional[Position]:
    """
    Find intersection point of two line segments.

    Args:
        p1, p2: First line segment endpoints
        p3, p4: Second line segment endpoints

    Returns:
        Intersection point or None if no intersection
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return Position(x=x, y=y)

    return None


def point_to_line_distance(
    point: Position,
    line_start: Position,
    line_end: Position
) -> float:
    """
    Calculate shortest distance from point to line segment.

    Args:
        point: The point
        line_start: Start of line segment
        line_end: End of line segment

    Returns:
        Distance in meters
    """
    # Vector from start to end
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y

    # Length squared
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        # Line is a point
        return calculate_distance(point, line_start)

    # Project point onto line
    t = max(0, min(1, (
        (point.x - line_start.x) * dx +
        (point.y - line_start.y) * dy
    ) / length_sq))

    # Find nearest point on line
    nearest = Position(
        x=line_start.x + t * dx,
        y=line_start.y + t * dy
    )

    return calculate_distance(point, nearest)


def calculate_convex_hull(positions: List[Position]) -> List[Position]:
    """
    Calculate convex hull of a set of positions using Graham scan.

    Args:
        positions: List of positions

    Returns:
        List of positions forming the convex hull
    """
    if len(positions) < 3:
        return positions

    # Find lowest point
    points = sorted(positions, key=lambda p: (p.y, p.x))
    start = points[0]

    # Sort by angle from start
    def angle_key(p):
        if p == start:
            return -np.inf
        return np.arctan2(p.y - start.y, p.x - start.x)

    sorted_points = sorted(points, key=angle_key)

    # Build hull
    hull = []
    for p in sorted_points:
        while len(hull) >= 2:
            # Check if we turn left
            o = hull[-2]
            a = hull[-1]
            cross = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    return hull


def calculate_polygon_area(vertices: List[Position]) -> float:
    """
    Calculate area of a polygon using shoelace formula.

    Args:
        vertices: List of polygon vertices in order

    Returns:
        Area in square meters
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].x * vertices[j].y
        area -= vertices[j].x * vertices[i].y

    return abs(area) / 2.0


def calculate_centroid(positions: List[Position]) -> Position:
    """
    Calculate centroid of a set of positions.

    Args:
        positions: List of positions

    Returns:
        Centroid position
    """
    if not positions:
        return Position(x=0, y=0)

    x_sum = sum(p.x for p in positions)
    y_sum = sum(p.y for p in positions)
    n = len(positions)

    return Position(x=x_sum / n, y=y_sum / n)


def interpolate_position(
    pos1: Position,
    pos2: Position,
    t: float
) -> Position:
    """
    Linear interpolation between two positions.

    Args:
        pos1: Start position
        pos2: End position
        t: Interpolation factor (0 = pos1, 1 = pos2)

    Returns:
        Interpolated position
    """
    return Position(
        x=pos1.x + t * (pos2.x - pos1.x),
        y=pos1.y + t * (pos2.y - pos1.y)
    )


def smooth_trajectory(
    positions: List[Position],
    window_size: int = 3
) -> List[Position]:
    """
    Smooth a trajectory using moving average.

    Args:
        positions: List of positions
        window_size: Size of smoothing window

    Returns:
        Smoothed positions
    """
    if len(positions) < window_size:
        return positions

    smoothed = []
    half_window = window_size // 2

    for i in range(len(positions)):
        start = max(0, i - half_window)
        end = min(len(positions), i + half_window + 1)
        window = positions[start:end]

        avg_x = sum(p.x for p in window) / len(window)
        avg_y = sum(p.y for p in window) / len(window)
        smoothed.append(Position(x=avg_x, y=avg_y))

    return smoothed


def calculate_voronoi_area(
    player_pos: Position,
    all_positions: List[Position],
    pitch_bounds: Tuple[float, float, float, float]
) -> float:
    """
    Calculate approximate Voronoi cell area for a player.
    (Simplified version - proper Voronoi would need scipy)

    Args:
        player_pos: Player's position
        all_positions: All player positions
        pitch_bounds: (min_x, min_y, max_x, max_y)

    Returns:
        Approximate area in square meters
    """
    min_x, min_y, max_x, max_y = pitch_bounds

    # Sample grid points
    grid_resolution = 5  # meters
    x_points = np.arange(min_x, max_x, grid_resolution)
    y_points = np.arange(min_y, max_y, grid_resolution)

    cell_count = 0

    for x in x_points:
        for y in y_points:
            test_pos = Position(x=x, y=y)

            # Find nearest player
            min_dist = float('inf')
            nearest = None
            for pos in all_positions:
                dist = calculate_distance(test_pos, pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = pos

            # Check if this player owns this cell
            if nearest and (
                abs(nearest.x - player_pos.x) < 0.01 and
                abs(nearest.y - player_pos.y) < 0.01
            ):
                cell_count += 1

    return cell_count * (grid_resolution ** 2)
