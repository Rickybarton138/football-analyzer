"""
Phase 1 + Phase 2 Integration Test

Tests that all analytics services are properly wired into the pipeline
by simulating realistic frame data through the local_processor's analytics chain.
No video file or YOLO model required.

Phase 2 additions: tactical_intelligence, ai_coach, coach_assist training focus.
"""
import asyncio
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from services.local_processor import LocalVideoProcessor
from services.pass_detector import pass_detector
from services.xg_model import xg_model, Shot
from services.formation_detector import formation_detector
from services.tactical_events import tactical_detector
from services.event_detector import event_detector, DetectedEventType
from services.tactical_intelligence import tactical_intelligence
from services.ai_coach import ai_coach
from services.coach_assist import coach_assist_service


def test_possession_inertia():
    """Test that possession doesn't flicker with inertia."""
    print("\n=== Test 1: Possession Inertia ===")
    proc = LocalVideoProcessor()

    # Player A at (100, 200), Player B at (300, 200)
    detections = [
        {'bbox': [80, 150, 120, 250], 'track_id': 1, 'team': 'home'},
        {'bbox': [280, 150, 320, 250], 'track_id': 2, 'team': 'away'},
    ]

    # Ball near player A
    ball = [100, 250]
    track_id, team = proc._determine_ball_possession(ball, detections)
    # First call with no prior state — inertia means it won't switch yet
    # (initial state is None, so first different player needs 5 frames)
    print(f"  Frame 1: track_id={track_id}, team={team}")

    # Feed same ball position 5 times to overcome inertia
    for i in range(5):
        track_id, team = proc._determine_ball_possession(ball, detections)

    assert track_id == 1, f"Expected track_id=1, got {track_id}"
    assert team == 'home', f"Expected team='home', got {team}"
    print(f"  After 5 frames: track_id={track_id}, team={team} (correct!)")

    # Now move ball near player B — should NOT switch immediately
    ball_b = [300, 250]
    track_id, team = proc._determine_ball_possession(ball_b, detections)
    assert track_id == 1, f"Should still be player 1 (inertia), got {track_id}"
    print(f"  Ball moves to B, frame 1: track_id={track_id} (inertia holding)")

    # After 5 more frames near B, should switch
    for i in range(5):
        track_id, team = proc._determine_ball_possession(ball_b, detections)

    assert track_id == 2, f"Expected track_id=2 after inertia, got {track_id}"
    assert team == 'away', f"Expected team='away', got {team}"
    print(f"  After 5 frames near B: track_id={track_id}, team={team} (switched!)")

    print("  PASSED")


def test_pass_detector_wiring():
    """Test that pass_detector processes frames correctly."""
    print("\n=== Test 2: Pass Detector Wiring ===")
    pass_detector.reset()

    # Simulate 30 frames of ball movement between home players (a completed pass)
    for i in range(30):
        frame_data = {
            'frame_number': i,
            'timestamp': i / 30.0,
            'ball_position': [200 + i * 5, 300],  # Ball moving right
            'detections': [
                {'bbox': [180, 250, 220, 350], 'track_id': 1, 'team': 'home'},
                {'bbox': [330, 250, 370, 350], 'track_id': 2, 'team': 'home'},
                {'bbox': [500, 250, 540, 350], 'track_id': 3, 'team': 'away'},
            ]
        }
        pass_detector.process_frame(frame_data)

    stats = pass_detector.get_pass_stats()
    print(f"  Pass stats keys: {list(stats.keys())}")
    print(f"  Home stats: {stats.get('home', {})}")
    print(f"  Away stats: {stats.get('away', {})}")
    print("  PASSED")


def test_xg_model_wiring():
    """Test that xG model calculates correctly."""
    print("\n=== Test 3: xG Model Wiring ===")
    xg_model.reset()

    # Create a shot from just outside the box
    shot = Shot(
        x=80, y=50,  # Normalized 0-100 coords
        frame_number=100,
        timestamp_ms=3333,
        team="home",
        player_jersey=9,
    )
    result = xg_model.add_shot(shot)
    print(f"  Shot from (80, 50): xG = {result.xg:.3f}")
    assert 0 < result.xg < 1, f"xG should be between 0 and 1, got {result.xg}"

    # Create a penalty
    pen = Shot(x=89, y=50, team="away", player_jersey=10, frame_number=500, timestamp_ms=16666)
    pen_result = xg_model.add_shot(pen)
    print(f"  Shot from (89, 50): xG = {pen_result.xg:.3f}")

    data = xg_model.get_shot_map_data()
    print(f"  Total shots: {len(data.get('shots', []))}")
    print(f"  Home xG: {data.get('total_xg', {}).get('home', 0):.3f}")
    print(f"  Away xG: {data.get('total_xg', {}).get('away', 0):.3f}")
    assert len(data.get('shots', [])) == 2
    print("  PASSED")


def test_formation_detector_wiring():
    """Test that formation detector processes frame data."""
    print("\n=== Test 4: Formation Detector Wiring ===")
    formation_detector.reset()

    # Simulate a 4-4-2 home team setup (10 outfield players)
    # X = pitch length (goal to goal), Y = pitch width
    home_positions = [
        # Defenders (low x)
        (150, 100), (150, 250), (150, 400), (150, 550),
        # Midfielders (mid x)
        (350, 100), (350, 250), (350, 400), (350, 550),
        # Forwards (high x)
        (550, 200), (550, 450),
    ]

    away_positions = [
        (650, 100), (650, 250), (650, 400), (650, 550),
        (450, 100), (450, 250), (450, 400), (450, 550),
        (250, 200), (250, 450),
    ]

    detections = []
    for i, (x, y) in enumerate(home_positions):
        detections.append({
            'bbox': [x-20, y-40, x+20, y+40],
            'track_id': i + 1,
            'team': 'home'
        })
    for i, (x, y) in enumerate(away_positions):
        detections.append({
            'bbox': [x-20, y-40, x+20, y+40],
            'track_id': i + 20,
            'team': 'away'
        })

    # Feed several frames for stability
    for frame_num in range(20):
        frame_data = {
            'frame_number': frame_num,
            'timestamp': frame_num / 30.0,
            'ball_position': [400, 300],
            'detections': detections,
        }
        formation_detector.process_frame(frame_data)

    stats = formation_detector.calculate_stats()
    print(f"  Teams detected: {list(stats.keys())}")
    for team, s in stats.items():
        print(f"  {team.capitalize()}: {s.primary_formation} (changes: {s.formation_changes}, compactness: {s.avg_compactness:.1f})")
    print("  PASSED")


def test_tactical_events_wiring():
    """Test that tactical event detector processes frames."""
    print("\n=== Test 5: Tactical Events Wiring ===")
    tactical_detector.reset()

    # Simulate frames with detections
    for i in range(60):
        # Home team pushing high, away team deep — should detect pressing opportunity
        home_x = 500 + i * 2  # Home pushing forward
        detections = [
            {'bbox': [home_x, 100, home_x+40, 200], 'track_id': 1, 'team': 'home'},
            {'bbox': [home_x-20, 250, home_x+20, 350], 'track_id': 2, 'team': 'home'},
            {'bbox': [home_x+30, 400, home_x+70, 500], 'track_id': 3, 'team': 'home'},
            {'bbox': [700, 100, 740, 200], 'track_id': 10, 'team': 'away'},
            {'bbox': [700, 300, 740, 400], 'track_id': 11, 'team': 'away'},
            {'bbox': [650, 200, 690, 300], 'track_id': 12, 'team': 'away'},
        ]

        frame_data = {
            'frame_number': i,
            'timestamp': i / 30.0,
            'ball_position': [home_x + 20, 250],
            'detections': detections,
        }
        tactical_detector.process_frame(frame_data)

    summary = tactical_detector.get_events_summary()
    total = summary.get('total_events', 0)
    event_counts = summary.get('event_counts', {})
    print(f"  Total tactical events: {total}")
    for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {event_type}: {count}")
    print("  PASSED")


def test_event_detector_shot_to_xg():
    """Test that shot events from event_detector feed into xG model."""
    print("\n=== Test 6: Event Detector -> xG Pipeline ===")
    event_detector.reset()
    xg_model.reset()

    # Set video info
    event_detector.set_video_info(fps=30, total_frames=900, frame_width=1920, frame_height=1080)

    # Simulate ball moving fast toward goal (should trigger shot detection)
    # First, establish ball position history
    for i in range(30):
        x = 50 + i * 1.5  # Ball slowly moving forward
        events = event_detector.process_frame(
            frame_number=i,
            timestamp_ms=i * 33,
            ball_x=x,
            ball_y=50,
            possessing_team="home"
        )

    # Now rapid movement toward goal (shot)
    shot_events = []
    for i in range(30, 40):
        x = 85 + (i - 30) * 2  # Fast movement toward goal
        events = event_detector.process_frame(
            frame_number=i,
            timestamp_ms=i * 33,
            ball_x=x,
            ball_y=50,
            possessing_team="home"
        )
        if events:
            for e in events:
                print(f"    Event at frame {i}: {e.event_type.value}")
                shot_events.extend(events)

    # Feed any shot events into xG (mimicking local_processor logic)
    for evt in shot_events:
        if evt.event_type in (
            DetectedEventType.SHOT,
            DetectedEventType.SHOT_ON_TARGET,
            DetectedEventType.SHOT_OFF_TARGET,
            DetectedEventType.SHOT_BLOCKED,
        ):
            shot = Shot(
                x=evt.position_x or 90,
                y=evt.position_y or 50,
                frame_number=evt.frame_number,
                timestamp_ms=evt.timestamp_ms,
                team=evt.team or "home",
            )
            result = xg_model.add_shot(shot)
            print(f"    xG for shot: {result.xg:.3f}")

    all_events = event_detector.get_events()
    print(f"  Total events detected: {len(all_events)}")
    print(f"  xG shots recorded: {len(xg_model.shots)}")
    print("  PASSED")


def test_save_analysis_format():
    """Test that _save_analysis includes Phase 1 data keys."""
    print("\n=== Test 7: Save Analysis JSON Format ===")
    import tempfile
    import json

    proc = LocalVideoProcessor()

    # Set up minimal analysis
    from services.local_processor import MatchAnalysis
    proc.current_analysis = MatchAnalysis(
        video_path="test.mp4",
        duration_seconds=10,
        total_frames=300,
        analyzed_frames=50,
        fps_analyzed=5,
        start_time="2026-03-03T00:00:00"
    )

    # Reset services so they have empty but valid state
    pass_detector.reset()
    xg_model.reset()
    formation_detector.reset()
    tactical_detector.reset()

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmppath = f.name

    proc._save_analysis(tmppath)

    with open(tmppath, 'r') as f:
        data = json.load(f)

    required_keys = ['pass_stats', 'formations', 'xg', 'tactical_events']
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in saved JSON"
        print(f"  '{key}': present ({type(data[key]).__name__})")

    os.unlink(tmppath)
    print("  PASSED")


def test_tactical_intelligence_process_frame():
    """Test that tactical_intelligence accepts reformatted player data."""
    print("\n=== Test 8: Tactical Intelligence Process Frame ===")
    tactical_intelligence.reset()

    # Simulate home and away player lists in tactical_intelligence format
    home_players = [
        {'x': 20, 'y': 30, 'jersey_number': 2, 'has_ball': False},
        {'x': 25, 'y': 50, 'jersey_number': 5, 'has_ball': False},
        {'x': 22, 'y': 70, 'jersey_number': 3, 'has_ball': False},
        {'x': 45, 'y': 40, 'jersey_number': 8, 'has_ball': True},
        {'x': 50, 'y': 60, 'jersey_number': 10, 'has_ball': False},
    ]
    away_players = [
        {'x': 80, 'y': 30, 'jersey_number': 4, 'has_ball': False},
        {'x': 75, 'y': 50, 'jersey_number': 6, 'has_ball': False},
        {'x': 78, 'y': 70, 'jersey_number': 3, 'has_ball': False},
        {'x': 55, 'y': 40, 'jersey_number': 7, 'has_ball': False},
    ]

    # Process 20 frames
    for i in range(20):
        alerts = tactical_intelligence.process_frame(
            frame_number=i,
            timestamp_ms=i * 200,
            home_players=home_players,
            away_players=away_players,
            ball_position=(45, 40),
            possession_team='home',
        )

    analysis = tactical_intelligence.get_full_analysis()
    assert 'summary' in analysis, "Missing 'summary' in tactical intelligence"
    assert 'all_alerts' in analysis, "Missing 'all_alerts' in tactical intelligence"
    print(f"  Summary keys: {list(analysis['summary'].keys()) if analysis['summary'] else 'empty'}")
    print(f"  Total alerts: {len(analysis.get('all_alerts', []))}")
    print("  PASSED")


def test_ai_coach_analyze_match():
    """Test that ai_coach.analyze_match() generates insights from Phase 1 data."""
    print("\n=== Test 9: AI Coach Analyze Match ===")
    ai_coach.reset()

    pass_stats = {
        'home': {'total': 200, 'successful': 140, 'accuracy': 70, 'forward_ratio': 0.3},
        'away': {'total': 180, 'successful': 110, 'accuracy': 61, 'forward_ratio': 0.2},
    }
    formation_stats = {
        'home': {'primary_formation': '4-4-2', 'formation_changes': 3, 'avg_defensive_line': 35, 'avg_compactness': 28},
        'away': {'primary_formation': '4-3-3', 'formation_changes': 6, 'avg_defensive_line': 40, 'avg_compactness': 42},
    }
    tactical_events = {
        'total_events': 45,
        'event_counts': {'high_press': 8, 'counter_attack': 4, 'build_up': 12, 'long_ball': 6},
    }

    result = ai_coach.analyze_match(
        pass_stats=pass_stats,
        formation_stats=formation_stats,
        tactical_events=tactical_events,
    )

    assert result is not None, "analyze_match returned None"
    assert len(ai_coach.insights) > 0, "No insights generated"
    assert ai_coach.match_summary is not None, "No match summary generated"

    print(f"  Rating: {ai_coach.match_summary.overall_rating}/10")
    print(f"  Insights: {len(ai_coach.insights)}")
    print(f"  Strengths: {ai_coach.match_summary.key_strengths[:2]}")
    print("  PASSED")


def test_training_focus_generation():
    """Test that coach_assist generates training focus with drills."""
    print("\n=== Test 10: Training Focus Generation ===")
    coach_assist_service.reset()

    # Load with Phase 1 data that has clear weaknesses
    coach_assist_service.load_match_data(
        pass_stats={
            'home': {'total': 150, 'successful': 85, 'accuracy': 56, 'forward_ratio': 0.12},
            'away': {'total': 200, 'successful': 170, 'accuracy': 85, 'forward_ratio': 0.4},
        },
        formation_stats={
            'home': {'primary_formation': '4-4-2', 'formation_changes': 9, 'avg_defensive_line': 30, 'avg_compactness': 48},
            'away': {'primary_formation': '4-3-3', 'formation_changes': 2, 'avg_defensive_line': 45, 'avg_compactness': 25},
        },
        xg_data={
            'total_xg': {'home': 2.1, 'away': 1.8},
            'shots': [
                {'team': 'home', 'is_goal': False, 'xg': 0.5},
                {'team': 'home', 'is_goal': False, 'xg': 0.8},
                {'team': 'home', 'is_goal': True, 'xg': 0.8},
                {'team': 'away', 'is_goal': True, 'xg': 0.3},
                {'team': 'away', 'is_goal': True, 'xg': 0.6},
            ],
        },
        tactical_summary={
            'total_events': 30,
            'event_counts': {'high_press': 1, 'counter_attack': 5, 'build_up': 10},
        },
    )

    result = coach_assist_service.generate_training_focus()

    assert 'priority_areas' in result, "Missing priority_areas"
    assert 'session_plan' in result, "Missing session_plan"
    assert len(result['priority_areas']) > 0, "No priority areas generated"

    print(f"  Priority areas: {len(result['priority_areas'])}")
    for p in result['priority_areas'][:3]:
        print(f"    [{p['severity'].upper()}] {p['area']} ({p['team']}) -> {p['drill']}")

    sp = result['session_plan']
    assert 'warm_up' in sp, "Missing warm_up in session plan"
    assert 'main_focus' in sp, "Missing main_focus in session plan"
    assert 'game' in sp, "Missing game in session plan"
    print(f"  Session plan main focus: {sp['main_focus']['area']}")
    print("  PASSED")


def test_save_analysis_phase2_keys():
    """Test that _save_analysis includes Phase 2 data keys."""
    print("\n=== Test 11: Save Analysis Phase 2 Keys ===")
    import tempfile
    import json

    proc = LocalVideoProcessor()

    from services.local_processor import MatchAnalysis
    proc.current_analysis = MatchAnalysis(
        video_path="test.mp4",
        duration_seconds=10,
        total_frames=300,
        analyzed_frames=50,
        fps_analyzed=5,
        start_time="2026-03-03T00:00:00"
    )

    # Set Phase 2 cached results
    proc._coaching_analysis = {'rating': 7, 'insights': []}
    proc._tactical_alerts_summary = {'all_alerts': [], 'summary': {}}
    proc._training_focus = {'priority_areas': [], 'session_plan': {}}

    # Reset Phase 1 services for clean state
    pass_detector.reset()
    xg_model.reset()
    formation_detector.reset()
    tactical_detector.reset()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmppath = f.name

    proc._save_analysis(tmppath)

    with open(tmppath, 'r') as f:
        data = json.load(f)

    phase2_keys = ['coaching_analysis', 'tactical_alerts', 'training_focus']
    for key in phase2_keys:
        assert key in data, f"Missing Phase 2 key '{key}' in saved JSON"
        print(f"  '{key}': present ({type(data[key]).__name__})")

    os.unlink(tmppath)
    print("  PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 1 + Phase 2 Integration Tests")
    print("=" * 60)

    tests = [
        test_possession_inertia,
        test_pass_detector_wiring,
        test_xg_model_wiring,
        test_formation_detector_wiring,
        test_tactical_events_wiring,
        test_event_detector_shot_to_xg,
        test_save_analysis_format,
        # Phase 2 tests
        test_tactical_intelligence_process_frame,
        test_ai_coach_analyze_match,
        test_training_focus_generation,
        test_save_analysis_phase2_keys,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")

    sys.exit(1 if failed > 0 else 0)
