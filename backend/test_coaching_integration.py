"""
Test script to validate coaching intelligence integration across all platform tabs.
Run this after the backend server is started to verify all endpoints work correctly.
"""
import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import httpx

BASE_URL = "http://127.0.0.1:8000"

async def test_endpoint(client: httpx.AsyncClient, method: str, endpoint: str, data: dict = None, name: str = None):
    """Test a single endpoint and return result."""
    url = f"{BASE_URL}{endpoint}"
    name = name or endpoint
    try:
        if method == "GET":
            response = await client.get(url, timeout=30.0)
        elif method == "POST":
            response = await client.post(url, json=data, timeout=30.0)
        else:
            return {"name": name, "status": "error", "message": f"Unknown method: {method}"}

        if response.status_code == 200:
            result = response.json()
            # Check if result has meaningful data
            if isinstance(result, dict):
                has_data = any(v for v in result.values() if v)
            elif isinstance(result, list):
                has_data = len(result) > 0
            else:
                has_data = bool(result)

            return {
                "name": name,
                "status": "success" if has_data else "empty",
                "data_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            }
        else:
            return {"name": name, "status": "error", "code": response.status_code, "message": response.text[:200]}
    except Exception as e:
        return {"name": name, "status": "exception", "message": str(e)}


async def run_all_tests():
    """Run all integration tests for coaching intelligence."""
    print("=" * 70)
    print("FOOTBALL ANALYZER - COACHING INTELLIGENCE INTEGRATION TEST")
    print("=" * 70)

    async with httpx.AsyncClient() as client:
        results = {"tabs": {}}

        # ==================== TAB 1: MATCH VIDEO ====================
        print("\n📹 TAB 1: MATCH VIDEO")
        print("-" * 40)

        video_tests = [
            ("GET", "/api/video/full-analysis", None, "Full Analysis Data"),
            ("GET", "/api/video/process-status", None, "Processing Status"),
        ]

        tab1_results = []
        for method, endpoint, data, name in video_tests:
            result = await test_endpoint(client, method, endpoint, data, name)
            tab1_results.append(result)
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "empty" else "❌"
            print(f"  {status_icon} {name}: {result['status']}")

        results["tabs"]["match_video"] = tab1_results

        # ==================== TAB 2: MATCH ANALYSIS ====================
        print("\n📊 TAB 2: MATCH ANALYSIS")
        print("-" * 40)

        analysis_tests = [
            ("GET", "/api/video/full-analysis", None, "Match Stats & Frames"),
            ("GET", "/api/analytics/tactical-events", None, "Tactical Events"),
            ("GET", "/api/analytics/match-stats", None, "Match Statistics"),
        ]

        tab2_results = []
        for method, endpoint, data, name in analysis_tests:
            result = await test_endpoint(client, method, endpoint, data, name)
            tab2_results.append(result)
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "empty" else "❌"
            print(f"  {status_icon} {name}: {result['status']}")

        results["tabs"]["match_analysis"] = tab2_results

        # ==================== TAB 3: AI COACH ====================
        print("\n🧠 TAB 3: AI COACH (Coaching Intelligence)")
        print("-" * 40)

        ai_coach_tests = [
            ("GET", "/api/ai-coach/analysis", None, "AI Coach Analysis"),
            ("POST", "/api/ai-coach/chat", {"question": "How was our possession?"}, "AI Coach Chat"),
            ("GET", "/api/expert-coach/status", None, "Expert Coach Status"),
        ]

        tab3_results = []
        for method, endpoint, data, name in ai_coach_tests:
            result = await test_endpoint(client, method, endpoint, data, name)
            tab3_results.append(result)
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "empty" else "❌"
            print(f"  {status_icon} {name}: {result['status']}")

        results["tabs"]["ai_coach"] = tab3_results

        # ==================== TAB 4: TACTICAL EVENTS ====================
        print("\n⚡ TAB 4: TACTICAL EVENTS")
        print("-" * 40)

        tactical_tests = [
            ("GET", "/api/analytics/tactical-events", None, "Tactical Events Timeline"),
            ("GET", "/api/analytics/momentum", None, "Momentum Analysis"),
        ]

        tab4_results = []
        for method, endpoint, data, name in tactical_tests:
            result = await test_endpoint(client, method, endpoint, data, name)
            tab4_results.append(result)
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "empty" else "❌"
            print(f"  {status_icon} {name}: {result['status']}")

        results["tabs"]["tactical_events"] = tab4_results

        # ==================== TAB 5: PLAYERS ====================
        print("\n👤 TAB 5: PLAYERS")
        print("-" * 40)

        player_tests = [
            ("GET", "/api/player/list", None, "Player List"),
        ]

        tab5_results = []
        for method, endpoint, data, name in player_tests:
            result = await test_endpoint(client, method, endpoint, data, name)
            tab5_results.append(result)
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "empty" else "❌"
            print(f"  {status_icon} {name}: {result['status']}")

        results["tabs"]["players"] = tab5_results

        # ==================== COACHING INTELLIGENCE SERVICES ====================
        print("\n🎓 COACHING INTELLIGENCE SERVICES")
        print("-" * 40)

        coaching_tests = [
            ("GET", "/api/coaching-intelligence/status", None, "Coaching Intelligence Status"),
            ("GET", "/api/coaching-intelligence/full-report", None, "Full Coaching Report"),
            ("GET", "/api/coaching-intelligence/tab/aicoach", None, "AI Coach Tab Data"),
            ("GET", "/api/coaching-intelligence/tab/tactical", None, "Tactical Tab Data"),
            ("GET", "/api/coaching-intelligence/tab/players", None, "Players Tab Data"),
            ("GET", "/api/team-profile/list", None, "Team Profiles"),
            ("POST", "/api/tactical/detect-formation", {
                "players": [
                    {"x": 10, "y": 50, "team": "home"},
                    {"x": 30, "y": 30, "team": "home"},
                    {"x": 30, "y": 50, "team": "home"},
                    {"x": 30, "y": 70, "team": "home"},
                    {"x": 50, "y": 20, "team": "home"},
                    {"x": 50, "y": 40, "team": "home"},
                    {"x": 50, "y": 60, "team": "home"},
                    {"x": 50, "y": 80, "team": "home"},
                    {"x": 70, "y": 35, "team": "home"},
                    {"x": 70, "y": 65, "team": "home"},
                    {"x": 85, "y": 50, "team": "home"},
                ]
            }, "Formation Detection"),
            ("GET", "/api/ml-export/stats", None, "ML Export Stats"),
        ]

        coaching_results = []
        for method, endpoint, data, name in coaching_tests:
            result = await test_endpoint(client, method, endpoint, data, name)
            coaching_results.append(result)
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "empty" else "❌"
            print(f"  {status_icon} {name}: {result['status']}")

        results["coaching_services"] = coaching_results

        # ==================== SUMMARY ====================
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total_tests = 0
        successful = 0
        empty = 0
        failed = 0

        for tab_name, tab_results in results["tabs"].items():
            for r in tab_results:
                total_tests += 1
                if r["status"] == "success":
                    successful += 1
                elif r["status"] == "empty":
                    empty += 1
                else:
                    failed += 1

        for r in results.get("coaching_services", []):
            total_tests += 1
            if r["status"] == "success":
                successful += 1
            elif r["status"] == "empty":
                empty += 1
            else:
                failed += 1

        print(f"\nTotal Tests: {total_tests}")
        print(f"  ✅ Successful: {successful}")
        print(f"  ⚠️ Empty/No Data: {empty}")
        print(f"  ❌ Failed: {failed}")

        if failed == 0 and successful > 0:
            print("\n🎉 All endpoints are working! The coaching intelligence is ready to populate all tabs.")
        elif failed > 0:
            print(f"\n⚠️ Some endpoints failed. Check the backend logs for errors.")

        return results


if __name__ == "__main__":
    print("Starting integration tests...")
    print("Make sure the backend server is running on http://127.0.0.1:8000")
    print()

    results = asyncio.run(run_all_tests())
