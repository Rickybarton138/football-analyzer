"""
Manager Analysis Service

Analyses the manager's tactical decisions from match data and generates
a Manager Development Plan (MDP). Aggregates data from:
- Formation detector (formation choices, changes, shape)
- Tactical intelligence (pressing, transitions, alerts)
- Event detector (set pieces, possession changes)
"""
import logging
from dataclasses import asdict
from typing import Dict, List, Optional
from datetime import datetime

from services.ai_coaching_engine import (
    ai_coaching_engine, ManagerDevelopmentPlan, DevelopmentPriority
)

logger = logging.getLogger(__name__)


class ManagerAnalysisService:
    """Analyses manager tactical performance and generates MDPs."""

    def __init__(self):
        self._latest_mdp: Optional[ManagerDevelopmentPlan] = None
        self._latest_narrative: Optional[str] = None

    def gather_tactical_data(
        self,
        formation_stats: Optional[Dict] = None,
        tactical_summary: Optional[Dict] = None,
        pass_stats: Optional[Dict] = None,
        xg_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Gather and structure all tactical data for MDP generation.

        Returns a structured dict suitable for Claude analysis.
        """
        data: Dict = {"gathered_at": datetime.now().isoformat()}

        # Formation analysis
        if formation_stats:
            for team in ['home', 'away']:
                tf = formation_stats.get(team, {})
                if tf:
                    data[f"{team}_formation"] = {
                        "primary": tf.get('primary_formation', 'unknown'),
                        "changes": tf.get('formation_changes', 0),
                        "avg_defensive_line": tf.get('avg_defensive_line', 0),
                        "avg_compactness": tf.get('avg_compactness', 0),
                        "avg_width": tf.get('avg_width', 0),
                        "avg_depth": tf.get('avg_depth', 0),
                    }

        # Tactical events
        if tactical_summary:
            event_counts = tactical_summary.get('event_counts', {})
            data["tactical_events"] = {
                "total": tactical_summary.get('total_events', 0),
                "high_presses": event_counts.get('high_press', 0)
                                + event_counts.get('pressing_trigger', 0),
                "press_success_est": self._estimate_press_success(event_counts),
                "counter_attacks": event_counts.get('counter_attack', 0),
                "counter_attacks_conceded": event_counts.get('counter_attack_conceded', 0),
                "transition_dangers": event_counts.get('transition_danger', 0),
                "defensive_gaps": event_counts.get('defensive_gap', 0),
                "overloads_created": event_counts.get('overload_opportunity', 0),
                "space_behind_exploited": event_counts.get('space_behind_defense', 0),
                "formation_shifts": event_counts.get('formation_shift', 0),
            }

        # Pass analysis (team-level)
        if pass_stats:
            for team in ['home', 'away']:
                tp = pass_stats.get(team, {})
                if tp:
                    data[f"{team}_passing"] = {
                        "total": tp.get('total', 0),
                        "accuracy": tp.get('accuracy', 0),
                        "forward_ratio": tp.get('forward_ratio', 0),
                        "progressive": tp.get('progressive', 0),
                    }

        # xG
        if xg_data:
            total_xg = xg_data.get('total_xg', {})
            shots = xg_data.get('shots', [])
            data["xg_analysis"] = {
                "home_xg": total_xg.get('home', 0),
                "away_xg": total_xg.get('away', 0),
                "total_shots": len(shots),
                "home_shots": len([s for s in shots if s.get('team') == 'home']),
                "away_shots": len([s for s in shots if s.get('team') == 'away']),
            }

        return data

    def _estimate_press_success(self, event_counts: Dict) -> float:
        """Estimate pressing success rate from event counts."""
        total_presses = (
            event_counts.get('high_press', 0)
            + event_counts.get('pressing_trigger', 0)
        )
        if total_presses == 0:
            return 0.0
        # Rough estimate: counter_press events suggest successful recoveries
        successful = event_counts.get('counter_press', 0)
        return round(successful / total_presses * 100, 1) if total_presses > 0 else 0.0

    def rule_based_assessment(self, tactical_data: Dict) -> Dict:
        """Generate rule-based tactical assessment (no AI)."""
        assessment = {
            "formation_management": {"score": 6.0, "notes": []},
            "pressing_strategy": {"score": 6.0, "notes": []},
            "transition_management": {"score": 6.0, "notes": []},
            "strengths": [],
            "weaknesses": [],
            "priorities": [],
        }

        # Formation assessment
        home_f = tactical_data.get("home_formation", {})
        if home_f:
            changes = home_f.get("changes", 0)
            if changes <= 2:
                assessment["formation_management"]["score"] = 7.5
                assessment["strengths"].append(
                    f"Stable formation — only {changes} change(s) throughout match"
                )
            elif changes > 5:
                assessment["formation_management"]["score"] = 5.0
                assessment["weaknesses"].append(
                    f"Formation instability — {changes} formation changes detected"
                )
                assessment["priorities"].append({
                    "level": "HIGH",
                    "area": "Formation stability",
                    "recommendation": "Shadow play 11v0, freeze & check distances",
                })

            compactness = home_f.get("avg_compactness", 0)
            if compactness > 40:
                assessment["weaknesses"].append(
                    f"Poor compactness ({compactness:.0f}m avg)"
                )

        # Pressing assessment
        events = tactical_data.get("tactical_events", {})
        if events:
            presses = events.get("high_presses", 0)
            if presses >= 10:
                assessment["pressing_strategy"]["score"] = 7.0
                assessment["strengths"].append(
                    f"Active pressing — {presses} high presses"
                )
            elif presses < 5:
                assessment["pressing_strategy"]["score"] = 5.0
                assessment["weaknesses"].append(
                    f"Low pressing intensity — only {presses} high presses"
                )
                assessment["priorities"].append({
                    "level": "MEDIUM",
                    "area": "Pressing intensity",
                    "recommendation": "4v4+4 pressing rehearsal with 6-second rule",
                })

            # Transition danger
            transitions = events.get("transition_dangers", 0)
            if transitions > 3:
                assessment["transition_management"]["score"] = 5.0
                assessment["weaknesses"].append(
                    f"Transition vulnerability — {transitions} danger moments"
                )
                assessment["priorities"].append({
                    "level": "HIGH",
                    "area": "Counter-press / transition defence",
                    "recommendation": "5v5+GKs transition game with 5-second counter-press rule",
                })

        return assessment

    async def generate_mdp(
        self,
        formation_stats: Optional[Dict] = None,
        tactical_summary: Optional[Dict] = None,
        pass_stats: Optional[Dict] = None,
        xg_data: Optional[Dict] = None,
    ) -> ManagerDevelopmentPlan:
        """Generate full MDP — uses Claude if available, rule-based fallback."""
        tactical_data = self.gather_tactical_data(
            formation_stats, tactical_summary, pass_stats, xg_data
        )
        formation_data = {
            "home": tactical_data.get("home_formation", {}),
            "away": tactical_data.get("away_formation", {}),
        }
        event_summary = tactical_data.get("tactical_events", {})

        mdp = await ai_coaching_engine.generate_manager_mdp(
            tactical_data=tactical_data,
            formation_data=formation_data,
            event_summary=event_summary,
        )
        self._latest_mdp = mdp
        return mdp

    async def generate_narrative(
        self,
        tactical_summary: Optional[Dict] = None,
    ) -> str:
        """Generate tactical narrative from intelligence data."""
        intelligence = tactical_summary or {}

        # Try tactical intelligence service
        try:
            from services.tactical_intelligence import tactical_intelligence_service
            full = tactical_intelligence_service.get_full_analysis()
            if full:
                intelligence = full
        except Exception:
            pass

        narrative = await ai_coaching_engine.generate_tactical_narrative(intelligence)
        self._latest_narrative = narrative
        return narrative

    @property
    def latest_mdp(self) -> Optional[ManagerDevelopmentPlan]:
        return self._latest_mdp

    @property
    def latest_narrative(self) -> Optional[str]:
        return self._latest_narrative


# Global instance
manager_analysis_service = ManagerAnalysisService()
