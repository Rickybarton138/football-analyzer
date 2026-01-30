"""
Alert Manager

Manages and prioritizes coaching alerts, handling deduplication,
expiration, and delivery.
"""
import uuid
from typing import List, Dict, Optional, Set
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from models.schemas import TacticalAlert, AlertPriority


@dataclass
class AlertState:
    """Internal state for an alert."""
    alert: TacticalAlert
    created_at: int
    dismissed: bool = False
    delivered: bool = False
    delivery_count: int = 0


class AlertManager:
    """
    Manages tactical alerts for real-time coaching.

    Features:
    - Alert prioritization
    - Deduplication (similar alerts within timeframe)
    - Expiration handling
    - Rate limiting
    """

    def __init__(
        self,
        max_active_alerts: int = 5,
        dedup_window_ms: int = 5000,
        max_alerts_per_second: int = 2
    ):
        self.max_active_alerts = max_active_alerts
        self.dedup_window_ms = dedup_window_ms
        self.max_alerts_per_second = max_alerts_per_second

        # Alert storage
        self.active_alerts: Dict[str, AlertState] = {}
        self.alert_history: deque = deque(maxlen=100)
        self.dismissed_ids: Set[str] = set()

        # Rate limiting
        self.recent_alerts: deque = deque(maxlen=10)

        # Callbacks for alert delivery
        self._on_alert_callbacks: List[callable] = []

    async def add_alert(self, alert: TacticalAlert) -> bool:
        """
        Add a new alert to the system.

        Args:
            alert: The tactical alert to add

        Returns:
            True if alert was added, False if filtered out
        """
        current_time = alert.timestamp_ms

        # Check rate limit
        if not self._check_rate_limit(current_time):
            return False

        # Check for duplicates
        if self._is_duplicate(alert):
            return False

        # Check expiration
        if alert.expires_at_ms and alert.expires_at_ms < current_time:
            return False

        # Create alert state
        state = AlertState(
            alert=alert,
            created_at=current_time
        )

        # Add to active alerts (may evict lower priority)
        added = self._add_to_active(state)

        if added:
            # Add to history
            self.alert_history.append(alert)
            self.recent_alerts.append(current_time)

            # Trigger callbacks
            await self._deliver_alert(alert)

        return added

    async def add_alerts(self, alerts: List[TacticalAlert]) -> List[TacticalAlert]:
        """
        Add multiple alerts, filtering and prioritizing.

        Args:
            alerts: List of alerts to process

        Returns:
            List of alerts that were actually added
        """
        added = []

        # Sort by priority (immediate first)
        sorted_alerts = sorted(
            alerts,
            key=lambda a: self._priority_value(a.priority),
            reverse=True
        )

        for alert in sorted_alerts:
            if await self.add_alert(alert):
                added.append(alert)

        return added

    def _check_rate_limit(self, current_time: int) -> bool:
        """Check if we're within rate limit."""
        if not self.recent_alerts:
            return True

        # Count alerts in last second
        one_second_ago = current_time - 1000
        recent_count = sum(
            1 for t in self.recent_alerts
            if t > one_second_ago
        )

        return recent_count < self.max_alerts_per_second

    def _is_duplicate(self, alert: TacticalAlert) -> bool:
        """Check if a similar alert was recently added."""
        for existing_state in self.active_alerts.values():
            existing = existing_state.alert

            # Check time window
            time_diff = abs(alert.timestamp_ms - existing.timestamp_ms)
            if time_diff > self.dedup_window_ms:
                continue

            # Check similarity
            if self._alerts_similar(alert, existing):
                return True

        return False

    def _alerts_similar(
        self,
        alert1: TacticalAlert,
        alert2: TacticalAlert
    ) -> bool:
        """Check if two alerts are similar enough to be duplicates."""
        # Same priority type
        if alert1.priority != alert2.priority:
            return False

        # Similar message (simple check)
        msg1_words = set(alert1.message.lower().split())
        msg2_words = set(alert2.message.lower().split())

        overlap = len(msg1_words & msg2_words)
        total = len(msg1_words | msg2_words)

        if total > 0 and overlap / total > 0.5:
            return True

        return False

    def _priority_value(self, priority: AlertPriority) -> int:
        """Get numeric value for priority comparison."""
        return {
            AlertPriority.IMMEDIATE: 3,
            AlertPriority.TACTICAL: 2,
            AlertPriority.STRATEGIC: 1
        }.get(priority, 0)

    def _add_to_active(self, state: AlertState) -> bool:
        """Add alert to active list, potentially evicting lower priority."""
        alert = state.alert

        # If we have room, just add
        if len(self.active_alerts) < self.max_active_alerts:
            self.active_alerts[alert.alert_id] = state
            return True

        # Find lowest priority alert
        lowest_priority = None
        lowest_id = None

        for aid, astate in self.active_alerts.items():
            priority_val = self._priority_value(astate.alert.priority)
            if lowest_priority is None or priority_val < lowest_priority:
                lowest_priority = priority_val
                lowest_id = aid

        # Check if new alert has higher priority
        new_priority = self._priority_value(alert.priority)
        if new_priority > lowest_priority:
            # Evict lower priority alert
            del self.active_alerts[lowest_id]
            self.active_alerts[alert.alert_id] = state
            return True

        return False

    async def _deliver_alert(self, alert: TacticalAlert):
        """Deliver alert via registered callbacks."""
        for callback in self._on_alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Dismiss an active alert.

        Args:
            alert_id: ID of alert to dismiss

        Returns:
            True if alert was dismissed
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].dismissed = True
            self.dismissed_ids.add(alert_id)
            del self.active_alerts[alert_id]
            return True
        return False

    def get_active_alerts(self) -> List[TacticalAlert]:
        """Get all active (non-expired, non-dismissed) alerts."""
        current_time = int(datetime.now().timestamp() * 1000)
        active = []

        for alert_id in list(self.active_alerts.keys()):
            state = self.active_alerts[alert_id]

            # Check expiration
            if state.alert.expires_at_ms and state.alert.expires_at_ms < current_time:
                del self.active_alerts[alert_id]
                continue

            if not state.dismissed:
                active.append(state.alert)

        # Sort by priority
        active.sort(
            key=lambda a: self._priority_value(a.priority),
            reverse=True
        )

        return active

    def get_alerts_by_priority(
        self,
        priority: AlertPriority
    ) -> List[TacticalAlert]:
        """Get active alerts of specific priority."""
        return [
            a for a in self.get_active_alerts()
            if a.priority == priority
        ]

    def on_alert(self, callback: callable):
        """
        Register callback for new alerts.

        Args:
            callback: Function to call with new alerts
        """
        self._on_alert_callbacks.append(callback)

    def clear_expired(self, current_time_ms: int):
        """Remove expired alerts."""
        for alert_id in list(self.active_alerts.keys()):
            state = self.active_alerts[alert_id]
            if state.alert.expires_at_ms and state.alert.expires_at_ms < current_time_ms:
                del self.active_alerts[alert_id]

    def get_alert_stats(self) -> Dict:
        """Get statistics about alerts."""
        total = len(self.alert_history)
        by_priority = {
            "immediate": 0,
            "tactical": 0,
            "strategic": 0
        }

        for alert in self.alert_history:
            by_priority[alert.priority.value] += 1

        return {
            "total_alerts": total,
            "active_alerts": len(self.active_alerts),
            "dismissed_alerts": len(self.dismissed_ids),
            "by_priority": by_priority
        }

    def reset(self):
        """Reset all alert state."""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.dismissed_ids.clear()
        self.recent_alerts.clear()
