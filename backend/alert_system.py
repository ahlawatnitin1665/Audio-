"""
Alert System
Generates alerts and notifications based on distress scores
"""
import logging
from typing import Dict, List, Callable
from datetime import datetime
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Alert:
    """Individual alert object"""

    def __init__(
        self,
        level: AlertLevel,
        message: str,
        distress_score: float,
        modality_scores: Dict[str, float],
        timestamp: datetime = None,
        alert_id: str = None
    ):
        self.level = level
        self.message = message
        self.distress_score = distress_score
        self.modality_scores = modality_scores
        self.timestamp = timestamp or datetime.now()
        self.alert_id = alert_id or f"alert_{int(self.timestamp.timestamp() * 1000)}"

    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "message": self.message,
            "distress_score": self.distress_score,
            "modality_scores": self.modality_scores,
            "timestamp": self.timestamp.isoformat()
        }


class AlertManager:
    """Manages alert generation and tracking"""

    def __init__(self, callback: Callable = None):
        """
        Initialize alert manager.
        
        Args:
            callback: Optional function to call when alerts are generated
                     Called as: callback(alert: Alert)
        """
        self.logger = logging.getLogger(__name__)
        self.callback = callback
        self.alert_history: List[Alert] = []
        self.last_severe_alert_time = None
        self.suppression_duration = 5  # seconds, suppress duplicate alerts
        self.alert_count = {
            AlertLevel.NORMAL: 0,
            AlertLevel.MILD: 0,
            AlertLevel.MODERATE: 0,
            AlertLevel.SEVERE: 0
        }

    def generate_alert(
        self,
        distress_score: float,
        audio_score: float,
        face_score: float,
        eye_score: float,
        posture_score: float,
        trend: str = "stable"
    ) -> Alert:
        """
        Generate alert based on distress score.
        
        Args:
            distress_score: Combined distress score (0-1)
            audio_score: Audio component score
            face_score: Face component score
            eye_score: Eye component score
            posture_score: Posture component score
            trend: "improving", "stable", or "worsening"
        
        Returns:
            Alert object
        """
        # Determine alert level
        if distress_score < 0.3:
            level = AlertLevel.NORMAL
        elif distress_score < 0.5:
            level = AlertLevel.MILD
        elif distress_score < 0.7:
            level = AlertLevel.MODERATE
        else:
            level = AlertLevel.SEVERE

        # Generate message
        message = self._generate_message(
            level, distress_score, audio_score, face_score, eye_score, posture_score, trend
        )

        modality_scores = {
            "audio": audio_score,
            "face": face_score,
            "eye": eye_score,
            "posture": posture_score
        }

        alert = Alert(level, message, distress_score, modality_scores)

        # Check if should suppress (avoid alert spam)
        if not self._should_suppress(alert):
            self.alert_history.append(alert)
            self.alert_count[level] += 1
            self.logger.info(f"Alert generated: {alert.level.value} - {alert.message}")

            # Call callback if provided
            if self.callback:
                self.callback(alert)

        return alert

    def _generate_message(
        self,
        level: AlertLevel,
        distress_score: float,
        audio_score: float,
        face_score: float,
        eye_score: float,
        posture_score: float,
        trend: str
    ) -> str:
        """Generate human-readable alert message"""
        trend_text = {
            "improving": "improving",
            "stable": "consistent",
            "worsening": "worsening"
        }.get(trend, "stable")

        # Find dominant modality
        scores = {
            "audio": audio_score,
            "face": face_score,
            "eye": eye_score,
            "posture": posture_score
        }
        dominant = max(scores, key=scores.get)

        if level == AlertLevel.NORMAL:
            return f"Patient is stable. Distress score: {distress_score:.2f} ({trend_text})"

        elif level == AlertLevel.MILD:
            return f"Mild distress detected ({dominant}). Score: {distress_score:.2f}. Status: {trend_text}"

        elif level == AlertLevel.MODERATE:
            return f"Moderate distress detected. Primary indicator: {dominant}. Score: {distress_score:.2f}. Trend: {trend_text}"

        else:  # SEVERE
            return f"SEVERE distress alert! Primary: {dominant} ({scores[dominant]:.2f}). Overall score: {distress_score:.2f}. IMMEDIATE ATTENTION REQUIRED"

    def _should_suppress(self, alert: Alert) -> bool:
        """Check if alert should be suppressed to avoid spam"""
        if alert.level == AlertLevel.SEVERE:
            # For severe alerts, only suppress if one occurred very recently
            if self.last_severe_alert_time:
                time_diff = (datetime.now() - self.last_severe_alert_time).total_seconds()
                if time_diff < self.suppression_duration:
                    return True
            self.last_severe_alert_time = datetime.now()

        return False

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        recent = self.alert_history[-limit:]
        return [alert.to_dict() for alert in recent]

    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            "total_alerts": len(self.alert_history),
            "normal": self.alert_count[AlertLevel.NORMAL],
            "mild": self.alert_count[AlertLevel.MILD],
            "moderate": self.alert_count[AlertLevel.MODERATE],
            "severe": self.alert_count[AlertLevel.SEVERE]
        }

    def clear_history(self):
        """Clear alert history"""
        self.alert_history = []
        self.alert_count = {
            AlertLevel.NORMAL: 0,
            AlertLevel.MILD: 0,
            AlertLevel.MODERATE: 0,
            AlertLevel.SEVERE: 0
        }
