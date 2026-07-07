"""
Combined Distress Score Calculator
Integrates audio, face, eye, and posture data to calculate overall distress score
"""
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CombinedDistressCalculator:
    """
    Calculates distress score by combining multiple modalities:
    - Audio: coughing, crying, groaning, gasping, normal
    - Face: facial expression indicators
    - Eye: eye tracking and blinking patterns
    - Posture: body posture indicators
    """

    # Weights for each modality (sum = 1.0)
    MODALITY_WEIGHTS = {
        "audio": 0.40,      # Audio is primary indicator
        "face": 0.30,       # Facial expressions
        "eye": 0.15,        # Eye movement and blinking
        "posture": 0.15     # Body posture
    }

    # Audio class importance (higher = more distressful)
    AUDIO_IMPORTANCE = {
        "coughing": 0.8,
        "crying": 0.95,
        "groaning": 0.7,
        "gasping": 0.9,
        "normal": 0.0,
        "noise": 0.0  # Noise doesn't indicate distress
    }

    # Alert level thresholds
    ALERT_THRESHOLDS = {
        "normal": (0.0, 0.3),
        "mild": (0.3, 0.5),
        "moderate": (0.5, 0.7),
        "severe": (0.7, 1.0)
    }

    def __init__(self):
        self.history = []
        self.smoothing_window = 5  # For smoothing scores over time

    def calculate_audio_score(self, audio_probs: Dict[str, float]) -> float:
        """
        Calculate distress score from audio classification probabilities.
        
        Args:
            audio_probs: Dict with keys like 'coughing', 'crying', etc.
        
        Returns:
            Audio distress score (0.0 to 1.0)
        """
        if not audio_probs:
            return 0.0

        audio_score = 0.0
        for event_type, probability in audio_probs.items():
            if event_type in self.AUDIO_IMPORTANCE:
                importance = self.AUDIO_IMPORTANCE[event_type]
                audio_score += probability * importance

        return min(1.0, audio_score)

    def calculate_face_score(self, face_data: Dict) -> float:
        """
        Calculate distress score from face analysis.
        
        Args:
            face_data: Dict with keys like:
                - 'detected': bool (face detected)
                - 'emotion': str (emotion label)
                - 'emotion_confidence': float (0-1)
                - 'mouth_open': float (0-1, how open)
        
        Returns:
            Face distress score (0.0 to 1.0)
        """
        if not face_data or not face_data.get("detected"):
            return 0.5  # Unknown if face not detected

        face_score = 0.0

        # Emotion-based distress
        emotion = face_data.get("emotion", "neutral").lower()
        emotion_confidence = face_data.get("emotion_confidence", 0.0)

        emotion_distress = {
            "angry": 0.7,
            "sad": 0.8,
            "fearful": 0.9,
            "disgusted": 0.6,
            "surprised": 0.5,
            "neutral": 0.2,
            "happy": 0.1
        }

        face_score += emotion_distress.get(emotion, 0.5) * emotion_confidence

        # Mouth openness (could indicate gasping, pain)
        mouth_open = face_data.get("mouth_open", 0.0)
        if mouth_open > 0.6:  # Wide open = potential distress
            face_score += 0.3

        return min(1.0, face_score)

    def calculate_eye_score(self, eye_data: Dict) -> float:
        """
        Calculate distress score from eye analysis.
        
        Args:
            eye_data: Dict with keys like:
                - 'detected': bool (eyes detected)
                - 'blink_rate': float (blinks per minute)
                - 'gaze_direction': str or tuple
                - 'eye_openness': float (0-1)
                - 'pupil_dilation': float (0-1)
        
        Returns:
            Eye distress score (0.0 to 1.0)
        """
        if not eye_data or not eye_data.get("detected"):
            return 0.5  # Unknown if eyes not detected

        eye_score = 0.0

        # Abnormal blink rate (normal: 15-20 blinks/min)
        # Higher or lower than normal can indicate stress
        blink_rate = eye_data.get("blink_rate", 17.5)
        if blink_rate > 30 or blink_rate < 8:
            eye_score += 0.4

        # Eye openness (wide open = alert/stressed)
        eye_openness = eye_data.get("eye_openness", 0.5)
        if eye_openness > 0.8:
            eye_score += 0.3

        # Pupil dilation (dilated = stress/interest)
        pupil_dilation = eye_data.get("pupil_dilation", 0.5)
        if pupil_dilation > 0.7:
            eye_score += 0.3

        return min(1.0, eye_score)

    def calculate_posture_score(self, posture_data: Dict) -> float:
        """
        Calculate distress score from posture analysis.
        
        Args:
            posture_data: Dict with keys like:
                - 'detected': bool (person detected)
                - 'hunched': float (0-1, how hunched)
                - 'head_down': float (0-1, how much head is down)
                - 'movement_intensity': float (0-1)
                - 'hand_to_face': bool (hand touching face)
        
        Returns:
            Posture distress score (0.0 to 1.0)
        """
        if not posture_data or not posture_data.get("detected"):
            return 0.5  # Unknown if person not detected

        posture_score = 0.0

        # Hunched posture = stress
        hunched = posture_data.get("hunched", 0.0)
        if hunched > 0.5:
            posture_score += 0.3

        # Head down = withdrawal/pain
        head_down = posture_data.get("head_down", 0.0)
        if head_down > 0.5:
            posture_score += 0.3

        # Excessive movement = agitation
        movement_intensity = posture_data.get("movement_intensity", 0.0)
        if movement_intensity > 0.7:
            posture_score += 0.2

        # Hand to face = self-soothing/stress
        if posture_data.get("hand_to_face", False):
            posture_score += 0.2

        return min(1.0, posture_score)

    def calculate_combined_score(
        self,
        audio_probs: Dict[str, float] = None,
        face_data: Dict = None,
        eye_data: Dict = None,
        posture_data: Dict = None,
        smooth: bool = True
    ) -> Dict:
        """
        Calculate combined distress score from all modalities.
        
        Args:
            audio_probs: Audio classification probabilities
            face_data: Face analysis data
            eye_data: Eye analysis data
            posture_data: Posture analysis data
            smooth: Whether to smooth the score over time
        
        Returns:
            Dict with 'score', 'audio_score', 'face_score', 'eye_score', 'posture_score'
        """
        # Calculate individual scores
        audio_score = self.calculate_audio_score(audio_probs or {})
        face_score = self.calculate_face_score(face_data or {})
        eye_score = self.calculate_eye_score(eye_data or {})
        posture_score = self.calculate_posture_score(posture_data or {})

        # Calculate weighted combined score
        combined_score = (
            audio_score * self.MODALITY_WEIGHTS["audio"] +
            face_score * self.MODALITY_WEIGHTS["face"] +
            eye_score * self.MODALITY_WEIGHTS["eye"] +
            posture_score * self.MODALITY_WEIGHTS["posture"]
        )

        # Apply smoothing
        if smooth and len(self.history) > 0:
            recent_scores = [h["score"] for h in self.history[-self.smoothing_window:]]
            combined_score = np.mean(recent_scores + [combined_score])

        result = {
            "score": combined_score,
            "audio_score": audio_score,
            "face_score": face_score,
            "eye_score": eye_score,
            "posture_score": posture_score,
            "breakdown": {
                "audio": audio_score * self.MODALITY_WEIGHTS["audio"],
                "face": face_score * self.MODALITY_WEIGHTS["face"],
                "eye": eye_score * self.MODALITY_WEIGHTS["eye"],
                "posture": posture_score * self.MODALITY_WEIGHTS["posture"]
            }
        }

        self.history.append(result)
        return result

    def get_alert_level(self, distress_score: float) -> Tuple[str, str]:
        """
        Determine alert level and priority based on distress score.
        
        Returns:
            Tuple of (alert_level, priority)
        """
        for level, (min_val, max_val) in self.ALERT_THRESHOLDS.items():
            if min_val <= distress_score < max_val:
                if level == "normal":
                    priority = "info"
                elif level == "mild":
                    priority = "low"
                elif level == "moderate":
                    priority = "medium"
                else:  # severe
                    priority = "high"
                return level, priority

        return "severe", "high"

    def get_trend(self, window: int = 5) -> str:
        """
        Determine if distress is improving, stable, or worsening.
        
        Args:
            window: Number of recent entries to analyze
        
        Returns:
            "improving", "stable", or "worsening"
        """
        if len(self.history) < 2:
            return "stable"

        recent = [h["score"] for h in self.history[-window:]]
        if len(recent) < 2:
            return "stable"

        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        change = second_half - first_half
        if change > 0.05:
            return "worsening"
        elif change < -0.05:
            return "improving"
        else:
            return "stable"
