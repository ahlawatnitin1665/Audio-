import numpy as np
import librosa
from scipy.signal import butter, filtfilt, medfilt


class ICUNoiseFilter:
    def __init__(self):
        self.TARGET_SR = 16000

    def apply_notch_filter(self, y, sr, freqs=[50, 60]):
        """Remove power line hum only (50/60 Hz). Gentle, order-2 filter."""
        nyq = sr / 2.0
        for freq in freqs:
            low = (freq - 5) / nyq
            high = (freq + 5) / nyq
            if low <= 0 or high >= 1:
                continue
            b, a = butter(2, [low, high], btype='bandstop')
            y = filtfilt(b, a, y)
        return y

    def apply_light_cleanup(self, y, sr):
        """Very gentle broadband noise reduction using median filtering"""
        y_abs = np.abs(y)
        y_med = medfilt(y_abs, kernel_size=5)
        mask = (y_abs > y_med * 1.5).astype(float)
        mask = np.convolve(mask, np.ones(5) / 5, mode='same')
        y_cleaned = y * (0.3 + 0.7 * mask)
        return y_cleaned

    def apply_all_filters(self, y, sr):
        """Gentle noise filtering: notch hum + light cleanup"""
        y = self.apply_notch_filter(y, sr)
        y = self.apply_light_cleanup(y, sr)
        return y
