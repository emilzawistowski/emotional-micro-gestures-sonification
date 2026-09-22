"""
gesture_mapper.py  –  EMS  (no resonant pitch, full compatibility)
"""

from __future__ import annotations

import math
import numpy as np
from feature_extractor import FeatureVector
from dataclasses import dataclass


@dataclass
class SynthParams:
    spectral_brightness: float = 0.3
    fm_index: float = 0.2
    dissonance: float = 0.1
    harmonic_spread: float = 0.1
    grain_density: float = 0.3
    grain_duration: float = 0.08
    reverb_decay: float = 0.5
    reverb_mix: float = 0.3
    base_pitch: float = 60.0
    pitch_register: float = 0.5
    mode_tonality: float = 0.5
    master_volume: float = 0.7
    envelope_attack: float = 0.05
    envelope_release: float = 0.5
    lfo_rate: float = 0.1

    def to_osc_dict(self):
        return {
            "brightness": self.spectral_brightness,
            "fm_index": self.fm_index,
            "dissonance": self.dissonance,
            "harmonic_spread": self.harmonic_spread,
            "grain_density": self.grain_density,
            "grain_dur": self.grain_duration,
            "reverb_decay": self.reverb_decay,
            "reverb_mix": self.reverb_mix,
            "pitch": self.base_pitch,
            "register": self.pitch_register,
            "mode": self.mode_tonality,
            "volume": self.master_volume,
            "attack": self.envelope_attack,
            "release": self.envelope_release,
            "lfo_rate": self.lfo_rate,
        }


class EMAFilter:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.values = {}

    def smooth(self, key, new_val):
        prev = self.values.get(key, new_val)
        smoothed = self.alpha * new_val + (1 - self.alpha) * prev
        self.values[key] = smoothed
        return smoothed


class GestureMapper:
    _AROUSAL_WEIGHTS = {
        "cursor_speed": 0.20,
        "cursor_jerk": 0.25,
        "cursor_acceleration": 0.10,
        "burst_density": 0.25,
        "iki_mean_norm": 0.10,
        "click_rate": 0.10,
    }
    _VALENCE_WEIGHTS = {
        "cursor_entropy": 0.30,
        "speed_variance": 0.25,
        "pause_ratio": 0.25,
        "iki_std_norm": 0.20,
    }

    def __init__(self, ema_alpha=0.08, pitch_center=48.0, pitch_range=8.0):
        self.pitch_center = pitch_center
        self.pitch_range = pitch_range
        self.slow = EMAFilter(ema_alpha)
        self.fast = EMAFilter(min(ema_alpha * 2.5, 0.4))
        self.pitch_filter = EMAFilter(alpha=0.02)

    def map(self, fv: FeatureVector) -> SynthParams:
        arousal, valence = self._project(fv)
        return self._assign(arousal, valence)

    def _project(self, fv):
        a = sum(self._AROUSAL_WEIGHTS[k] * getattr(fv, k) for k in self._AROUSAL_WEIGHTS)
        a = np.clip(a, 0.0, 1.0)
        v = (
            self._VALENCE_WEIGHTS["cursor_entropy"] * fv.cursor_entropy
            + self._VALENCE_WEIGHTS["speed_variance"] * (1.0 - fv.speed_variance)
            + self._VALENCE_WEIGHTS["pause_ratio"] * (1.0 - fv.pause_ratio)
            + self._VALENCE_WEIGHTS["iki_std_norm"] * (1.0 - fv.iki_std_norm)
        )
        v = np.clip(v, 0.0, 1.0)
        return float(a), float(v)

    def _assign(self, arousal, valence):
        p = SynthParams()

        # Brightness (szum)
        p.spectral_brightness = self.fast.smooth("brightness", 0.05 + 0.95 * math.sqrt(arousal))

        # FM index – bardzo niski
        fm_raw = min(0.1, arousal ** 0.7 * 0.1)
        p.fm_index = self.fast.smooth("fm_index", fm_raw)

        # Dissonance
        diss_raw = 1.0 - self._sigmoid(valence, steepness=10.0, center=0.4)
        p.dissonance = self.slow.smooth("dissonance", diss_raw)

        # Harmonic spread
        p.harmonic_spread = self.fast.smooth("harmonic_spread", arousal ** 1.2)

        # Grain density
        p.grain_density = self.fast.smooth("grain_density", 0.05 + 0.65 * arousal)

        # Grain duration (krótkie)
        min_dur, max_dur = 0.010, 0.08
        dur_raw = min_dur * (max_dur / min_dur) ** (1.0 - arousal)
        p.grain_duration = self.slow.smooth("grain_duration", dur_raw)

        # Reverb decay
        p.reverb_decay = self.slow.smooth("reverb_decay", 0.4 + 0.5 * (1.0 - arousal))

        # Reverb mix
        p.reverb_mix = self.slow.smooth("reverb_mix", 0.2 + 0.4 * (1.0 - valence))

        # PITCH – bardzo wąski zakres, wolny
        arousal_for_pitch = arousal ** 2.2
        pitch_offset = ((arousal_for_pitch - 0.5) * 0.6 + (valence - 0.5) * 0.2) * self.pitch_range
        base_pitch_raw = self.pitch_center + pitch_offset
        p.base_pitch = float(np.clip(
            self.pitch_filter.smooth("base_pitch", base_pitch_raw), 40, 56
        ))

        # Pitch register
        p.pitch_register = self.slow.smooth("pitch_register", arousal * 0.5 + valence * 0.3)

        # Mode
        p.mode_tonality = self.slow.smooth("mode_tonality", valence)

        # LFO rate
        p.lfo_rate = self.fast.smooth("lfo_rate", arousal ** 1.5)

        # Volume
        p.master_volume = self.fast.smooth("volume", 0.5 + 0.2 * arousal)

        # Envelope
        p.envelope_attack = self.slow.smooth("attack", 0.05 + 0.25 * (1.0 - arousal))
        p.envelope_release = self.slow.smooth("release", 0.6 + 1.2 * (1.0 - arousal))

        return p

    @staticmethod
    def _sigmoid(x, steepness=6.0, center=0.5):
        exp_arg = -steepness * (x - center)
        exp_arg = max(-88, min(88, exp_arg))
        return 1.0 / (1.0 + math.exp(exp_arg))