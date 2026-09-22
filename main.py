from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from input_collector import InputCollector
from feature_extractor import FeatureExtractor, FeatureConfig, FeatureVector
from gesture_mapper import GestureMapper, SynthParams
from synth_engine import SuperColliderEngine, SynthesisPipeline
from session_logger import SessionLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ems.main")


DEFAULT_CONFIG: dict = {
    "sc_host":                  "127.0.0.1",
    "sc_port":                  57110,
    "sc_node_id":               1000,
    "window_seconds":           0.5,
    "kb_window_seconds":        2.0,
    "tick_interval":            0.05,
    "mouse_throttle_hz":        100.0,
    "calibration_seconds":      20.0,
    "calibration_percentile":   99.0,
    "ema_alpha":                0.08,
    "pitch_center":             60.0,
    "pitch_range":              14.0,
    "max_update_hz":            20.0,
    "output_dir":               "sessions",
    "use_hdf5":                 True,
}


def load_config(path: Optional[str]) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if path and Path(path).exists():
        with open(path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)
        logger.info("Config loaded from %s", path)
    else:
        logger.info("Using default config (no config.yaml found or specified).")
    return cfg


class EMSPipeline:
    """Orchestrates the full real-time EMS processing pipeline."""

    def __init__(
        self,
        participant_id: str = "P00",
        condition: str = "free",
        duration: Optional[float] = None,
        config: Optional[dict] = None,
    ) -> None:
        cfg = config or DEFAULT_CONFIG
        self._duration        = duration
        self._participant_id  = participant_id
        self._condition       = condition

        self._collector = InputCollector(
            mouse_throttle_hz=cfg["mouse_throttle_hz"],
        )
        self._feature_cfg = FeatureConfig(
            window_seconds=cfg["window_seconds"],
            kb_window_seconds=cfg.get("kb_window_seconds", 2.0),
            tick_interval=cfg["tick_interval"],
            calibration_seconds=cfg.get("calibration_seconds", 20.0),
            calibration_percentile=cfg.get("calibration_percentile", 99.0),
        )
        self._mapper = GestureMapper(
            ema_alpha=cfg["ema_alpha"],
            pitch_center=cfg["pitch_center"],
            pitch_range=cfg["pitch_range"],
        )
        engine = SuperColliderEngine(
            host=cfg["sc_host"],
            port=cfg["sc_port"],
            node_id=cfg["sc_node_id"],
        )
        self._synth = SynthesisPipeline(
            engine=engine,
            max_update_hz=cfg["max_update_hz"],
        )
        self._logger = SessionLogger(
            output_dir=cfg["output_dir"],
            participant_id=participant_id,
            condition=condition,
            use_hdf5=cfg["use_hdf5"],
        )
        self._extractor = FeatureExtractor(
            collector=self._collector,
            config=self._feature_cfg,
            on_features=self._on_features,
        )
        self._running = False

    def _on_features(self, fv: FeatureVector) -> None:
        """Fires at 20 Hz from FeatureExtractor thread. Must return fast."""
        params: SynthParams = self._mapper.map(fv)
        self._synth.push_params(params)
        self._logger.log(fv, params)

    def start(self) -> None:
        logger.info(
            "Starting EMS — participant=%s  condition=%s",
            self._participant_id, self._condition,
        )
        self._logger.start()
        self._synth.start()
        self._collector.start()
        self._extractor.start()
        self._running = True
        logger.info("Pipeline running [%s].", self._synth.backend_name)
        logger.info("Move your mouse and type to generate sound. Press Ctrl-C to stop.")

    def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping EMS pipeline...")
        self._extractor.stop()
        self._collector.stop()
        self._synth.stop()
        self._logger.stop()
        self._running = False
        logger.info("Session saved to: %s", self._logger.output_path)

    def run_blocking(self) -> None:
        self.start()

        def _sigint(sig, frame):
            print("\nInterrupt received — stopping cleanly...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint)

        if self._duration:
            end_time = time.monotonic() + self._duration
            while time.monotonic() < end_time and self._running:
                remaining = end_time - time.monotonic()
                fv = self._extractor.latest
                print(
                    f"\r  {remaining:6.1f}s left  |  "
                    f"arousal={fv.arousal_proxy:.2f}  "
                    f"valence={fv.valence_proxy:.2f}  |  "
                    f"jerk={fv.cursor_jerk:.2f}  "
                    f"burst={fv.burst_density:.2f}  "
                    f"pause={fv.pause_ratio:.2f}",
                    end="", flush=True,
                )
                time.sleep(0.25)
            print()
            self.stop()
        else:
            while self._running:
                fv = self._extractor.latest
                print(
                    f"\r  arousal={fv.arousal_proxy:.2f}  "
                    f"valence={fv.valence_proxy:.2f}  |  "
                    f"jerk={fv.cursor_jerk:.2f}  "
                    f"burst={fv.burst_density:.2f}  "
                    f"pause={fv.pause_ratio:.2f}",
                    end="", flush=True,
                )
                time.sleep(0.25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EMS — Emotional Micro-gestures Sonification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--participant", default="P00",
                   help="Participant ID (e.g. P01)")
    p.add_argument("--condition", default="free",
                   choices=["free", "stress", "relax", "baseline"],
                   help="Experimental condition")
    p.add_argument("--duration", type=float, default=None,
                   help="Session duration in seconds (omit = run until Ctrl-C)")
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--sc-host", default=None,
                   help="Override SuperCollider host from config")
    p.add_argument("--sc-port", type=int, default=None,
                   help="Override SuperCollider port from config")
    p.add_argument("--sc-node-id", type=int, default=None,
                   help="Override SC node ID (check SC Post window after evaluating EMS.scd)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides take precedence over config file
    if args.sc_host:
        config["sc_host"] = args.sc_host
    if args.sc_port:
        config["sc_port"] = args.sc_port
    if args.sc_node_id:
        config["sc_node_id"] = args.sc_node_id

    pipeline = EMSPipeline(
        participant_id=args.participant,
        condition=args.condition,
        duration=args.duration,
        config=config,
    )
    pipeline.run_blocking()

if __name__ == "__main__":
    main()