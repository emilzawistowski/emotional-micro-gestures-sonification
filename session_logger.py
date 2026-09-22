

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from feature_extractor import FeatureVector
from gesture_mapper import SynthParams

logger = logging.getLogger(__name__)

# HDF5 chunk size: 10 seconds of data at 20 Hz = 200 rows per chunk
_CHUNK_ROWS = 200
_FLUSH_INTERVAL = 5.0   # seconds between HDF5 flush-to-disk


class SessionLogger:
    """
    Buffers feature vectors and synth parameters and writes them to disk.

    Supports both HDF5 (preferred) and CSV (fallback if h5py unavailable).
    Writing is asynchronous — the logger maintains an internal buffer and
    flushes on a background thread to avoid blocking the real-time pipeline.

    Parameters
    ----------
    output_dir : str | Path
        Directory where session files are written.
    participant_id : str
        Participant identifier (used in filename and metadata).
    condition : str
        Experimental condition label ('stress', 'relax', 'baseline', etc.)
    session_id : str | None
        Unique session identifier; auto-generated from timestamp if None.
    use_hdf5 : bool
        Use HDF5 if True and h5py is available; fall back to CSV otherwise.
    """

    FEATURE_COLS = [
        "cursor_speed", "cursor_acceleration", "cursor_jerk",
        "speed_variance", "cursor_entropy",
        "iki_mean_norm", "iki_std_norm", "burst_density",
        "pause_ratio", "arousal_proxy", "valence_proxy",
    ]
    SYNTH_COLS = [
        "spectral_brightness", "fm_index", "dissonance",
        "grain_density", "grain_duration",
        "reverb_decay", "reverb_mix",
        "base_pitch", "pitch_register", "mode_tonality",
        "master_volume", "envelope_attack", "envelope_release",
    ]

    def __init__(
        self,
        output_dir: str | Path = "sessions",
        participant_id: str = "P00",
        condition: str = "unknown",
        session_id: Optional[str] = None,
        use_hdf5: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._participant_id = participant_id
        self._condition = condition
        self._session_id = session_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        self._use_hdf5 = use_hdf5 and self._check_h5py()

        # In-memory buffer lists (flushed periodically)
        self._buf_timestamps:   list[float] = []
        self._buf_features:     list[list[float]] = []
        self._buf_synth_params: list[list[float]] = []
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

        # File handles opened in start()
        self._h5_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open file handles and begin background flush thread."""
        self._start_time = time.monotonic()
        if self._use_hdf5:
            self._open_hdf5()
        else:
            self._open_csv()

        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        logger.info(
            "SessionLogger started [%s] — participant=%s condition=%s",
            "HDF5" if self._use_hdf5 else "CSV",
            self._participant_id,
            self._condition,
        )

    def stop(self) -> None:
        """Flush remaining buffer and close file handles."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._flush_buffer()
        if self._h5_file:
            self._h5_file.close()
        if self._csv_file:
            self._csv_file.close()
        logger.info("SessionLogger stopped. Data written to %s", self._output_dir)

    def log(self, fv: FeatureVector, params: SynthParams) -> None:
        """
        Buffer one frame of data (non-blocking).

        Called at 20 Hz from the main pipeline; must return quickly.
        """
        ts = fv.timestamp - self._start_time  # relative session time
        feat_row  = [getattr(fv, c) for c in self.FEATURE_COLS]
        param_row = [getattr(params, c) for c in self.SYNTH_COLS]

        with self._lock:
            self._buf_timestamps.append(ts)
            self._buf_features.append(feat_row)
            self._buf_synth_params.append(param_row)

    @property
    def output_path(self) -> Path:
        ext = ".h5" if self._use_hdf5 else ".csv"
        return self._output_dir / f"ems_{self._participant_id}_{self._session_id}{ext}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(_FLUSH_INTERVAL)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        with self._lock:
            if not self._buf_timestamps:
                return
            ts   = self._buf_timestamps[:]
            feat = self._buf_features[:]
            synt = self._buf_synth_params[:]
            self._buf_timestamps.clear()
            self._buf_features.clear()
            self._buf_synth_params.clear()

        if self._use_hdf5:
            self._write_hdf5(ts, feat, synt)
        else:
            self._write_csv(ts, feat, synt)

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def _open_hdf5(self) -> None:
        import h5py
        path = self.output_path
        self._h5_file = h5py.File(path, "w")
        grp = self._h5_file.create_group("session")

        # Metadata attributes
        grp.attrs["participant_id"] = self._participant_id
        grp.attrs["condition"]      = self._condition
        grp.attrs["session_id"]     = self._session_id
        grp.attrs["sample_rate_hz"] = 20.0
        grp.attrs["created_utc"]    = datetime.utcnow().isoformat()
        grp.attrs["feature_cols"]   = self.FEATURE_COLS
        grp.attrs["synth_cols"]     = self.SYNTH_COLS

        n_feat  = len(self.FEATURE_COLS)
        n_synth = len(self.SYNTH_COLS)

        # Resizable datasets
        grp.create_dataset(
            "timestamps",
            shape=(0,), maxshape=(None,),
            dtype=np.float64,
            chunks=(_CHUNK_ROWS,), compression="lzf",
        )
        grp.create_dataset(
            "features",
            shape=(0, n_feat), maxshape=(None, n_feat),
            dtype=np.float32,
            chunks=(_CHUNK_ROWS, n_feat), compression="lzf",
        )
        grp.create_dataset(
            "synth_params",
            shape=(0, n_synth), maxshape=(None, n_synth),
            dtype=np.float32,
            chunks=(_CHUNK_ROWS, n_synth), compression="lzf",
        )
        logger.debug("HDF5 file opened: %s", path)

    def _write_hdf5(
        self,
        ts: list[float],
        feat: list[list[float]],
        synt: list[list[float]],
    ) -> None:
        grp = self._h5_file["session"]
        n = len(ts)

        for ds_name, data in [
            ("timestamps",   np.array(ts, dtype=np.float64)),
            ("features",     np.array(feat, dtype=np.float32)),
            ("synth_params", np.array(synt, dtype=np.float32)),
        ]:
            ds = grp[ds_name]
            old_len = ds.shape[0]
            ds.resize(old_len + n, axis=0)
            if ds_name == "timestamps":
                ds[old_len:] = data
            else:
                ds[old_len:] = data

        self._h5_file.flush()

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def _open_csv(self) -> None:
        path = self.output_path
        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        all_cols = (
            ["timestamp_s", "participant_id", "condition"]
            + self.FEATURE_COLS
            + self.SYNTH_COLS
        )
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=all_cols)
        self._csv_writer.writeheader()
        logger.debug("CSV file opened: %s", path)

    def _write_csv(
        self,
        ts: list[float],
        feat: list[list[float]],
        synt: list[list[float]],
    ) -> None:
        for t, f_row, s_row in zip(ts, feat, synt):
            row = {
                "timestamp_s":    round(t, 4),
                "participant_id": self._participant_id,
                "condition":      self._condition,
            }
            for col, val in zip(self.FEATURE_COLS, f_row):
                row[col] = round(float(val), 5)
            for col, val in zip(self.SYNTH_COLS, s_row):
                row[col] = round(float(val), 5)
            self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _check_h5py() -> bool:
        try:
            import h5py  # noqa: F401
            return True
        except ImportError:
            logger.warning("h5py not found; falling back to CSV logging.")
            return False
