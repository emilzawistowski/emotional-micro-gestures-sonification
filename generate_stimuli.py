
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Stimulus extraction from session file
# ---------------------------------------------------------------------------

def load_session(path: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load feature and synth parameter timeseries from an HDF5 or CSV session file.

    Returns
    -------
    timestamps : np.ndarray  shape (N,)
    synth_params : np.ndarray  shape (N, 13)
    feature_cols : list[str]
    synth_cols : list[str]
    """
    p = Path(path)
    if p.suffix in (".h5", ".hdf5"):
        return _load_hdf5(p)
    elif p.suffix == ".csv":
        return _load_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def _load_hdf5(path: Path):
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py not installed: pip install h5py")
    with h5py.File(path, "r") as f:
        grp = f["session"]
        timestamps   = grp["timestamps"][:]
        synth_params = grp["synth_params"][:]
        synth_cols   = list(grp.attrs["synth_cols"])
        feature_cols = list(grp.attrs["feature_cols"])
    return timestamps, synth_params, feature_cols, synth_cols


def _load_csv(path: Path):
    import pandas as pd
    df = pd.read_csv(path)
    from session_logger import SessionLogger
    feature_cols = SessionLogger.FEATURE_COLS
    synth_cols   = SessionLogger.SYNTH_COLS
    timestamps   = df["timestamp_s"].values
    synth_params = df[synth_cols].values
    return timestamps, synth_params, feature_cols, synth_cols


def extract_window(
    timestamps: np.ndarray,
    synth_params: np.ndarray,
    start_s: float = 15.0,
    duration_s: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a [start_s, start_s + duration_s] window from the session."""
    end_s = start_s + duration_s
    mask = (timestamps >= start_s) & (timestamps < end_s)
    return timestamps[mask] - start_s, synth_params[mask]


# ---------------------------------------------------------------------------
# Audio rendering via SuperCollider OSC replay
# ---------------------------------------------------------------------------

def render_to_wav(
    timestamps: np.ndarray,
    synth_params: np.ndarray,
    synth_cols: list[str],
    output_path: str,
    sc_host: str = "127.0.0.1",
    sc_port: int = 57110,
    sc_node_id: int = 1000,
    record_duration: float = 62.0,
) -> None:
    """
    Replay synth parameters to SuperCollider while recording via SC's built-in
    recorder, then save to WAV.

    SC must be running with EMS.scd evaluated before calling this.

    Parameters
    ----------
    output_path : str
        WAV file path for output.
    record_duration : float
        Total recording time in SC (slightly longer than 60s to capture tail).
    """
    try:
        from pythonosc import udp_client
    except ImportError:
        raise RuntimeError("python-osc not installed: pip install python-osc")

    client = udp_client.SimpleUDPClient(sc_host, sc_port)
    out_path_abs = str(Path(output_path).resolve())

    # Start SC recording
    client.send_message("/b_alloc", [0, 65536, 2])  # allocate recording buffer
    client.send_message("/c_set", [0, 1])            # enable recording flag
    client.send_message(
        "/d_recv",
        # SC will write to the output path
        # Using NRT (non-real-time) approach via OSC record command
    )
    # Note: full NRT rendering requires SC's -o flag at startup.
    # For live recording, use SC IDE's built-in record: s.record(path)
    # Send that command via OSC using a helper SynthDef.

    print(f"  Replaying {len(timestamps)} parameter frames...")
    print(f"  Please start recording in SC IDE: s.record(\"{out_path_abs}\")")
    print("  Press Enter when SC recording has started...")
    input()

    # Replay parameter stream in real-time
    prev_t = 0.0
    for i, (t, row) in enumerate(zip(timestamps, synth_params)):
        wait = float(t) - prev_t
        if wait > 0:
            time.sleep(wait)
        prev_t = float(t)

        args = [sc_node_id]
        for col, val in zip(synth_cols, row):
            sc_key = _col_to_sc_key(col)
            args.extend([sc_key, float(val)])
        client.send_message("/n_set", args)

    # Tail
    time.sleep(2.0)
    print(f"  Replay complete. Stop SC recording: s.stopRecording")
    print(f"  Output saved to: {output_path}")


def _col_to_sc_key(col: str) -> str:
    mapping = {
        "spectral_brightness": "brightness",
        "fm_index":            "fm_index",
        "dissonance":          "dissonance",
        "grain_density":       "grain_density",
        "grain_duration":      "grain_dur",
        "reverb_decay":        "reverb_decay",
        "reverb_mix":          "reverb_mix",
        "base_pitch":          "pitch",
        "pitch_register":      "register",
        "mode_tonality":       "mode",
        "master_volume":       "volume",
        "envelope_attack":     "attack",
        "envelope_release":    "release",
    }
    return mapping.get(col, col)


# ---------------------------------------------------------------------------
# Stimulus naming and verification
# ---------------------------------------------------------------------------

def verify_stimuli(output_dir: str) -> None:
    """Print a summary of stimuli found in output_dir."""
    d = Path(output_dir)
    wavs = sorted(d.glob("*.wav"))
    if not wavs:
        print(f"No WAV files found in {output_dir}")
        return
    print(f"\nStimuli in {output_dir}:")
    for w in wavs:
        size_kb = w.stat().st_size / 1024
        print(f"  {w.name:30s}  {size_kb:8.1f} KB")
    stress = [w for w in wavs if "stress" in w.name]
    relax  = [w for w in wavs if "relax"  in w.name]
    print(f"\n  Stress stimuli: {len(stress)}")
    print(f"  Relax stimuli:  {len(relax)}")
    if len(stress) == len(relax) == 4:
        print("  Status: READY for experiment")
    else:
        print("  Status: INCOMPLETE — need 4 stress + 4 relax")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EMS stimulus generation")
    p.add_argument("--stress", nargs="+", help="Stress session file(s) (.h5 or .csv)")
    p.add_argument("--relax",  nargs="+", help="Relax session file(s) (.h5 or .csv)")
    p.add_argument("--n", type=int, default=4, help="Stimuli per condition")
    p.add_argument("--start", type=float, default=15.0,
                   help="Start time within session (s) — skips ramp-up")
    p.add_argument("--out", default="stimuli", help="Output directory")
    p.add_argument("--verify", action="store_true", help="Just verify existing stimuli")
    p.add_argument("--sc-node-id", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify_stimuli(args.out)
        return

    for condition, files in [("stress", args.stress), ("relax", args.relax)]:
        if not files:
            print(f"No {condition} files provided — skipping.")
            continue
        for i, f in enumerate(files[:args.n], start=1):
            print(f"\n[{condition.upper()} {i}/{args.n}] Loading {f}...")
            ts, params, fcols, scols = load_session(f)
            ts_win, params_win = extract_window(ts, params, start_s=args.start)
            out_path = out_dir / f"{condition}_{i:02d}.wav"
            print(f"  Window: {args.start:.0f}s – {args.start+60:.0f}s "
                  f"({len(ts_win)} frames)")
            render_to_wav(
                ts_win, params_win, scols,
                str(out_path),
                sc_node_id=args.sc_node_id,
            )

    verify_stimuli(args.out)


if __name__ == "__main__":
    main()
