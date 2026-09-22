
import sys
import time
from pathlib import Path

import h5py
import numpy as np


# ─── 1. Inspect the session file ────────────────────────────────────────────

def check_session(path: str) -> bool:
    print("\n" + "="*60)
    print("STEP 1 — Session file inspection")
    print("="*60)

    with h5py.File(path, "r") as f:
        grp        = f["session"]
        timestamps  = grp["timestamps"][:]
        features    = grp["features"][:]
        synth       = grp["synth_params"][:]
        feat_cols   = list(grp.attrs["feature_cols"])
        synth_cols  = list(grp.attrs["synth_cols"])

    n = len(timestamps)
    duration = timestamps[-1] if n > 0 else 0
    print(f"  Frames recorded : {n}")
    print(f"  Duration        : {duration:.1f}s")
    print(f"  Sample rate     : {n/duration:.1f} Hz\n" if duration > 0 else "")

    if n < 20:
        print("  ERROR: Too few frames. The session is too short or empty.")
        return False

    print("  Feature ranges (should NOT all be 0.00 – 0.00):")
    all_zero = True
    for i, col in enumerate(feat_cols):
        col_data = features[:, i]
        mn, mx, std = col_data.min(), col_data.max(), col_data.std()
        flag = "  <-- FLAT (PROBLEM)" if std < 0.001 else ""
        print(f"    {col:30s}  min={mn:.3f}  max={mx:.3f}  std={std:.3f}{flag}")
        if std > 0.001:
            all_zero = False

    print()
    print("  Synth param ranges:")
    synth_varying = False
    for i, col in enumerate(synth_cols):
        col_data = synth[:, i]
        mn, mx, std = col_data.min(), col_data.max(), col_data.std()
        flag = "  <-- FLAT" if std < 0.001 else ""
        print(f"    {col:30s}  min={mn:.3f}  max={mx:.3f}  std={std:.3f}{flag}")
        if std > 0.001:
            synth_varying = True

    print()
    if all_zero:
        print("  VERDICT: Features are ALL ZERO — input collection failed.")
        print("  Likely cause: pynput had no events (mouse/keyboard not captured).")
        print("  Fix: make sure you were actively typing/moving mouse during recording.")
        return False
    elif not synth_varying:
        print("  VERDICT: Features vary but synth params are flat — mapping bug.")
        return False
    else:
        print("  VERDICT: Session data looks good — features and params vary.")
        return True


# ─── 2. Live SC connectivity test ────────────────────────────────────────────

def check_sc_live(node_id: int = 1000):
    print("\n" + "="*60)
    print("STEP 2 — SuperCollider live connectivity test")
    print("="*60)

    try:
        from pythonosc import udp_client
    except ImportError:
        print("  ERROR: python-osc not installed. Run: pip install python-osc")
        return

    client = udp_client.SimpleUDPClient("127.0.0.1", 57110)

    print(f"  Sending to node {node_id} on 127.0.0.1:57110")
    print("  You should hear CLEAR changes in sound for each step.")
    print("  If sound is SILENT: SC server not running or EMS.scd not evaluated.")
    print("  If sound does NOT change: node_id mismatch (check config.yaml).\n")

    steps = [
        ("Silence (volume=0)",
         ["volume", 0.0]),
        ("Loud + calm (low arousal)",
         ["volume", 0.7, "brightness", 0.1, "grain_density", 0.05,
          "fm_index", 0.05, "dissonance", 0.02, "reverb_decay", 0.9,
          "pitch", 48.0, "mode", 0.9]),
        ("LOUD + stressed (high arousal)",
         ["volume", 0.85, "brightness", 0.95, "grain_density", 0.95,
          "fm_index", 0.9, "dissonance", 0.8, "reverb_decay", 0.15,
          "pitch", 74.0, "mode", 0.1]),
        ("Back to calm",
         ["volume", 0.7, "brightness", 0.1, "grain_density", 0.05,
          "fm_index", 0.05, "dissonance", 0.02, "reverb_decay", 0.9,
          "pitch", 48.0, "mode", 0.9]),
        ("Neutral reset",
         ["volume", 0.65, "brightness", 0.3, "grain_density", 0.2,
          "fm_index", 0.1, "dissonance", 0.05, "reverb_decay", 0.6,
          "pitch", 60.0, "mode", 0.5]),
    ]

    for label, params in steps:
        args = [node_id] + params
        client.send_message("/n_set", args)
        print(f"  [{label}]")
        time.sleep(3.0)

    print("\n  Did the sound change noticeably between calm and stressed?")
    answer = input("  Type y (yes) or n (no): ").strip().lower()

    if answer == "y":
        print("\n  SC connection is WORKING.")
        print("  The problem is in the replay script — the param values")
        print("  sent during replay are not varying enough.")
        print("  Run: python diagnose.py YOUR_FILE.h5 --show-replay")
    else:
        print("\n  SC is NOT responding to parameter changes.")
        print("  Checklist:")
        print("    [ ] SC server booted? (Cmd+B, wait for 'Server running')")
        print("    [ ] EMS.scd evaluated? (Cmd+A then Cmd+Return)")
        print("    [ ] Node ID correct? Check SC Post window vs config.yaml sc_node_id")
        print("    [ ] Volume up on your Mac?")
        print(f"\n  To check the actual node ID, run in SC IDE:")
        print(f"    s.queryAllNodes;")
        print(f"  Look for a node near ID {node_id} in the Post window.")


# ─── 3. Print a sample of replay values ──────────────────────────────────────

def show_replay_sample(path: str):
    print("\n" + "="*60)
    print("STEP 3 — Replay parameter sample (first 10 frames of window)")
    print("="*60)

    with h5py.File(path, "r") as f:
        grp       = f["session"]
        timestamps = grp["timestamps"][:]
        synth      = grp["synth_params"][:]
        synth_cols = list(grp.attrs["synth_cols"])

    # Same window as quick_export: 15s–75s
    mask = (timestamps >= 15.0) & (timestamps < 75.0)
    ts   = timestamps[mask] - 15.0
    data = synth[mask]

    if len(data) == 0:
        print("  No data in the 15–75s window. Session may be too short.")
        print(f"  Total session length: {timestamps[-1]:.1f}s")
        print("  Try --start 0 in quick_export.py")
        return

    print(f"  Window: 15s–75s, {len(data)} frames\n")
    print(f"  {'time':>6}  {'brightness':>10}  {'grain_density':>13}  "
          f"{'fm_index':>8}  {'pitch':>6}  {'volume':>6}")
    print("  " + "-"*60)

    # Map column names to indices
    col_idx = {c: i for i, c in enumerate(synth_cols)}
    b_i  = col_idx.get("spectral_brightness", 0)
    gd_i = col_idx.get("grain_density",       3)
    fm_i = col_idx.get("fm_index",            1)
    p_i  = col_idx.get("base_pitch",          7)
    v_i  = col_idx.get("master_volume",       10)

    # Print every 50th frame (~2.5s intervals) for overview
    indices = list(range(0, min(len(data), 500), 50))
    for i in indices:
        row = data[i]
        print(f"  {ts[i]:6.1f}s  "
              f"{row[b_i]:10.3f}  {row[gd_i]:13.3f}  "
              f"{row[fm_i]:8.3f}  {row[p_i]:6.1f}  {row[v_i]:6.3f}")

    # Summarise variance
    print()
    print("  If all rows above look IDENTICAL → params not varying → mapping bug.")
    print("  If rows vary → SC is not applying the changes → node ID wrong.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EMS diagnostic tool")
    parser.add_argument("session_file", nargs="?", help="Path to .h5 session file")
    parser.add_argument("--node-id", type=int, default=1000)
    parser.add_argument("--show-replay", action="store_true",
                        help="Show sample of replay values from the session file")
    args = parser.parse_args()

    if args.session_file:
        ok = check_session(args.session_file)
        if args.show_replay or not ok:
            show_replay_sample(args.session_file)

    check_sc_live(node_id=args.node_id)


if __name__ == "__main__":
    main()
