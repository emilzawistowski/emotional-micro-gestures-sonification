

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from pythonosc import udp_client


# SC parameter names — must match EMS.scd arg list exactly
SYNTH_COLS_TO_SC = {
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


def load_session(path: str):
    """Load timestamps and synth params from HDF5 session file."""
    with h5py.File(path, "r") as f:
        grp        = f["session"]
        timestamps  = grp["timestamps"][:]
        synth_data  = grp["synth_params"][:]
        synth_cols  = list(grp.attrs["synth_cols"])
        condition   = grp.attrs.get("condition", "unknown")
        participant = grp.attrs.get("participant_id", "unknown")
    return timestamps, synth_data, synth_cols, condition, participant


def extract_window(timestamps, synth_data, start_s=15.0, duration_s=60.0):
    """Extract a 60-second window starting at start_s (skips ramp-up)."""
    end_s = start_s + duration_s
    mask  = (timestamps >= start_s) & (timestamps < end_s)
    ts    = timestamps[mask] - start_s   # relative time from 0
    data  = synth_data[mask]
    return ts, data


def replay(ts, data, synth_cols, node_id=1000, host="127.0.0.1", port=57110):
    """Replay parameter stream to SuperCollider in real time."""
    client = udp_client.SimpleUDPClient(host, port)

    print(f"\n  Replaying {len(ts)} frames over {ts[-1]:.1f} seconds...")
    print("  Watch the terminal — press Ctrl+C to abort.\n")

    prev_t = 0.0
    for i, (t, row) in enumerate(zip(ts, data)):
        # Maintain real-time pacing
        wait = float(t) - prev_t
        if wait > 0.001:
            time.sleep(wait)
        prev_t = float(t)

        # Build /n_set argument list
        args = [node_id]
        for col_idx, col_name in enumerate(synth_cols):
            sc_key = SYNTH_COLS_TO_SC.get(col_name)
            if sc_key:
                args.extend([sc_key, float(row[col_idx])])

        client.send_message("/n_set", args)

        # Progress indicator every 5 seconds
        if i % 100 == 0:
            print(f"  {t:5.1f}s / {ts[-1]:.1f}s", end="\r", flush=True)

    print(f"\n  Replay complete ({len(ts)} frames sent).")


def main():
    parser = argparse.ArgumentParser(
        description="Replay EMS session to SC and record WAV"
    )
    parser.add_argument("session_file", help="Path to .h5 session file")
    parser.add_argument("--start",   type=float, default=15.0,
                        help="Start time in session (default: 15s, skips ramp-up)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration to export in seconds (default: 60)")
    parser.add_argument("--node-id", type=int, default=1000,
                        help="SC synth node ID (check config.yaml)")
    parser.add_argument("--out",     default=None,
                        help="Suggested output filename (you set this in SC IDE)")
    args = parser.parse_args()

    # Sanity check
    if not Path(args.session_file).exists():
        print(f"ERROR: File not found: {args.session_file}")
        sys.exit(1)

    # Load data
    print(f"\nLoading: {args.session_file}")
    ts_full, data_full, synth_cols, condition, participant = load_session(
        args.session_file
    )
    print(f"  Participant : {participant}")
    print(f"  Condition   : {condition}")
    print(f"  Total length: {ts_full[-1]:.1f}s  ({len(ts_full)} frames at ~20 Hz)")

    # Extract window
    ts, data = extract_window(ts_full, data_full, args.start, args.duration)
    print(f"  Export window: {args.start:.0f}s – {args.start + args.duration:.0f}s "
          f"({len(ts)} frames)")

    if len(ts) < 10:
        print("ERROR: Window too short — check --start and --duration values.")
        sys.exit(1)

    # Suggest output filename
    out_name = args.out or f"{condition}_01.wav"
    abs_out  = str(Path(out_name).resolve())

    # ----------------------------------------------------------------
    # Instructions
    # ----------------------------------------------------------------
    print("\n" + "=" * 58)
    print("  BEFORE YOU PRESS ENTER — do this in SuperCollider IDE:")
    print()
    print(f'  s.record("{abs_out}");')
    print()
    print("  Copy-paste that line into SC IDE and press Cmd+Return.")
    print("  Wait until you see 'Recording channels...' in the Post window.")
    print("=" * 58)
    input("\n  Press Enter here when SC is recording... ")

    # ----------------------------------------------------------------
    # Replay
    # ----------------------------------------------------------------
    replay(ts, data, synth_cols, node_id=args.node_id)

    # ----------------------------------------------------------------
    # Stop instructions
    # ----------------------------------------------------------------
    print("\n" + "=" * 58)
    print("  NOW in SuperCollider IDE run:")
    print()
    print("  s.stopRecording;")
    print()
    print(f"  Your WAV file: {abs_out}")
    print("=" * 58)
    print()
    print("  Listen to it to verify the sound changed over the 60 seconds.")
    print("  If it sounds static/identical throughout, check that EMS.scd")
    print("  is evaluated and the node ID matches config.yaml.")


if __name__ == "__main__":
    main()
