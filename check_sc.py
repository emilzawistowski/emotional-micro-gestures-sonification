

import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Test SC OSC connection")
    parser.add_argument("--node-id", type=int, default=1000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=57110)
    args = parser.parse_args()

    try:
        from pythonosc import udp_client
    except ImportError:
        print("ERROR: python-osc not installed. Run: pip install python-osc")
        return

    client = udp_client.SimpleUDPClient(args.host, args.port)
    node   = args.node_id

    print(f"Targeting SC node {node} on {args.host}:{args.port}")
    print("Sending test sequence: calm → stress → calm (3 seconds each)")
    print("You should hear the sound change. If silent, check EMS.scd is evaluated.")
    print()

    # Calm state
    print("  [1/3] Calm state...")
    client.send_message("/n_set", [
        node,
        "brightness", 0.15, "grain_density", 0.08, "fm_index", 0.05,
        "dissonance", 0.02, "reverb_decay", 0.85, "reverb_mix", 0.5,
        "pitch", 48.0, "mode", 0.8, "volume", 0.65, "attack", 0.8, "release", 2.0,
    ])
    time.sleep(3.0)

    # Stress state
    print("  [2/3] Stress state...")
    client.send_message("/n_set", [
        node,
        "brightness", 0.92, "grain_density", 0.95, "fm_index", 0.85,
        "dissonance", 0.75, "reverb_decay", 0.18, "reverb_mix", 0.22,
        "pitch", 74.0, "mode", 0.15, "volume", 0.82, "attack", 0.01, "release", 0.15,
    ])
    time.sleep(3.0)

    # Back to calm
    print("  [3/3] Back to calm...")
    client.send_message("/n_set", [
        node,
        "brightness", 0.15, "grain_density", 0.08, "fm_index", 0.05,
        "dissonance", 0.02, "reverb_decay", 0.85, "reverb_mix", 0.5,
        "pitch", 48.0, "mode", 0.8, "volume", 0.65, "attack", 0.8, "release", 2.0,
    ])
    time.sleep(3.0)

    # Reset to neutral
    client.send_message("/n_set", [
        node,
        "brightness", 0.3, "grain_density", 0.2, "fm_index", 0.1,
        "dissonance", 0.05, "reverb_decay", 0.6, "reverb_mix", 0.3,
        "pitch", 60.0, "mode", 0.5, "volume", 0.6, "attack", 0.5, "release", 1.0,
    ])

    print()
    print("Test complete.")
    print("If you heard sound changing: connection is working. Run main.py.")
    print("If silent: check the SC Post window — is 'EMS SynthDef loaded.' shown?")
    print(f"           Also verify sc_node_id in config.yaml matches node {node}.")

if __name__ == "__main__":
    main()
