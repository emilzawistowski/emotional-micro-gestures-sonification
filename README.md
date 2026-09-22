# Emotional Micro-Gestures Sonification (EMS)

A real-time affective sonification system that turns everyday computer interaction —
mouse movement, clicks, and typing dynamics — into a continuous soundscape.
Features extracted at 20 Hz are projected onto arousal/valence dimensions
(Russell's circumplex model) and mapped to perceptually grounded synthesis
parameters: grain density, spectral brightness, harmonic dissonance, and
reverberation decay. Synthesis runs in SuperCollider, controlled from Python
via Open Sound Control.

## Contents

- `main.py` - real-time pipeline runner (`--participant`, `--condition`, `--duration`, `--config`)
- `input_collector.py` - OS-level mouse/keyboard capture via `pynput`
- `feature_extractor.py` - 20 Hz feature extraction with adaptive percentile normalisation
- `gesture_mapper.py` - arousal/valence projection and mapping to `SynthParams`
- `synth_engine.py` - OSC client sending `/n_set` updates to the SuperCollider node
- `EMS.scd` - SuperCollider `ems_main` SynthDef (granular + FM layers, evaluate in SC IDE first)
- `session_logger.py` - session recording to HDF5/CSV
- `config.yaml` - SuperCollider connection, windowing, calibration, and mapping settings
- `requirements.txt` - Python dependencies
- `check_sc.py` - connection test (plays calm → stress → calm through the SC node)
- `diagnose.py` - session-file inspection helper
- `generate_stimuli.py` / `quick_export.py` - replay of recorded sessions for stimulus rendering
- `analysis.py` - listener-study statistics (`python analysis.py --data results.csv`)
- `results.csv` - anonymised listener ratings (12 participants, 8 trials each)
- `participant_instructions.txt` / `response_sheet.txt` - study materials

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Open SuperCollider IDE, evaluate `EMS.scd`, and note the node ID in the Post window (default `1000`, matching `sc_node_id` in `config.yaml`).
2. Verify the connection:
   ```bash
   python check_sc.py
   ```
3. Run the pipeline:
   ```bash
   python main.py --participant P00 --condition free
   ```
4. Analyse the included listener data:
   ```bash
   python analysis.py --data results.csv
   ```

macOS requires accessibility/input-monitoring permission for the terminal or IDE
running `main.py`, otherwise `pynput` receives no mouse/keyboard events.
Recorded sessions (`sessions/*.h5`) and rendered stimuli (`*.wav`) are
intentionally excluded; obtain or record compatible input separately.

## Status

This is an academic prototype. A listener study (N = 12) found above-chance
identification of the operator's emotional condition from audio alone, but the
mapping has not been validated beyond that study and reproducibility depends on
the local display, input hardware, and SuperCollider setup.

## Attribution

Created by Emil Zawistowski for coursework in Sound and Music Computing at Aalborg University.
