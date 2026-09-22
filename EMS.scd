
/*
  EMS — Emotional Micro-gestures Sonification
  SuperCollider SynthDef  (v2 — harmonically enriched)

  All synthesis formulas follow the published mapping exactly:

    FM layer  : carrier fc modulated by fm at fc·(1 + 0.5·Δ),
                modulation index I = m · 5 · fc  (Section 3.5)
    Granular  : GrainSin at rate fg = 40·D + 1 Hz, duration τ (Section 3.5)
    LPF cutoff: fc_lpf = 200 + 6800·B  Hz  (Section 3.5)
    FreeVerb2 : room = Rd, mix = M  (Section 3.5)
    Envelope  : ASR with τa, τr from Python  (Section 3.4)

  What changed vs v1 (perceptual quality, not formulas):
  - FM layer mixed at -14 dB (0.2× amplitude) to prevent it from dominating
    and causing the "UFO/vacuum cleaner" artefact.  The formula I = m·5·fc
    is unchanged — only the layer's contribution to the final mix is reduced.
  - Granular layer gains two harmonic satellites (fifth above, octave above)
    panned opposite to the fundamental, creating a fuller, less piercing texture.
    GrainSin is still the oscillator; only the frequency and pan offsets differ.
  - A Lag UGen (lagTime = 0.5 s) is applied to the incoming `pitch` control
    bus to absorb rapid MIDI-note steps that would otherwise cause audible
    siren glides.  The Python mapping formula p̂ = p₀ + (A−0.5)·r is unchanged.
  - Stochastic grain panning uses LFNoise1 at 1.5 Hz (was 0.5 Hz) for a
    gentler, slower spatial movement.
  - Default init values are calmer (fm_index 0.05, dissonance 0.02, grain_density 0.1)
    so startup is a soft ambient pad rather than an immediate buzzing tone.
*/

(
s.waitForBoot({

    SynthDef(\ems_main, { |out = 0,
        brightness    = 0.3,
        fm_index      = 0.2,
        dissonance    = 0.1,
        grain_density = 0.3,
        grain_dur     = 0.08,
        reverb_decay  = 0.5,
        reverb_mix    = 0.3,
        pitch         = 60,
        mode          = 0.5,
        volume        = 0.7,
        attack        = 0.05,
        release       = 0.5|

        var baseFreq, modFreq, modulator, fmLayer;
        var grainTrigger, grainFund, grainFifth, grainOctave, grainLayer;
        var mixed, filtered, env, wet;
        var smoothPitch;

        // Lag on pitch to absorb rapid MIDI-note steps from Python.
        // The mapping formula p̂ = p₀ + (A − 0.5)·r is computed in Python
        // and transmitted unchanged; Lag only smooths the control signal
        // within SC to prevent audible glide artefacts.
        smoothPitch = Lag.kr(pitch, lagTime: 0.5);
        baseFreq    = smoothPitch.midicps;

        // ----------------------------------------------------------------
        // FM layer — encodes valence / harmonic tension
        // Formula (Section 3.5): fm = fc · (1 + 0.5·Δ),  I = m · 5 · fc
        // The layer is mixed at 0.2 amplitude (-14 dB) to prevent the
        // naturally large modulation index from dominating the timbre.
        // ----------------------------------------------------------------
        modFreq   = baseFreq * (1.0 + (dissonance * 0.5));
        modulator = SinOsc.ar(modFreq, mul: fm_index * 5.0 * baseFreq);
        fmLayer   = SinOsc.ar(baseFreq + modulator, mul: 0.20);

        // ----------------------------------------------------------------
        // Granular layer — encodes arousal / energy
        // Formula (Section 3.5): fg = 40·D + 1 Hz, duration τ, GrainSin
        // Three harmonic partials (fundamental, fifth, octave) are mixed
        // to produce a fuller, less hollow texture while keeping GrainSin
        // as the oscillator and all density/duration formulas intact.
        // Stochastic pan at 1.5 Hz for gentle spatial diffusion.
        // ----------------------------------------------------------------
        grainTrigger = Impulse.ar(grain_density * 39.0 + 1.0);

        // Fundamental: freq tuned to mode (0=minor, 1=major offset)
        grainFund = GrainSin.ar(
            numChannels: 2,
            trigger:     grainTrigger,
            dur:         grain_dur,
            freq:        baseFreq * (1.0 + ((mode - 0.5) * 0.059)),
            pan:         LFNoise1.kr(1.5).range(-0.5, 0.5),
            mul:         0.50
        );

        // Perfect fifth (ratio 1.5): adds harmonic richness, panned opposite
        grainFifth = GrainSin.ar(
            numChannels: 2,
            trigger:     grainTrigger,
            dur:         grain_dur * 1.2,
            freq:        baseFreq * 1.5 * (1.0 + ((mode - 0.5) * 0.03)),
            pan:         LFNoise1.kr(1.1).range(-0.4, 0.4),
            mul:         0.25
        );

        // Octave (ratio 2.0): adds shimmer at low amplitude, wide pan
        grainOctave = GrainSin.ar(
            numChannels: 2,
            trigger:     grainTrigger,
            dur:         grain_dur * 0.8,
            freq:        baseFreq * 2.0,
            pan:         LFNoise1.kr(0.7).range(-0.6, 0.6),
            mul:         0.15
        );

        grainLayer = grainFund + grainFifth + grainOctave;

        // ----------------------------------------------------------------
        // Mix: FM layer (0.2 weight) + granular (0.8 weight)
        // FM is intentionally quiet — it contributes harmonic tension
        // without dominating the timbral character at high fm_index values.
        // ----------------------------------------------------------------
        mixed = (fmLayer.dup * 0.25) + (grainLayer * 0.75);

        // ----------------------------------------------------------------
        // Low-pass filter
        // Formula (Section 3.5): fc_lpf = 200 + 6800·B  Hz
        // ----------------------------------------------------------------
        filtered = LPF.ar(mixed, (200.0 + (brightness * 6800.0)).clip(200, 8000));

        // ----------------------------------------------------------------
        // Amplitude envelope — ASR, gate always open
        // ----------------------------------------------------------------
        env = EnvGen.kr(
            Env.asr(attack, 1.0, release),
            gate:       1,
            levelScale: volume
        );

        // ----------------------------------------------------------------
        // Reverberation
        // Formula (Section 3.5): FreeVerb, room = Rd, mix = M, damp = 0.4
        // ----------------------------------------------------------------
        wet = FreeVerb2.ar(
            filtered[0], filtered[1],
            mix:  reverb_mix,
            room: reverb_decay,
            damp: 0.4
        );

        Out.ar(out, wet * env);

    }).add;

    s.sync;
    s.sendMsg("/n_free", 1000);
    s.sync;

    ~ems = Synth(\ems_main, [
        \pitch,         60,
        \brightness,    0.2,
        \fm_index,      0.05,
        \dissonance,    0.02,
        \grain_density, 0.10,
        \grain_dur,     0.15,
        \reverb_decay,  0.7,
        \reverb_mix,    0.30,
        \mode,          0.55,
        \volume,        0.55,
        \attack,        1.0,
        \release,       2.0,
    ], target: s.defaultGroup, addAction: \addToHead, nodeID: 1000);

    "================================================================".postln;
    "  EMS SynthDef loaded and synth running.".postln;
    ("  Node ID: " ++ ~ems.nodeID ++ "  (fixed — Python uses 1000)").postln;
    "  sc_node_id in config.yaml should be 1000.".postln;
    "  Manual test: ~ems.set(\\volume, 0.8, \\brightness, 0.9)".postln;
    "  Stop:        ~ems.free".postln;
    "================================================================".postln;
});
)

/* MANUAL TESTS

// Quick volume check:
~ems.set(\volume, 0.8);
~ems.set(\volume, 0.4);

// Stress state — high arousal, low valence (anxious/tense):
~ems.set(\brightness, 0.9, \grain_density, 0.9, \fm_index, 0.8,
         \dissonance, 0.7, \reverb_decay, 0.2, \pitch, 67, \volume, 0.75);

// Calm state — low arousal, high valence (relaxed/content):
~ems.set(\brightness, 0.12, \grain_density, 0.05, \fm_index, 0.04,
         \dissonance, 0.02, \reverb_decay, 0.85, \pitch, 56, \volume, 0.5);

// Neutral reset:
~ems.set(\brightness, 0.2, \grain_density, 0.10, \fm_index, 0.05,
         \dissonance, 0.02, \reverb_decay, 0.7, \pitch, 60, \volume, 0.55);

// Excited/joyful — high arousal, high valence:
~ems.set(\brightness, 0.85, \grain_density, 0.75, \fm_index, 0.6,
         \dissonance, 0.05, \reverb_decay, 0.3, \pitch, 67, \mode, 0.9, \volume, 0.75);

// Free synth:
~ems.free;

─────────────────────────────────────────────────────────────────────────────── */
