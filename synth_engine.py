"""SuperCollider OSC engine for EMS — sends /n_set updates to the ems_main node."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from gesture_mapper import SynthParams

logger = logging.getLogger(__name__)

_SC_HOST    = "127.0.0.1"
_SC_PORT    = 57110
_SC_NODE_ID = 1000   # default first user synth node in SC


class SuperColliderEngine:
    """
    Sends OSC /n_set messages to the 'ems_main' synth node running in SC.

    The node must already exist (created by evaluating EMS.scd).
    Python does NOT create or free the node — SC owns the node lifecycle.

    Parameters
    ----------
    host : str
        scsynth host (always 127.0.0.1 for local SC).
    port : int
        scsynth UDP port (default 57110).
    node_id : int
        The node ID printed by SC after evaluating EMS.scd.
        Check the SC Post window for: "Node ID: XXXX"
    """

    def __init__(
        self,
        host: str = _SC_HOST,
        port: int = _SC_PORT,
        node_id: int = _SC_NODE_ID,
    ) -> None:
        self._host    = host
        self._port    = port
        self._node_id = node_id
        self._client  = None
        self._running = False

    def start(self) -> None:
        try:
            from pythonosc import udp_client
        except ImportError:
            raise RuntimeError(
                "python-osc is not installed.\n"
                "Fix: pip install python-osc"
            )
        self._client  = udp_client.SimpleUDPClient(self._host, self._port)
        self._running = True

        # Send a no-op /n_set with volume to verify connectivity.
        # If SC is not running this silently fails (UDP is fire-and-forget).
        self._client.send_message("/n_set", [self._node_id, "volume", 0.6])
        logger.info(
            "SuperCollider engine ready — targeting node %d on %s:%d",
            self._node_id, self._host, self._port,
        )
        logger.info(
            "If you hear no sound, verify EMS.scd is evaluated in SC IDE "
            "and the node ID matches sc_node_id in config.yaml."
        )

    def stop(self) -> None:
        """
        Fade out gracefully by setting volume to 0.
        We do NOT free the node — SC keeps it alive for the session.
        Free manually with ~ems.free in the SC IDE when done.
        """
        if self._client and self._running:
            self._client.send_message(
                "/n_set", [self._node_id, "volume", 0.0, "release", 1.0]
            )
            time.sleep(1.1)   # wait for release to complete
            self._running = False
            logger.info(
                "SuperCollider engine stopped (node %d faded out; "
                "free it with ~ems.free in SC IDE if desired).",
                self._node_id,
            )

    def update(self, params: SynthParams) -> None:
        """Send all synthesis parameters as a single /n_set message."""
        if not self._running or self._client is None:
            return
        args: list = [self._node_id]
        for key, value in _params_to_sc_args(params):
            args.append(key)
            args.append(float(value))
        self._client.send_message("/n_set", args)

    @property
    def backend_name(self) -> str:
        return f"SuperCollider OSC → node {self._node_id} @ {self._host}:{self._port}"


def _params_to_sc_args(params: SynthParams) -> list[tuple[str, float]]:
    """
    Map SynthParams fields to SC SynthDef argument names.
    Names must exactly match the |arg names| in EMS.scd.
    """
    return [
        ("brightness",    params.spectral_brightness),
        ("fm_index",      params.fm_index),
        ("dissonance",    params.dissonance),
        ("grain_density", params.grain_density),
        ("grain_dur",     params.grain_duration),
        ("reverb_decay",  params.reverb_decay),
        ("reverb_mix",    params.reverb_mix),
        ("pitch",         params.base_pitch),
        ("mode",          params.mode_tonality),
        ("volume",        params.master_volume),
        ("attack",        params.envelope_attack),
        ("release",       params.envelope_release),
    ]


class SynthesisPipeline:
    """
    Rate-limited dispatch wrapper around SuperColliderEngine.

    Decouples the 20 Hz feature-extraction tick from OSC sends.
    Runs its own daemon thread — push_params() is non-blocking.
    """

    def __init__(
        self,
        engine: SuperColliderEngine,
        max_update_hz: float = 20.0,
    ) -> None:
        self._engine   = engine
        self._interval = 1.0 / max_update_hz
        self._latest:  Optional[SynthParams] = None
        self._lock     = threading.Lock()
        self._thread:  Optional[threading.Thread] = None
        self._running  = False

    def start(self) -> None:
        self._engine.start()
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="SynthDispatch", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._engine.stop()

    def push_params(self, params: SynthParams) -> None:
        """Thread-safe; called from the feature-extractor callback at 20 Hz."""
        with self._lock:
            self._latest = params

    def _loop(self) -> None:
        next_send = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now >= next_send:
                with self._lock:
                    params = self._latest
                if params is not None:
                    try:
                        self._engine.update(params)
                    except Exception as exc:
                        logger.warning("OSC send failed: %s", exc)
                next_send = now + self._interval
            else:
                time.sleep(max(0.001, next_send - now - 0.001))

    @property
    def backend_name(self) -> str:
        return self._engine.backend_name
