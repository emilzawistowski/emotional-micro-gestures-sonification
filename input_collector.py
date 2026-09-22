from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from pynput import keyboard, mouse


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

class EventType(Enum):
    MOUSE_MOVE   = auto()
    MOUSE_CLICK  = auto()
    KEY_PRESS    = auto()
    KEY_RELEASE  = auto()


@dataclass(frozen=True, slots=True)
class InputEvent:
    """Immutable record of a single raw input event."""
    event_type: EventType
    timestamp:  float          # seconds since epoch (time.monotonic base)
    # Mouse fields
    x: Optional[float] = None
    y: Optional[float] = None
    button: Optional[str] = None
    pressed: Optional[bool] = None
    # Keyboard fields
    key: Optional[str] = None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class InputCollector:
    """
    Listens to OS-level mouse and keyboard events and places them into a
    thread-safe queue for downstream consumption.

    Parameters
    ----------
    max_queue_size : int
        Drop oldest events when the queue exceeds this size to prevent
        unbounded memory growth during analysis pauses.
    mouse_throttle_hz : float
        Maximum mouse-move events per second. The OS can fire >1000 Hz on
        some platforms; throttling keeps CPU usage reasonable.
    """

    def __init__(
        self,
        max_queue_size: int = 4096,
        mouse_throttle_hz: float = 100.0,
    ) -> None:
        self._queue: queue.Queue[InputEvent] = queue.Queue(maxsize=max_queue_size)
        self._throttle_interval: float = 1.0 / mouse_throttle_hz
        self._last_mouse_time: float = 0.0
        self._lock = threading.Lock()

        self._mouse_listener:    Optional[mouse.Listener]    = None
        self._keyboard_listener: Optional[keyboard.Listener] = None
        self._running = False

        # Track pressed keys to compute dwell time downstream
        self._key_press_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start OS-level listeners in background daemon threads."""
        if self._running:
            return
        self._running = True

        self._mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._mouse_listener.daemon = True
        self._keyboard_listener.daemon = True
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self) -> None:
        """Gracefully stop all listeners."""
        self._running = False
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._keyboard_listener:
            self._keyboard_listener.stop()

    def get_event(self, timeout: float = 0.01) -> Optional[InputEvent]:
        """
        Retrieve the next event from the queue (non-blocking with timeout).

        Returns None if the queue is empty within `timeout` seconds.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[InputEvent]:
        """Return all currently queued events without blocking."""
        events: list[InputEvent] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internal callbacks (called from pynput daemon threads)
    # ------------------------------------------------------------------

    def _push(self, event: InputEvent) -> None:
        """Thread-safe enqueue; silently drops if full (backpressure)."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest to maintain real-time behaviour
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                pass

    def _on_move(self, x: int, y: int) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_mouse_time < self._throttle_interval:
                return
            self._last_mouse_time = now
        self._push(InputEvent(
            event_type=EventType.MOUSE_MOVE,
            timestamp=now,
            x=float(x),
            y=float(y),
        ))

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        self._push(InputEvent(
            event_type=EventType.MOUSE_CLICK,
            timestamp=time.monotonic(),
            x=float(x),
            y=float(y),
            button=str(button),
            pressed=pressed,
        ))

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.monotonic()
        key_str = self._key_to_str(key)
        with self._lock:
            self._key_press_times[key_str] = now
        self._push(InputEvent(
            event_type=EventType.KEY_PRESS,
            timestamp=now,
            key=key_str,
        ))

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        now = time.monotonic()
        key_str = self._key_to_str(key)
        self._push(InputEvent(
            event_type=EventType.KEY_RELEASE,
            timestamp=now,
            key=key_str,
        ))

    @staticmethod
    def _key_to_str(key: keyboard.Key | keyboard.KeyCode) -> str:
        """Normalise pynput key objects to consistent strings."""
        if isinstance(key, keyboard.KeyCode):
            return key.char if key.char else f"<{key.vk}>"
        return key.name  # e.g. 'space', 'enter', 'backspace'


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------

class CollectorSession:
    """
    Context manager for safe collector lifecycle.

    Usage
    -----
    with CollectorSession() as collector:
        event = collector.get_event()
    """
    def __init__(self, **kwargs) -> None:
        self._collector = InputCollector(**kwargs)

    def __enter__(self) -> InputCollector:
        self._collector.start()
        return self._collector

    def __exit__(self, *_) -> None:
        self._collector.stop()