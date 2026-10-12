"""Run a long job in a background thread and let the page read its progress.

Streamlit pages must not touch widgets from another thread. So the worker thread only
writes to a RunJob object, and the page reads it every half second.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RunJob:
    """Progress and outcome of one background run."""

    fraction: float = 0.0
    message: str = "Waiting to start"
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    error_type: str | None = None
    done: bool = False
    _thread: threading.Thread | None = None

    def progress(self, fraction: float, message: str) -> None:
        """Callback the engine calls. Clamps fraction to 0..1 and keeps a short log."""
        self.fraction = min(max(float(fraction), 0.0), 1.0)
        self.message = message
        self.log.append(message)

    def start(self, fn: Callable[..., Any], **kwargs: Any) -> None:
        """Start fn(progress=..., **kwargs) in a thread. fn receives this job's progress callback."""

        def worker() -> None:
            try:
                self.result = fn(progress=self.progress, **kwargs)
                self.fraction = 1.0
                self.message = "Finished"
            except Exception as err:  # the page shows this, so keep the message readable
                self.error = str(err) or err.__class__.__name__
                self.error_type = err.__class__.__name__
                self.message = "Stopped with an error"
            finally:
                self.done = True

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and not self.done


def what_to_try(error_type: str | None, message: str) -> str:
    """A plain-English suggestion for the most common failures. Never hides the engine's own message."""
    text = message.lower()
    if "too large" in text or "limit" in text or "smaller" in text or "km²" in text or "km2" in text:
        return "Draw a smaller box, then run again."
    if "no pixels" in text or "clear" in text or "cloud" in text:
        return "Try different dates. Cloud cover blocked the usable pixels, so a different season may work."
    if "catalog" in text or "download" in text or "network" in text or "timeout" in text or "connect" in text:
        return "Check your internet connection and try again. The satellite catalogue may be busy."
    if error_type == "ValueError":
        return "Check the area and dates on the earlier steps, then run again."
    return "Run again. If it keeps failing, copy the message above and report it with the area and dates you used."
