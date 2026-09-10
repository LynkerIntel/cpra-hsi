"""Logging configuration for a single model run.

The rule this module exists to enforce: **modules get loggers, the run
attaches handlers**. Every module in this package obtains its logger with
`get_logger(__name__)` and never touches handlers, levels or filters. All
of those loggers sit under the `vegprocessor` root, and handlers are added
to that root in exactly one place — `attach_run_handlers` — for the
duration of a single run. The model owns that scope: `_setup_logger`
attaches, `close` detaches, and `__exit__` calls `close`, so
`with HSI(...) as hsi:` bounds it.

That matters because `logging.getLogger(name)` returns a process-wide
singleton. When each model instance configured its own named loggers, a
batch of configs sharing one process (see `batch_run.py`) accumulated one
file handler per run, and every message from run N was written into the
log files of runs 1..N-1 as well. Scoping the handlers makes that
impossible: only one run's handlers are attached at a time, and `__exit__`
removes them even if the run raises.

The current timestep is carried in a `ContextVar` rather than by a filter
holding a reference to the model instance. That keeps finished runs
collectable, and it tags records from *any* module in the package —
including `utils` and `species_hsi`, which the old instance-bound filter
could not reach.
"""

import logging
import os
import warnings
from contextvars import ContextVar
from typing import Optional

import pandas as pd

# Root of the package logger tree. Handlers are attached here and nowhere
# else; every other logger in the package is a descendant and propagates up.
ROOT_LOGGER_NAME = "vegprocessor"

LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[Timestep: %(timestep)s] - %(message)s"
)

# Set by `step()` via `set_timestep`; read by `_TimestepFilter` when a
# record is emitted.
_current_timestep: ContextVar[Optional[str]] = ContextVar(
    "current_timestep", default=None
)


def get_logger(name: str) -> logging.Logger:
    """Return the package logger for a module.

    Parameters
    ----------
    name : str
        Module name, normally `__name__`.

    Returns
    -------
    logging.Logger
        A logger under `ROOT_LOGGER_NAME`. It has no handlers of its own —
        records propagate to the root, where the active run's handlers are
        attached.
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def set_timestep(timestep: Optional[pd.Timestamp]) -> None:
    """Set the timestep tagged onto subsequent log records.

    Parameters
    ----------
    timestep : pd.Timestamp or None
        Current model timestep. `None` renders as "N/A", for messages
        emitted outside the run loop.

    Returns
    -------
    None
    """
    _current_timestep.set(
        timestep.strftime("%Y-%m-%d") if timestep is not None else None
    )


class _TimestepFilter(logging.Filter):
    """Inject the current timestep into every record.

    Attached to the handlers rather than to a logger, so it is discarded
    with them when the run ends.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.timestep = _current_timestep.get() or "N/A"
        return True


def attach_run_handlers(
    output_dir: str,
    file_name: str,
    log_level: int = logging.INFO,
) -> list[logging.Handler]:
    """Route package logging to a run's output folder.

    Called by `VegTransition._setup_logger`. Every call must be matched by
    a `detach_run_handlers`; using the model as a context manager
    (`with HSI(...) as hsi:`) is what guarantees that.

    Parameters
    ----------
    output_dir : str
        The run's output directory. The log is written to its
        `run-metadata` subdirectory.
    file_name : str
        Run naming convention, used as the log file stem.
    log_level : int
        Level for the root package logger and both handlers.

    Returns
    -------
    list of logging.Handler
        The handlers that were attached, to be passed to
        `detach_run_handlers`.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers:
        # a previous run was never closed; its handlers would also receive
        # this run's messages. Loud, because it silently corrupts log files.
        warnings.warn(
            f"{len(root.handlers)} log handler(s) from an earlier run are "
            "still attached; that run's log file will also receive this "
            "run's messages. Use the model as a context manager "
            "(`with HSI(...) as hsi:`) or call `close()` on it.",
            RuntimeWarning,
            stacklevel=2,
        )
    root.setLevel(log_level)
    # handlers live here, so don't also hand records to the stdlib root
    root.propagate = False

    run_metadata_dir = os.path.join(output_dir, "run-metadata")
    os.makedirs(run_metadata_dir, exist_ok=True)
    log_file_path = os.path.join(
        run_metadata_dir, f"{file_name}_simulation.log"
    )

    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file_path),
    ]
    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        handler.addFilter(_TimestepFilter())
        root.addHandler(handler)

    return handlers


def detach_run_handlers(handlers: list[logging.Handler]) -> None:
    """Remove and close handlers previously attached to the package root.

    Parameters
    ----------
    handlers : list of logging.Handler
        The return value of `attach_run_handlers`.

    Returns
    -------
    None
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()
