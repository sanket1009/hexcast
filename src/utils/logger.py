"""
logger.py
=========
Shared logging utility for the entire CHBIndex pipeline.

Every module imports this and calls get_logger(__name__) to get
a configured logger that is consistent across the entire pipeline.

Usage in any module:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Starting PM2.5 ingestion")
    logger.warning("3 monitors had missing values")
    logger.error("EPA API returned 503")

The log level and format are controlled by pipeline.yaml:
    logging:
      level: "INFO"
      format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

Changing the level in pipeline.yaml affects every module simultaneously.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Get a configured logger for a pipeline module.

    Parameters
    ----------
    name : str
        Logger name — pass __name__ from the calling module.
        This makes the module name appear in every log line.
        Example: "src.ingestion.epa_aqs" or "src.spatial.kriging"

    level : str, optional
        Override the log level from pipeline.yaml.
        Valid values: "DEBUG", "INFO", "WARNING", "ERROR"
        If None, reads from pipeline.yaml → logging → level
        Useful for temporarily enabling DEBUG in one module only.

    Returns
    -------
    logging.Logger
        Configured logger ready to use.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started")
    2026-08-14 06:00:01 | src.ingestion.epa_aqs | INFO | Pipeline started

    >>> logger = get_logger(__name__, level="DEBUG")
    >>> logger.debug("Variogram parameters: nugget=0.2")
    2026-08-14 06:00:01 | src.ingestion.epa_aqs | DEBUG | Variogram parameters: nugget=0.2
    """

    # Get or create logger with this name
    # Python's logging module caches loggers by name —
    # calling get_logger("epa_aqs") twice returns the same logger object
    # This prevents duplicate log handlers being added on repeated calls
    logger = logging.getLogger(name)

    # If this logger already has handlers, it's already configured
    # Return it as-is to prevent duplicate log lines
    # This happens when a module is imported multiple times
    if logger.handlers:
        return logger

    # Load log level and format from pipeline.yaml
    # We import here (not at module top) to avoid circular imports —
    # config_loader imports logger, so logger cannot import config_loader
    # at the top level without creating a circular dependency
    log_level, log_format, date_format, log_to_file, log_file_path = _load_log_config()

    # Override level if explicitly provided
    if level is not None:
        log_level = level

    # Set the log level on this logger
    # Levels are hierarchical: DEBUG < INFO < WARNING < ERROR < CRITICAL
    # Setting INFO means DEBUG messages are silently ignored
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create formatter — defines how each log line looks
    # %(asctime)s   = timestamp: "2026-08-14 06:00:01"
    # %(name)s      = logger name: "src.ingestion.epa_aqs"
    # %(levelname)s = level: "INFO", "WARNING", "ERROR"
    # %(message)s   = the actual message you logged
    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )

    # Console handler — writes log lines to terminal/stdout
    # In Docker containers, stdout is captured by the container runtime
    # and accessible via: docker logs container_name
    # In Airflow, task logs appear in the Airflow UI
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — optionally also write logs to a file
    # Controlled by pipeline.yaml → logging → log_to_file
    # Useful in production to persist logs after container restarts
    if log_to_file and log_file_path:
        _add_file_handler(logger, log_file_path, numeric_level, formatter)

    # Prevent log messages from propagating to the root logger
    # Without this, messages appear twice — once from this handler
    # and once from the root logger's handler
    logger.propagate = False

    return logger


def _load_log_config() -> tuple[str, str, str, bool, Optional[str]]:
    """
    Load logging configuration from pipeline.yaml.

    Returns defaults if pipeline.yaml cannot be loaded —
    this prevents logger.py from crashing before the pipeline
    even starts, which would make debugging impossible.

    Returns
    -------
    tuple of (level, format, date_format, log_to_file, log_file_path)
    """
    # Default values — used if pipeline.yaml can't be loaded
    # INFO level, readable format, no file logging
    defaults = (
        "INFO",
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
        False,
        None,
    )

    try:
        # Import yaml directly — not through config_loader
        # to avoid circular imports
        import yaml

        # Find pipeline.yaml relative to this file's location
        # This file: src/utils/logger.py
        # pipeline.yaml: config/pipeline.yaml
        # Path(__file__).parent = src/utils/
        # .parent.parent = src/
        # .parent.parent.parent = chbindex/ (project root)
        project_root = Path(__file__).parent.parent.parent
        pipeline_config_path = project_root / "config" / "pipeline.yaml"

        if not pipeline_config_path.exists():
            return defaults

        with open(pipeline_config_path) as f:
            pipeline_config = yaml.safe_load(f)

        log_config = pipeline_config.get("logging", {})

        return (
            log_config.get("level", "INFO"),
            log_config.get(
                "format",
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            ),
            log_config.get("date_format", "%Y-%m-%d %H:%M:%S"),
            log_config.get("log_to_file", False),
            log_config.get("log_file_path", None),
        )

    except Exception:
        # If anything goes wrong loading config, use defaults
        # Never crash the logger — it's the tool you'd use to debug the crash
        return defaults


def _add_file_handler(
    logger: logging.Logger,
    log_file_path: str,
    level: int,
    formatter: logging.Formatter,
) -> None:
    """
    Add a file handler to write logs to disk.

    Creates the log directory if it doesn't exist.
    Called only when pipeline.yaml → logging → log_to_file is true.

    Parameters
    ----------
    logger : logging.Logger
        The logger to add the file handler to.
    log_file_path : str
        Path to the log file, relative to project root.
    level : int
        Numeric log level (e.g. logging.INFO = 20)
    formatter : logging.Formatter
        The formatter to apply to file log lines.
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        full_log_path = project_root / log_file_path

        # Create the logs/ directory if it doesn't exist
        full_log_path.parent.mkdir(parents=True, exist_ok=True)

        # RotatingFileHandler would be better in production
        # (automatically splits large log files)
        # FileHandler is simpler and sufficient for portfolio use
        file_handler = logging.FileHandler(full_log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    except Exception as e:
        # If file logging fails, warn on console but don't crash
        # Console logging still works — file logging is optional
        logger.warning(f"Could not set up file logging at {log_file_path}: {e}")