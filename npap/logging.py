import logging
import warnings
from enum import Enum
from typing import Optional

# Package-level logger
_PACKAGE_LOGGER_NAME = "npap"

# Default format for NPAP loggers
_DEFAULT_FORMAT = "%(levelname)s - %(name)s - %(message)s"


class LogCategory(Enum):
    """
    Log categories for organizing messages by subsystem.

    Each category maps to a specific logger under the npap namespace.
    """

    INPUT = "input"
    PARTITIONING = "partitioning"
    AGGREGATION = "aggregation"
    VISUALIZATION = "visualization"
    VALIDATION = "validation"
    MANAGER = "manager"
    UTILS = "utils"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    if not name.startswith(_PACKAGE_LOGGER_NAME):
        name = f"{_PACKAGE_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> None:
    """
    Configure NPAP logging globally.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_string: Custom format string for log messages
        handler: Custom handler (defaults to StreamHandler)

    Example:
        from npap.logging import configure_logging
        import logging

        # Enable debug logging
        configure_logging(level=logging.DEBUG)

        # Log to file
        configure_logging(handler=logging.FileHandler('npap.log'))
    """
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create handler
    if handler is None:
        handler = logging.StreamHandler()

    # Set format
    format_str = format_string or _DEFAULT_FORMAT
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def disable_logging() -> None:
    """
    Disable all NPAP logging output.

    Useful for testing or when running in production where
    logs should be suppressed.
    """
    logging.getLogger(_PACKAGE_LOGGER_NAME).setLevel(logging.CRITICAL + 1)


def enable_logging(level: int = logging.INFO) -> None:
    """
    Enable NPAP logging at the specified level.

    Args:
        level: Logging level to set
    """
    logging.getLogger(_PACKAGE_LOGGER_NAME).setLevel(level)


# Initialize with a NullHandler by default (library best practice)
logging.getLogger(_PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


# =============================================================================
# UNIFIED LOGGING FUNCTION
# =============================================================================


def log_message(
    message: str,
    category: LogCategory = LogCategory.UTILS,
    level: int = logging.INFO,
    warn_user: bool = False,
    stacklevel: int = 3,
) -> None:
    """
    Unified logging function for consistent message handling across NPAP.

    This is the preferred way to log messages in the package. It provides:
    - Categorized logging by subsystem
    - Optional user warnings for important messages
    - Consistent formatting and behavior

    Args:
        message: The log message
        category: Log category (determines logger name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        warn_user: If True, also emit a Python warning (for interactive use)
        stacklevel: Stack level for warnings.warn (default: 3)

    Examples:
        # Simple info message
        log_message("Processing started", LogCategory.INPUT)

        # Warning with user notification
        log_message("Missing data", LogCategory.VALIDATION, logging.WARNING, warn_user=True)

        # Debug message
        log_message("Matrix shape: (100, 100)", LogCategory.PARTITIONING, logging.DEBUG)
    """
    logger_name = f"{_PACKAGE_LOGGER_NAME}.{category.value}"
    logger = logging.getLogger(logger_name)
    logger.log(level, message)

    if warn_user:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def log_debug(message: str, category: LogCategory = LogCategory.UTILS) -> None:
    """Log a debug message."""
    log_message(message, category, logging.DEBUG)


def log_info(message: str, category: LogCategory = LogCategory.UTILS) -> None:
    """Log an info message."""
    log_message(message, category, logging.INFO)


def log_warning(
    message: str, category: LogCategory = LogCategory.UTILS, warn_user: bool = True
) -> None:
    """Log a warning message, optionally alerting the user."""
    log_message(message, category, logging.WARNING, warn_user=warn_user)


def log_error(message: str, category: LogCategory = LogCategory.UTILS) -> None:
    """Log an error message."""
    log_message(message, category, logging.ERROR)
