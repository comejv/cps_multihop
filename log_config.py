import logging
import sys
from pathlib import Path


class ColorizedFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[92m",  # Blue
        "INFO": "",  # Nothing
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)

        # Add colors if the level name is in our color dictionary
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        else:
            return log_message


def setup_logging(
    console_level=logging.WARN,
    file_level=logging.DEBUG,
    log_dir="logs",
    use_colors=True,
    components=["simulation", "environment", "vehicle", "network", "metrics", "main"],
):
    """
    Configure logging for components.

    Parameters:
    -----------
    console_level : int
        Logging level for console output (default: WARN)
    file_level : int
        Logging level for file output (default: DEBUG)
    log_dir : str
        Directory to store log files (default: "logs")
    use_colors : bool
        Whether to use colors in console output (default: True)
    components : list
        List of component names to create loggers for

    Returns:
    --------
    dict
        Dictionary of configured loggers
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)

    # Define formatter strings
    console_format = "[%(name)-10s:%(levelname)-8s] %(message)s"
    simple_format = "%(levelname)-8s: %(message)s"
    file_format = "[%(name)-10s:%(levelname)-8s] %(message)s"

    # Create formatters
    if use_colors:
        console_formatter = ColorizedFormatter(console_format)
        simple_formatter = ColorizedFormatter(simple_format)
    else:
        console_formatter = logging.Formatter(console_format)
        simple_formatter = logging.Formatter(simple_format)

    file_formatter = logging.Formatter(file_format)

    # Configure loggers
    loggers = {}

    # Create component-specific loggers
    for component in components:
        logger = logging.getLogger(component)
        logger.setLevel(logging.DEBUG)  # Allow all logs through the logger itself
        logger.propagate = False  # Don't propagate to root logger

        # Make sure handlers are cleared (in case function is called multiple times)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add file handler for all components except "main"
        if component != "main":
            component_file_handler = logging.FileHandler(
                f"{log_dir}/{component}.log", mode="w"
            )
            component_file_handler.setLevel(file_level)
            component_file_handler.setFormatter(file_formatter)
            logger.addHandler(component_file_handler)

        # Component-specific console handler
        component_console_handler = logging.StreamHandler(sys.stdout)
        component_console_handler.setLevel(console_level)

        # Main logger gets a simpler format
        if component == "main":
            component_console_handler.setFormatter(simple_formatter)
        else:
            component_console_handler.setFormatter(console_formatter)

        logger.addHandler(component_console_handler)

        loggers[component] = logger

    # Disable propagation of matplotlib and other noisy libraries
    for lib_logger in ["matplotlib", "PIL", "numpy", "pandas", "seaborn"]:
        logging.getLogger(lib_logger).setLevel(logging.WARNING)

    return loggers


def set_component_level(component, console_level=None, file_level=None):
    """
    Set logging level for a specific component

    Parameters:
    -----------
    component : str
        Name of the component (e.g., "environment", "vehicle")
    console_level : int, optional
        Logging level for console output (e.g., logging.INFO)
    file_level : int, optional
        Logging level for file output (e.g., logging.DEBUG)
    """
    logger = logging.getLogger(component)

    # If both levels are None, do nothing
    if console_level is None and file_level is None:
        return

    # If both levels are provided, set the logger's level to the lower of the two
    # to ensure all messages can pass through the logger itself
    if console_level is not None and file_level is not None:
        logger.setLevel(min(console_level, file_level))
    # Otherwise set it to whichever level is provided
    elif console_level is not None:
        logger.setLevel(console_level)
    elif file_level is not None:
        logger.setLevel(file_level)

    # Update handlers based on their type
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and file_level is not None:
            handler.setLevel(file_level)
        elif isinstance(handler, logging.StreamHandler) and console_level is not None:
            handler.setLevel(console_level)
