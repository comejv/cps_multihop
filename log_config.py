import logging
import sys
from pathlib import Path


def setup_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    log_dir="logs",
    log_filename="unnamed.log",
):
    """
    Configure logging for the VANET simulation.

    Parameters:
    -----------
    console_level : int
        Logging level for console output (default: INFO)
    file_level : int
        Logging level for file output (default: DEBUG)
    log_dir : str
        Directory to store log files (default: "logs")
    log_filename : str
        Filename for the main log file (default: "simulation.log")

    Returns:
    --------
    logging.Logger
        The root logger
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs

    # Clear any existing handlers (in case this is called multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    console_formatter = logging.Formatter("%(levelname)-8s: %(message)s")
    file_formatter = logging.Formatter(
        "[%(name)-10s:%(levelname)-8s] %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/{log_filename}", mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create component-specific loggers
    create_component_logger("simulation", console_level, file_level)
    create_component_logger("environment", console_level, file_level)
    create_component_logger("vehicle", console_level, file_level)
    create_component_logger("network", console_level, file_level)
    create_component_logger("metrics", console_level, file_level)

    # Special setup for __main__ logger - console only
    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(console_level)
    main_logger.propagate = False  # Don't propagate to root logger
    
    # Clear any existing handlers
    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)
    
    # Add console handler only
    main_console_handler = logging.StreamHandler(sys.stdout)
    main_console_handler.setLevel(console_level)
    main_console_handler.setFormatter(console_formatter)
    main_logger.addHandler(main_console_handler)

    # Disable propagation of matplotlib and other noisy libraries
    for lib_logger in ["matplotlib", "PIL", "numpy", "pandas", "seaborn"]:
        logging.getLogger(lib_logger).setLevel(logging.WARNING)

    return root_logger


def create_component_logger(name, console_level, file_level):
    """Create a logger for a specific component with its own file"""
    logger = logging.getLogger(name)

    # Component-specific file handler
    Path("logs").mkdir(exist_ok=True)
    file_handler = logging.FileHandler(f"logs/{name}.log", mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("[%(name)-10s:%(levelname)-8s] %(message)s")
    )
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def set_component_level(component, level):
    """
    Set logging level for a specific component

    Parameters:
    -----------
    component : str
        Name of the component (e.g., "environment", "vehicle")
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = logging.getLogger(component)
    logger.setLevel(level)

    # Update handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
