import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO, console_output: bool = False):
    """
    Set up a logger.

    Args:
        name (str): The name of the logger.
        log_file (str): The file path for the log file.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        console_output (bool): If True, also log to the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            # Handle potential race condition if directory is created by another process
            if not os.path.isdir(log_dir):
                raise  # Re-raise exception if it's not due to directory already existing

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if the logger is called multiple times
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler (optional)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
    return logger

if __name__ == '__main__':
    # Example of how to use the logger setup
    # This part only runs if you execute logger.py directly (e.g., python3 src/core/logger.py)
    
    # Ensure 'logs' directory exists for this example
    if not os.path.exists("logs"):
        os.makedirs("logs")

    test_logger_file = setup_logger('TestLoggerFile', 'logs/test_logger.log', level=logging.DEBUG)
    test_logger_console = setup_logger('TestLoggerConsole', 'logs/test_console.log', level=logging.INFO, console_output=True)

    test_logger_file.debug("This is a debug message for the file logger.")
    test_logger_file.info("This is an info message for the file logger.")
    
    test_logger_console.info("This info message will go to file and console.")
    test_logger_console.warning("This warning message will go to file and console.")
    
    print(f"Test logs generated. Check 'logs/test_logger.log' and 'logs/test_console.log'.")