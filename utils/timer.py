import time
from utils.config import Config
loggers= Config.init_logging()

logging = loggers['server']

class Timer:
    """
    A context manager class for timing code execution.
    Usage:
        with Timer() as t:
            # code block to time
    Attributes:
        start (float): The start time in seconds since the epoch.
        end (float): The end time in seconds since the epoch.
        interval (float): The elapsed time in seconds.
    Methods:
        __enter__(): Starts the timer and logs the start time.
        __exit__(*args): Stops the timer, calculates the elapsed time, and logs the end time and duration.
    """
    def __enter__(self):
        self.start = time.time()
        self.interval = None 
        logging.info(f"Timer started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start))}")
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f"Timer ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end))}")
        logging.info(f"Elapsed time: {self.interval:.4f} seconds")
