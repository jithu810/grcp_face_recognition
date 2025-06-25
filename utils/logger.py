import logging
import colorlog
import os
import sqlite3
from datetime import datetime
from interceptors.request_id_interceptor import request_id_ctx

class SQLiteHandler(logging.Handler):
    """
    A custom logging handler that writes log records to a SQLite database.
    Args:
        db_path (str): The file path to the SQLite database.
    Attributes:
        db_path (str): Path to the SQLite database file.
        conn (sqlite3.Connection): SQLite connection object.
    Methods:
        emit(record):
            Writes a log record to the SQLite database. Each log entry includes a timestamp,
            log level, logger name, and the log message. If an error occurs during logging,
            it prints an error message to the standard output.
    """
    def __init__(self, db_path):
        logging.Handler.__init__(self)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                logger TEXT,
                message TEXT
            )
        ''')
        self.conn.commit()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.conn.execute(
                "INSERT INTO logs (timestamp, level, logger, message) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), record.levelname, record.name, record.getMessage())
            )
            self.conn.commit()
        except Exception as e:
            print(f"Failed to log to SQLite: {e}")

class RequestIdFormatter(logging.Formatter):
    def format(self, record):
        try:
            request_id = request_id_ctx.get()
            if request_id:
                record.msg = f"[{request_id}] {record.msg}"
        except LookupError:
            pass
        return super().format(record)
    
class RequestIdColoredFormatter(colorlog.ColoredFormatter):
    def format(self, record):
        try:
            request_id = request_id_ctx.get()
            if request_id:
                record.msg = f"[{request_id}] {record.msg}"
        except LookupError:
            pass
        return super().format(record)

def configure_logging(project_root):
    LOG_DIR = "logs"
    os.makedirs(os.path.join(project_root, LOG_DIR), exist_ok=True)
    path= os.path.join(project_root, LOG_DIR)
    db_path = os.path.join(path, "application_logs.db")
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    colored_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )
    logger_names = [
        'server',
        'handler',
        'document_passport',
        'document_invoice',
        'document_match_amount',
        'document_summary',
        'handlerfile',
        'utils',
        'document_extractor',
        'pdf_loader',
        'minicpm_model',
        'llama_model',
        'chatservice',
        'chromadb',
        'nlp',
        'document_Query',
        'model_manager',
        'history'
    ]
    loggers = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        # SQLite handler
        sqlite_handler = SQLiteHandler(db_path)
        sqlite_handler.setFormatter(RequestIdFormatter(log_format))
        logger.addHandler(sqlite_handler)

        if os.getenv('ENVIRONMENT') != 'production':
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(colored_formatter)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

        loggers[name] = logger
    return loggers
