from pathlib import Path
from utils import constants as C
from utils.logger import configure_logging

class Config:
    """
    Config class for managing application-wide constants and project paths.
    Attributes:
        USE_SERVER (bool): Flag indicating whether to use the server.
        API_URL_MINICPM (str): API endpoint for MiniCPM.
        API_URL_LLAMA (str): API endpoint for Llama.
        HEADER_TOKEN (str): Token used for API authentication.
        LOCAL_MODEL_NAME (str): Name of the local model to use.
        QUANTIZED (bool): Indicates if the model is quantized.
        TEMPERATURE (float): Sampling temperature for model inference.
        MAX_NEW_TOKENS (int): Maximum number of new tokens to generate.
        ENVIRONMENT (str): Current environment (e.g., 'development', 'production').
        PORT (int): Port number for the application.
        API_URL_MINICPM_BASE (str): Base API URL for MiniCPM.
        PROJECT_ROOT (Path): Root directory of the project.
        OUTPUT_DIR (Path): Directory for storing extracted content.
    Methods:
        init_directories():
            Creates the output directory if it does not exist.
        init_logging():
            Initializes and returns the logging configuration for the project.
    """
    # Load constants
    USE_SERVER = C.USE_SERVER
    HEADER_TOKEN = C.HEADER_TOKEN
    ENVIRONMENT = C.ENVIRONMENT
    PORT = C.PORT

    # Project paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "content" / "extracted"

    @classmethod
    def init_directories(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def init_logging(cls):
        return configure_logging(cls.PROJECT_ROOT)
