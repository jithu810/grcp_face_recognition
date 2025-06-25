# core/image_processor.py

import os
import cv2
from utils.messages import ErrorMessages
from utils.config import Config

loggers = Config.init_logging()
service_logger = loggers['chatservice']

IMG_DIM=200

class ImageProcessor:
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

    @staticmethod
    def validate_image_path(image_path):
        service_logger.info(f"[VALIDATE] Checking image path: {image_path}")

        if not os.path.exists(image_path):
            service_logger.error(f"[INVALID] Image path does not exist: {image_path}")
            raise ValueError(f"{ErrorMessages.MISSING_PARAMS} Image path does not exist: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ImageProcessor.SUPPORTED_EXTENSIONS:
            service_logger.error(f"[INVALID] Unsupported file format: {ext}")
            raise ValueError(
                f"{ErrorMessages.UNSUPPORTED_FILE_FORMAT} File format '{ext}' is not supported. "
                f"Supported: {ImageProcessor.SUPPORTED_EXTENSIONS}"
            )

        service_logger.info(f"[VALID] Image path and format are valid: {image_path}")

    @staticmethod
    def validate_image_content(image_path, min_dim=(IMG_DIM,IMG_DIM)):
        service_logger.info(f"[VALIDATE] Checking image content and dimensions: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            service_logger.error(f"[CORRUPT] Image could not be read: {image_path}")
            raise ValueError(f"{ErrorMessages.INTERNAL_SERVER_ERROR} Image could not be read or is corrupted.")

        height, width = image.shape[:2]
        service_logger.info(f"[DIMENSION] Image size: {width}x{height}")
        if width < min_dim[0] or height < min_dim[1]:
            service_logger.warning(
                f"[INVALID DIM] Image dimensions too small: {width}x{height}, expected ≥ {min_dim[0]}x{min_dim[1]}"
            )
            raise ValueError(
                f"{ErrorMessages.INVALID_IMAGE_DIMENSION} Image dimensions too small: {width}x{height}, "
                f"expected ≥ {min_dim[0]}x{min_dim[1]}."
            )

        service_logger.info(f"[VALID] Image passed dimension check: {image_path}")
        return image
