# utils/validators.py

import os
import cv2
from core.image_processor import ImageProcessor
from utils.messages import ErrorMessages
from utils.config import Config

MAX_IMAGE_SIZE_MB = 5
MIN_FACE_DIM = 0

loggers = Config.init_logging()
service_logger = loggers['chatservice']

def validate_image_file(image_path: str):
    """
    Checks file path, extension, size, and decodability.
    Raises ValueError with appropriate error message.
    """
    service_logger.info(f"[VALIDATE] Validating image file: {image_path}")

    if not os.path.exists(image_path):
        service_logger.error(f"[INVALID] Image path does not exist: {image_path}")
        raise ValueError("Image path does not exist.")

    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        service_logger.error(f"[INVALID] Unsupported image format: {ext}")
        raise ValueError("Invalid image format. Supported: JPG, PNG, BMP.")

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        service_logger.error(f"[INVALID] Image size {file_size_mb:.2f} MB exceeds {MAX_IMAGE_SIZE_MB} MB.")
        raise ValueError(f"Image size exceeds {MAX_IMAGE_SIZE_MB} MB.")

    try:
        ImageProcessor.validate_image_path(image_path)
        ImageProcessor.validate_image_content(image_path)
    except ValueError as ve:
        service_logger.error(f"[INVALID] Image decoding/format failed: {str(ve)}")
        raise

    service_logger.info(f"[VALID] Image passed file checks: {image_path}")


def validate_single_face_and_size(image_path: str, recognizer, min_face_size: int = MIN_FACE_DIM):
    """
    Ensures exactly one face is present and of at least given size.
    Raises ValueError with message on failure.
    """
    service_logger.info(f"[FACE CHECK] Validating face in image: {image_path}")

    image = cv2.imread(image_path)
    faces = recognizer.app.get(image)

    if not faces:
        service_logger.warning("[FACE CHECK] No face detected.")
        raise ValueError("No face detected in image.")

    if len(faces) > 1:
        service_logger.warning(f"[FACE CHECK] Multiple faces detected ({len(faces)}).")
        raise ValueError("Multiple faces detected in image.")

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        width, height = x2 - x1, y2 - y1
        service_logger.info(f"[FACE SIZE] Face found with dimensions: {width}x{height}")
        if width >= min_face_size and height >= min_face_size:
            service_logger.info(f"[FACE SIZE] Valid face dimension found: {width}x{height}")
            return

    service_logger.warning(f"[FACE TOO SMALL] No face meets min size {min_face_size}x{min_face_size}")
    raise ValueError(f"All detected faces too small (<{min_face_size}x{min_face_size}).")
