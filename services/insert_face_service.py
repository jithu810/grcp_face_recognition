# -*- coding: utf-8 -*-
import os
import shutil
from utils.config import Config
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from interceptors.request_id_interceptor import request_id_ctx
from core.face_utils import FaceRecognitionManager
from core.validators import validate_image_file, validate_single_face_and_size  

loggers = Config.init_logging()
PRODUCTION = Config.ENVIRONMENT
service_logger = loggers['chatservice']

class InserrtFaceProcessor:
    def __init__(self, params: dict, context):
        service_logger.info("[INIT] InserrtFaceProcessor initialized with parameters.")
        self.params = params
        self.context = context

        self.query_id = self.params.get("QueryId")
        self.id = self.params.get("id")
        self.folder_path = self.params.get("folder_path")

        if not self.query_id:
            service_logger.warning("[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)

    def _copy_folder(self, src, dst):
        try:
            shutil.copytree(src, dst)
        except Exception as e:
            raise RuntimeError(f"Failed to copy folder: {str(e)}")

    def process(self):
        service_logger.info(f"[PROCESS] QueryId={self.query_id}, id={self.id}, folder_path={self.folder_path}")

        if not self.id or not self.folder_path or not self.query_id:
            missing = "id" if not self.id else "folder_path" if not self.folder_path else "QueryId"
            error_message = f"[MISSING PARAMS] {ErrorMessages.MISSING_PARAMS} {missing} is required."
            return _response(HttpStatusCodes.BAD_REQUEST, ErrorMessages.MISSING_PARAMS, error_message)

        try:
            with Timer() as total_timer:
                if not os.path.isdir(self.folder_path):
                    error_message = f"[FOLDER ERROR] Given path '{self.folder_path}' is not a valid directory."
                    service_logger.error(error_message)
                    return _response(HttpStatusCodes.BAD_REQUEST, ErrorMessages.INVALID_FOLDER_PATH, error_message)

                recognizer = FaceRecognitionManager()

                for filename in os.listdir(self.folder_path):
                    file_path = os.path.join(self.folder_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            service_logger.info(f"[VALIDATION] Checking file: {filename}")
                            validate_image_file(file_path)
                            validate_single_face_and_size(file_path, recognizer)
                            service_logger.info(f"[VALID] {filename} passed all checks.")
                        except ValueError as ve:
                            error_message = f"[IMAGE VALIDATION FAILED] {str(ve)} in file: {filename}"
                            service_logger.warning(error_message)
                            return _response(HttpStatusCodes.BAD_REQUEST, ErrorMessages.INVALID_IMAGE, error_message)

                dest_path = os.path.join("faces_db", self.id)
                if os.path.exists(dest_path):
                    error_message = f"[DUPLICATE] Person '{self.id}' already exists in faces_db."
                    service_logger.warning(error_message)
                    return _response(HttpStatusCodes.CONFLICT, ErrorMessages.RESOURCE_EXISTS, error_message)

                self._copy_folder(self.folder_path, dest_path)
                recognizer.add_new_person(self.id)

            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": {
                    "QueryId": self.query_id,
                    "PersonAdded": self.id,
                    "Time": f"{total_timer.interval:.2f}s"
                }
            }

        except Exception as e:
            message = ErrorMessages.INTERNAL_SERVER_ERROR
            error_message = f"[PROCESS ERROR] {message}: {str(e)}"
            service_logger.error(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)
