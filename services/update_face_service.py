# -*- coding: utf-8 -*-
import os
# from utils.config import loggers, TEMPERATURE, MAX_NEW_TOKENS
from utils.config import Config
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from interceptors.request_id_interceptor import request_id_ctx
from core.face_utils import FaceRecognitionManager
import shutil

loggers = Config.init_logging()

PRODUCTION=Config.ENVIRONMENT

service_logger = loggers['chatservice']

class UpdateFaceProcessor:
    def __init__(self, params: dict, context):
        """
        Initializes the ChatProcessor with parameters and context.
        :param params: Dictionary containing parameters for processing.
        :param context: Context for the service call.
        """
        service_logger.info("[INIT] UpdateFaceProcessor initialized with parameters.")
        self.params = params
        self.context = context
    
        self.query_id = self.params.get("QueryId")
        self.id=self.params.get("id")
        self.folder_path=self.params.get("folder_path")

        if not self.query_id:   
            service_logger.warning(f"[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)

    def process(self):
        service_logger.info(f"[INIT] QueryId={self.query_id},id={self.id} ,folder path={self.folder_path}")

        if not self.id:
            message= ErrorMessages.MISSING_PARAMS
            error_message = f"[MISSING PARAMS]{message} id not found."
            service_logger.warning(f"[PROCESS] {error_message}")
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)

        if not self.folder_path:
            message = ErrorMessages.MISSING_PARAMS
            error_message = f"[MISSING PARAMS] {message} folder path not given."
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR,message,error_message)
        
        if not self.query_id:
            message = ErrorMessages.MISSING_PARAMS
            error_message = f"[MISSING PARAMS] {message} QueryId not provided."
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.BAD_REQUEST,message,error_message)
        
        try:
            with Timer() as total_timer:
                dest_path = os.path.join("faces_db", self.id)
                try:
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                        service_logger.info(f"[UPDATE] Removed existing folder for {self.id} at {dest_path}")
                    shutil.copytree(self.folder_path, dest_path)
                    service_logger.info(f"[UPDATE] Copied new folder to {dest_path}")
                except Exception as copy_error:
                    error_message = f"[COPY ERROR] Failed to replace folder: {str(copy_error)}"
                    service_logger.error(error_message)
                    return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, ErrorMessages.INTERNAL_SERVER_ERROR, error_message)

                recognizer = FaceRecognitionManager()
                recognizer.update_person(self.id)

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