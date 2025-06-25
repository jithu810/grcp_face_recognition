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

class InserrtFaceProcessor:
    def __init__(self, params: dict, context):
        """
        Initializes the ChatProcessor with parameters and context.
        :param params: Dictionary containing parameters for processing.
        :param context: Context for the service call.
        """
        service_logger.info("[INIT] InserrtFaceProcessor initialized with parameters.")
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
                # Copy folder to faces_db/<id>
                dest_path = os.path.join("faces_db", self.id)
                if not os.path.exists(dest_path):
                    try:
                        shutil.copytree(self.folder_path, dest_path)
                    except Exception as copy_error:
                        error_message = f"[COPY ERROR] Failed to copy folder: {str(copy_error)}"
                        service_logger.error(error_message)
                        return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, ErrorMessages.INTERNAL_SERVER_ERROR, error_message)
                else:
                    error_message = f"[DUPLICATE] Person '{self.id}' already exists in faces_db."
                    service_logger.warning(error_message)
                    return _response(HttpStatusCodes.CONFLICT, ErrorMessages.RESOURCE_EXISTS, error_message)

                recognizer = FaceRecognitionManager()
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