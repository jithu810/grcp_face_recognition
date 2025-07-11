# -*- coding: utf-8 -*-
import os
import json
from utils.config import Config
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from interceptors.request_id_interceptor import request_id_ctx
from core.face_utils import FaceRecognitionManager
import json
import numpy as np

loggers = Config.init_logging()
PRODUCTION = Config.ENVIRONMENT
service_logger = loggers['chatservice']

class InferFaceProcessor:
    def __init__(self, params: dict, context):
        service_logger.info("[INIT] InferFaceProcessor initialized with parameters.")
        self.params = params
        self.context = context

        self.query_id = self.params.get("QueryId")
        self.image_path = self.params.get("image_path")
        self.target_id = self.params.get("target_id") 

        if not self.query_id:
            service_logger.warning("[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)

    def process(self):
        service_logger.info(f"[PROCESS] QueryId={self.query_id}, image_path={self.image_path}")

        if not self.query_id or not self.image_path:
            missing = "QueryId" if not self.query_id else "image_path"
            message = ErrorMessages.MISSING_PARAMS
            error_message = f"[MISSING PARAMS] {message} {missing} is required."
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.BAD_REQUEST, message, error_message)

        try:
            with Timer() as total_timer:

                # Validate image using common validators
                validate_image_file(self.image_path)

                recognizer = FaceRecognitionManager()
                result,score = recognizer.recognize_face(self.image_path,k=3)
                print("\n Predicted Person:",result)
                print(" Confidence Score:", round(score, 4) if score is not None else "N/A")
                data = {
                    "result": int(result) if isinstance(result, (np.integer, np.int_)) else result,
                    "conf": float(round(float(score), 4)) if score is not None else "N/A"
                }

            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": {
                    "QueryId": self.query_id,
                    "result": data,
                    "Time": f"{total_timer.interval:.2f}s"
                }
            }

        except ValueError as ve:
            error_message = f"[VALIDATION ERROR] {str(ve)}"
            service_logger.warning(error_message)
            return _response(HttpStatusCodes.BAD_REQUEST, ErrorMessages.INVALID_IMAGE, error_message)

        except Exception as e:
            message = ErrorMessages.INTERNAL_SERVER_ERROR
            error_message = f"[PROCESS ERROR] {message}: {str(e)}"
            service_logger.error(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, message, error_message)
