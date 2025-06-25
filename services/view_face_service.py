from core.face_utils import FaceRecognitionManager
from utils.config import Config
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from interceptors.request_id_interceptor import request_id_ctx

loggers = Config.init_logging()
service_logger = loggers['chatservice']

class ListFaceDatabaseProcessor:
    def __init__(self, params: dict, context):
        self.params = params
        self.context = context
        self.query_id = self.params.get("QueryId")
        if not self.query_id:
            service_logger.warning("[INIT] Missing QueryId parameter.")
        request_id_ctx.set(self.query_id)

    def process(self):
        try:
            with Timer() as total_timer:
                recognizer = FaceRecognitionManager()
                summary = recognizer.get_database_summary()

            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": {
                    "QueryId": self.query_id,
                    "TotalEmbeddings": summary["total_embeddings"],
                    "LabelSummary": summary["label_summary"],
                    "Time": f"{total_timer.interval:.2f}s"
                }
            }

        except Exception as e:
            error_message = f"[PROCESS ERROR] {ErrorMessages.INTERNAL_SERVER_ERROR}: {str(e)}"
            service_logger.error(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, ErrorMessages.INTERNAL_SERVER_ERROR, error_message)
