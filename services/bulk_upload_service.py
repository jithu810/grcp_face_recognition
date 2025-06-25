# -*- coding: utf-8 -*-
import os
import numpy as np
from utils.config import Config
from utils.messages import ErrorMessages, SuccessMessages
from utils.status_codes import HttpStatusCodes
from utils.timer import Timer
from utils.response_utils import response as _response
from interceptors.request_id_interceptor import request_id_ctx
from core.face_utils import FaceRecognitionManager

loggers = Config.init_logging()
PRODUCTION = Config.ENVIRONMENT
service_logger = loggers['chatservice']

class BulkFaceInsertProcessor:
    def __init__(self, params: dict, context):
        service_logger.info("[INIT] BulkFaceInsertProcessor initialized.")
        self.params = params
        self.context = context
        self.query_id = self.params.get("QueryId")
        self.bulk_folder = self.params.get("bulk_folder_path")

        if not self.query_id:
            service_logger.warning("[INIT] Missing QueryId.")
        request_id_ctx.set(self.query_id)

    def process(self):
        service_logger.info(f"[PROCESS] QueryId={self.query_id}, Bulk Folder={self.bulk_folder}")

        if not self.bulk_folder or not os.path.isdir(self.bulk_folder):
            msg = "Invalid or missing bulk_folder_path"
            service_logger.error(f"[ERROR] {msg}")
            return _response(HttpStatusCodes.BAD_REQUEST, ErrorMessages.MISSING_PARAMS, msg)

        try:
            with Timer() as total_timer:
                recognizer = FaceRecognitionManager()
                total_added = 0
                total_skipped = 0
                failed = []

                people = os.listdir(self.bulk_folder)
                service_logger.info(f"[INFO] Found {len(people)} entries in bulk folder.")

                for idx, person_name in enumerate(people, 1):
                    person_dir = os.path.join(self.bulk_folder, person_name)
                    if not os.path.isdir(person_dir):
                        service_logger.warning(f"[SKIP] Not a directory: {person_dir}")
                        continue

                    if person_name in recognizer.labels:
                        service_logger.info(f"[SKIP] Person '{person_name}' already in index. Skipping.")
                        total_skipped += 1
                        continue

                    service_logger.info(f"[PROCESSING] ({idx}/{len(people)}) Processing: {person_name}")
                    embeddings, labels = recognizer._extract_embeddings(person_name)

                    if not embeddings:
                        service_logger.warning(f"[FAILED] No valid face found for '{person_name}'")
                        failed.append(person_name)
                        continue

                    filtered_embeddings = []
                    for emb in embeddings:
                        emb = np.array([emb]).astype("float32")
                        if recognizer.index.ntotal > 0:
                            D, _ = recognizer.index.search(emb, 1)
                            if D[0][0] < 0.4:
                                service_logger.info(f"[SKIP] Face in {person_name} too similar (distance={D[0][0]:.4f})")
                                continue 
                        filtered_embeddings.append(emb[0])

                    if filtered_embeddings:
                        recognizer.index.add(np.array(filtered_embeddings).astype("float32"))
                        recognizer.labels.extend([person_name] * len(filtered_embeddings))
                        service_logger.info(f"[ADDED] {person_name}: {len(filtered_embeddings)} embeddings added.")
                        total_added += 1
                    else:
                        service_logger.warning(f"[SKIP] All embeddings for '{person_name}' too similar.")

                recognizer._save_index_and_labels()
                service_logger.info(f"[SUMMARY] Bulk insert completed. Added={total_added}, Skipped={total_skipped}, Failed={len(failed)}")

            return {
                "status_code": HttpStatusCodes.OK,
                "status_description": "OK",
                "remarks": SuccessMessages.PROCESSED_SUCCESSFULLY,
                "data": {
                    "QueryId": self.query_id,
                    "TotalAdded": total_added,
                    "TotalSkipped": total_skipped,
                    "Failed": failed,
                    "Time": f"{total_timer.interval:.2f}s"
                }
            }

        except Exception as e:
            error_message = f"[PROCESS ERROR] {str(e)}"
            service_logger.exception(error_message)
            return _response(HttpStatusCodes.INTERNAL_SERVER_ERROR, ErrorMessages.INTERNAL_SERVER_ERROR, error_message)
