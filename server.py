import grpc
from concurrent import futures
from generated import generic_service_pb2_grpc
from handlers.generic_service_handler import GenericServiceHandler
from handlers.service_registry import ServiceRegistry
from utils.config import Config
import sys
import time
import signal
import os

loggers = Config.init_logging()
default_port = 50058
PORT=Config.PORT

import warnings
warnings.simplefilter("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  

# Initialize logger
if PORT is None:
    PORT = default_port
    server_logger = loggers['server']
    server_logger.warning(f"[PORT WARNING] PORT is not set, using default port {default_port}.")

server_logger = loggers['server']

def serve():
    """
    Starts and manages the lifecycle of a gRPC server.
    This function initializes a gRPC server with a thread pool executor, registers the generic service handler,
    and binds the server to a specified port. It handles invalid port values by falling back to a default port.
    Graceful shutdown is supported via signal handlers for SIGINT and SIGTERM, ensuring proper server stop and logging.
    Server start and stop events, as well as errors, are logged for monitoring and debugging purposes.
    Raises:
        SystemExit: If a fatal server error occurs or upon receiving a shutdown signal.
    """
    registry = ServiceRegistry()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) 
    generic_service_pb2_grpc.add_GenericServiceServicer_to_server(
        GenericServiceHandler(registry),server
    )      

    try: 
        port = int(PORT)
    except ValueError:
        server_logger.error(f"[PORT ERROR] Invalid port number: {port}. Using default port {default_port}.") 
        port = default_port 
    server.add_insecure_port(f'[::]:{port}')

    def graceful_shutdown(*args):
        stop_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        server_logger.info(f"[STOP TIME] Shutting down server at {stop_time} ......")
        server.stop(0)
        sys.exit(0) 
        
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    try:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        server.start()
        server_logger.info(f"[START TIME] Server started at {start_time} ")
        server_logger.info(f"[SERVER STARTED] server running on port {port}....")
        server.wait_for_termination()
    except Exception as e:
        server_logger.critical(f"[SERVER ERROR] Fatal server error: {e}")
        sys.exit(1)
    finally: 
        server_logger.info("[SERVER STOPPED] Server has been stopped.")

if __name__ == '__main__':
    serve()