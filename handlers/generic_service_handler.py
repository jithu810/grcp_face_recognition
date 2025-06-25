# handlers/generic_service_handler.py
import json
from generated import generic_service_pb2, generic_service_pb2_grpc
from utils.config import Config
loggers= Config.init_logging()

logger = loggers['handler']

class GenericServiceHandler(generic_service_pb2_grpc.GenericServiceServicer):
    """
    GenericServiceHandler implements the gRPC service defined in generic_service_pb2_grpc.GenericServiceServicer.
    This handler provides two main RPC methods:
    - ExecuteRemoteMethod: Executes a remote method specified by the client, passing parameters as JSON, and returns the result as a structured RpcResponse.
    - PingRemoteMethod: Simple health check endpoint that returns a success response to indicate the server is online.
    Attributes:
        registry: An object responsible for executing registered methods based on the incoming request.
    Methods:
        ExecuteRemoteMethod(request, context):
            Handles execution of a remote method. Parses method name and parameters from the request,
            delegates execution to the registry, and returns the result in an RpcResponse. Handles and logs exceptions.
        PingRemoteMethod(request, context):
            Responds to health check requests, confirming server availability. Handles and logs exceptions.
    """
    def __init__(self, registry):
        self.registry = registry

    def ExecuteRemoteMethod(self, request, context):
        """
        Handles the execution of a remote method as specified in the request.
        Args:
            request: The gRPC request object containing:
                - MethodName (str): The name of the method to execute.
                - MethodParamData (str): JSON-encoded string of parameters for the method.
            context: The gRPC context object for the request.
        Returns:
            generic_service_pb2.RpcResponse: The response object containing:
                - StatusCode (str): The status code of the execution.
                - StatusDescription (str): Description of the status.
                - Remarks (str): Additional remarks or error messages.
                - ResponseData (str): JSON-encoded string of the response data.
        Logs:
            - The method name and parameters received.
            - The execution result or any exceptions encountered.
        Raises:
            Returns a response with status code "500" and error details in case of exceptions.
        """
        logger.debug("Received ExecuteRemoteMethod request")
        try:
            method = request.MethodName
            param_data = json.loads(request.MethodParamData)

            logger.debug(f"Method: {method}")
            logger.debug(f"Param data: {param_data}")

            result = self.registry.execute(method, param_data, context)

            status_code = result.get("status_code", "500")
            status_description = result.get("status_description", "Internal Error")
            remarks = result.get("remarks", "")
            data = result.get("data", {})

            logger.debug(f"Execution result: {result}")

            return generic_service_pb2.RpcResponse(
                StatusCode=status_code,
                StatusDescription=status_description,
                Remarks=remarks,
                ResponseData=json.dumps(data)
            )
        
        except Exception as e:
            logger.exception("Error during ExecuteRemoteMethod")
            return generic_service_pb2.RpcResponse(
                StatusCode="500",
                StatusDescription="Internal Error",
                Remarks=str(e),
                ResponseData=""
            )

    def PingRemoteMethod(self, request, context):
        """
        Handles the PingRemoteMethod gRPC request.

        This method responds to a ping request to verify server availability.
        It returns a successful response with a "Hello World!" message if the server is online.
        In case of an exception, it logs the error and returns an internal error response.

        Args:
            request: The incoming gRPC request object.
            context: The gRPC context object for the request.

        Returns:
            generic_service_pb2.RpcResponse: The response message containing status code, description,
            remarks, and response data.
        """
        logger.debug("Received PingRemoteMethod request")
        try:
            return generic_service_pb2.RpcResponse(
                StatusCode="200",
                StatusDescription="OK",
                Remarks="Server is online",
                ResponseData=json.dumps({"ping": "Hello World!"})
            )
        except Exception as e:
            logger.exception("Error during PingRemoteMethod")
            return generic_service_pb2.RpcResponse(
                StatusCode="500",
                StatusDescription="Internal Error",
                Remarks=str(e),
                ResponseData=""
            )