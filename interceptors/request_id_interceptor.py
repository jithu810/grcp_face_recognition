import grpc
import uuid
import contextvars

# Store request_id in a context var
request_id_ctx = contextvars.ContextVar("request_id", default=None)

class RequestIdInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        # Extract metadata from request
        metadata = dict(handler_call_details.invocation_metadata or [])
        # Use existing QueryId if present, else fallback to UUID
        request_id = metadata.get("QueryId") or str(uuid.uuid4())
        request_id_ctx.set(request_id)
        return continuation(handler_call_details)