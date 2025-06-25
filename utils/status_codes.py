# utils/status_codes.py

class HttpStatusCodes:
    """
    A collection of common HTTP status codes as string constants.

    Attributes:
        OK (str): Standard response for successful HTTP requests (200).
        BAD_REQUEST (str): The server could not understand the request due to invalid syntax (400).
        INTERNAL_SERVER_ERROR (str): The server encountered an unexpected condition (500).
        NOT_FOUND (str): The requested resource could not be found (404).
        UNAUTHORIZED (str): Authentication is required and has failed or has not yet been provided (401).
        FORBIDDEN (str): The client does not have access rights to the content (403).
        CONFLICT (str): The request could not be completed due to a conflict with the current state of the resource (409).
        SERVICE_UNAVAILABLE (str): The server is not ready to handle the request (503).
        TOO_MANY_REQUESTS (str): The user has sent too many requests in a given amount of time (429).
        CREATED (str): The request has succeeded and a new resource has been created as a result (201).
        ACCEPTED (str): The request has been accepted for processing, but the processing has not been completed (202).
        NO_CONTENT (str): The server successfully processed the request and is not returning any content (204).
        NOT_IMPLEMENTED (str): The server does not support the functionality required to fulfill the request (501).
        UNPROCESSABLE_ENTITY (str): The server understands the content type of the request entity, but was unable to process the contained instructions (422).
        GONE (str): The requested resource is no longer available and will not be available again (410).
        PRECONDITION_FAILED (str): The server does not meet one of the preconditions that the requester put on the request (412).
        REQUEST_TIMEOUT (str): The server timed out waiting for the request (408).
        METHOD_NOT_ALLOWED (str): The request method is known by the server but is not supported by the target resource (405).
        UNSUPPORTED_MEDIA_TYPE (str): The request entity has a media type which the server or resource does not support (415).
        IM_A_TEAPOT (str): The server refuses the attempt to brew coffee with a teapot (418).
    """
    OK = "200"
    BAD_REQUEST = "400"
    INTERNAL_SERVER_ERROR = "500"
    NOT_FOUND = "404"
    UNAUTHORIZED = "401"
    FORBIDDEN = "403"
    CONFLICT = "409"
    SERVICE_UNAVAILABLE = "503"
    TOO_MANY_REQUESTS = "429"
    CREATED = "201"
    ACCEPTED = "202"
    NO_CONTENT = "204"
    NOT_IMPLEMENTED = "501"
    UNPROCESSABLE_ENTITY = "422"
    GONE = "410"
    PRECONDITION_FAILED = "412"
    REQUEST_TIMEOUT = "408"
    METHOD_NOT_ALLOWED = "405"
    UNSUPPORTED_MEDIA_TYPE = "415"
    IM_A_TEAPOT = "418"  
    
    
