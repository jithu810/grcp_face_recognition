
def response(status_code,message,error_message):
    """
    Constructs a standardized response dictionary.

    Args:
        status_code (int): The HTTP status code to include in the response.
        message (str): A description of the status.
        error_message (str): Additional remarks or error details.

    Returns:
        dict: A dictionary containing the status code, description, remarks, and an empty data field.
    """
    return {
        "status_code": status_code,
        "status_description": message,
        "remarks": error_message,
        "data": {}
    }