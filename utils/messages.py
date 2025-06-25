class ErrorMessages:
    """
    A collection of standardized error message constants for use throughout the application.
    """

    BAD_REQUEST="BAD REQUEST"
    MISSING_PARAMS="Missing params"
    RESOURCE_EXISTS="id already exixts in db"
    INTERNAL_SERVER_ERROR="internal server error"
    INVALID_FOLDER_PATH = "Invalid folder path provided."
    INVALID_IMAGE = "One or more images failed validation checks."
    INVALID_IMAGE_DIMENSION="Image dimensions are too small."
    
class SuccessMessages:
    """
    A collection of success message templates used throughout the application.

    Attributes:
        PROCESSED_SUCCESSFULLY (str): Message indicating successful processing.
        VALIDATION_SUCCESS (str): Message indicating validation has completed successfully.
        TIME_CAL_SUCCESS (str): Message indicating the request was processed, including the duration in seconds.
        DIR_CLEARED (str): Message indicating the output directory has been cleared.
        DIR_CREATED (str): Message indicating a directory was created, with the directory path.
        NUMBER_NORMALIZED (str): Message indicating a number was normalized, showing the original and normalized values.
        AMOUNT_EXTRACTED (str): Message indicating a value was extracted.
        AMOUNT_COMPARISON (str): Message comparing extracted and matched values.
        DIR_CLEARED_AND_REMOVED (str): Message indicating the processed folder was deleted.
    """
    PROCESSED_SUCCESSFULLY="Successfully processed"
