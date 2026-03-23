import sys
import traceback
from typing import Optional


class RagAssistantException(Exception):
    def __init__(self, error_message, error_details: Optional[object] = None):
        norm_msg = str(error_message)

        exc_type = exc_value = exc_tb = None

        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif hasattr(error_details, "exc_info"):
            exc_type, exc_value, exc_tb = error_details.exc_info()
        elif isinstance(error_details, BaseException):
            exc_type, exc_value, exc_tb = (
                type(error_details),
                error_details,
                error_details.__traceback__,
            )
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        if exc_type and exc_tb:
            self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return (
            f"RagAssistantException(file={self.file_name!r}, "
            f"line={self.lineno}, message={self.error_message!r})"
        )