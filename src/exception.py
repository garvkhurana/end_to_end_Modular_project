
import sys

class Customexception(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message_detail = f"Error occurred in script: {file_name} at line {exc_tb.tb_lineno} - {error_message}"
        return error_message_detail

    def __str__(self):
        return self.error_message
