import os

def print_clickable_link(file_path, line_number):
    abs_path = os.path.abspath(file_path)
    print(f"file://{abs_path}#L{line_number}")