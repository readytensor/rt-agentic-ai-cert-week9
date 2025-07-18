import sys
import os

# Add the root directory to sys.path so 'code' is importable
this_file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_file_dir, ".."))
path_to_code = os.path.abspath(os.path.join(root_dir, "code"))
sys.path.insert(0, path_to_code)
