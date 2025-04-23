import sys
import os

sys.stdout = sys.stderr  # Redirect print() to Apache error log

# Dynamically set the correct path for the project base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f">>> WSGI file loaded from {BASE_DIR}")  # Should show up in Apache logs

from web_infer_yolo import app as application

