import sys
from pathlib import Path

# Add project root directory to sys.path so imports work seamlessly on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app

# Vercel Serverless Function looks for the ASGI 'app' object
