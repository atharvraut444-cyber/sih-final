import sys
import traceback
from pathlib import Path

# Add project root directory to sys.path so imports work seamlessly on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from main import app
except Exception as e:
    tb = traceback.format_exc()
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.api_route("/{path_name:path}", methods=["GET", "POST"])
    async def catch_all(path_name: str):
        return HTMLResponse(
            f"<h2>Vercel Serverless Initialization Error</h2><pre style='color:red; background:#111; padding:15px; border-radius:8px;'>{tb}</pre>",
            status_code=500,
        )
