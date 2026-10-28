"""Entry point. Run from the sentrygun folder:
    python3 -m pip install -r requirements.txt
    python3 run.py
Dashboard: http://127.0.0.1:8000
"""

import uvicorn

from app.core import Config
from app.server import app

if __name__ == "__main__":
    cfg = Config.from_env()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
