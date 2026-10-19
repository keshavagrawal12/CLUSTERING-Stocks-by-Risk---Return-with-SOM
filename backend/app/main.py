"""
FastAPI entry point.   Run from backend/:   uvicorn app.main:app --reload --port 8000

On start-up we try to load the CSVs from backend/data/raw/. If they're not there the app
still starts and every page shows an empty state asking for an upload.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.state import STATE
from app.data.loader import load_from_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ds = load_from_dir()
        STATE.set_dataset(ds)
        if ds is None:
            log.info("No CSVs in data/raw/ - starting with an empty state")
    except Exception as exc:  # bad files shouldn't stop the server; show the error in the UI
        STATE.load_error = str(exc)
        log.warning("Could not load data/raw/: %s", exc)
    yield


app = FastAPI(title="Beyond Sector Labels - SOM API", version="1.0.0", lifespan=lifespan)

# The Vite dev server proxies /api, but allow direct calls from localhost too.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
