from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from inference import MODEL_PACKAGE, run_screening

app = FastAPI(
    title="SPAD V4 External ML Service",
    version="4.0.0",
)

@app.get("/")
def serve_web_app():
    return FileResponse("static/index.html")

MAX_UPLOAD_BYTES = int(os.getenv("SPAD_MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "randomForest": "loaded" if MODEL_PACKAGE.get("module_b", {}).get("model") is not None else "unavailable",
        "isolationForest": "loaded" if MODEL_PACKAGE.get("module_a", {}).get("models") else "unavailable",
    }


@app.post("/run-screening")
async def screening(
    file: UploadFile = File(...),
    lotId: str = Form(...),
    engineeringLimits: str = Form("{}"),
):
    try:
        import json
        dataset_bytes = await file.read()
        if not dataset_bytes:
            raise ValueError("Dataset file is empty.")
        if len(dataset_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Dataset exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
        limits = json.loads(engineeringLimits)
        return JSONResponse(
            status_code=200,
            content=run_screening(
                dataset_bytes=dataset_bytes,
                lot_id=lotId,
                engineering_limits=limits,
                file_name=file.filename or "uploaded_dataset.zip",
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_TELEMETRY_PAYLOAD", "message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_TELEMETRY_PAYLOAD", "message": str(exc)}) from exc
    except Exception as exc:
        # Do not expose stack traces or local filesystem details.
        raise HTTPException(status_code=500, detail={"code": "MODEL_INFERENCE_FAILURE", "message": "SPAD inference failed."}) from exc
