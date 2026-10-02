from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import pandas as pd
import zipfile
from inference import (
    DEFAULT_MAX_UPLOAD_BYTES,
    MODEL_PACKAGE,
    MODEL_PACKAGE_SHA256,
    SERVICE_REVISION,
    SERVICE_BUILD,
    INPUT_FORMAT,
    run_screening,
)

app = FastAPI(
    title="SPAD V4 External ML Service",
    version="4.1.0-R1",
)

MAX_UPLOAD_BYTES = int(os.getenv("SPAD_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
API_KEY = os.getenv("SPAD_API_KEY", "").strip()
REQUIRE_API_KEY = os.getenv("SPAD_REQUIRE_API_KEY", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
HARD_MAX_UPLOAD_BYTES = 250 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 4 * 1024 * 1024
MAX_ENGINEERING_LIMITS_JSON_BYTES = 16 * 1024

if MAX_UPLOAD_BYTES <= 0 or MAX_UPLOAD_BYTES > HARD_MAX_UPLOAD_BYTES:
    raise RuntimeError(
        f"SPAD_MAX_UPLOAD_BYTES must be between 1 and {HARD_MAX_UPLOAD_BYTES} bytes."
    )
if REQUIRE_API_KEY and not API_KEY:
    raise RuntimeError(
        "SPAD_API_KEY must be configured when SPAD_REQUIRE_API_KEY is enabled."
    )


class _RequestTooLarge(Exception):
    pass


def _check_api_key(value: str | None) -> None:
    if not REQUIRE_API_KEY:
        return
    if not value or not secrets.compare_digest(value, API_KEY):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Invalid API key."},
        )


@app.middleware("http")
async def request_guard(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/run-screening":
        try:
            _check_api_key(request.headers.get("X-SPAD-API-Key"))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": {"code": "INVALID_CONTENT_LENGTH", "message": "Invalid Content-Length header."}},
                )
            if declared_length > MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": {
                            "code": "UPLOAD_TOO_LARGE",
                            "message": f"Request exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                        }
                    },
                )

        # Enforce the limit while the request body is being received as well.
        # This covers chunked/no-Content-Length requests before multipart parsing
        # can consume an arbitrarily large body.
        original_receive = request._receive
        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await original_receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                received_bytes += len(body)
                if received_bytes > MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
                    raise _RequestTooLarge
            return message

        request._receive = limited_receive
        try:
            return await call_next(request)
        except _RequestTooLarge:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": {
                        "code": "UPLOAD_TOO_LARGE",
                        "message": f"Request exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    }
                },
            )
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, Any]:
    module_b_loaded = MODEL_PACKAGE.get("module_b", {}).get("model") is not None
    module_a_loaded = bool(MODEL_PACKAGE.get("module_a", {}).get("models"))
    if not module_a_loaded or not module_b_loaded:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "message": "SPAD model package is not fully loaded."})
    return {
        "status": "ok",
        "serviceRevision": SERVICE_REVISION,
        "serviceBuild": SERVICE_BUILD,
        "randomForest": "loaded",
        "isolationForest": "loaded",
        "inputFormat": INPUT_FORMAT,
        "modelPackage": "SPAD_V4_Final_Model_Package.pkl",
        "modelPackageSha256": MODEL_PACKAGE_SHA256,
        "apiKeyRequired": REQUIRE_API_KEY,
    }


@app.post("/run-screening")
async def screening(
    file: UploadFile = File(..., description="ZIP containing exactly component_data.csv and transient_evidence.csv."),
    lotId: str = Form(...),
    engineeringLimits: str = Form("{}"),
    x_spad_api_key: str | None = Header(default=None, alias="X-SPAD-API-Key"),
):
    _check_api_key(x_spad_api_key)
    try:
        await file.seek(0)
        try:
            file.file.seek(0, 2)
            file_size = int(file.file.tell())
            file.file.seek(0)
        except Exception:
            file_size = None
        if file_size is not None and file_size > MAX_UPLOAD_BYTES:
            raise ValueError(f"Dataset exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
        if file_size == 0:
            raise ValueError("Dataset bundle is empty.")

        if len(engineeringLimits.encode("utf-8")) > MAX_ENGINEERING_LIMITS_JSON_BYTES:
            raise ValueError("engineeringLimits JSON exceeds the allowed size.")

        import json
        try:
            limits = json.loads(engineeringLimits)
        except json.JSONDecodeError as exc:
            raise ValueError("engineeringLimits must be valid JSON.") from exc

        result = run_screening(
            dataset=file.file,
            lot_id=lotId,
            engineering_limits=limits,
            file_name=file.filename or "SPAD_V4_WEB_INPUT_V2.zip",
            max_upload_bytes=MAX_UPLOAD_BYTES,
        )
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, zipfile.BadZipFile) as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_TELEMETRY_PAYLOAD", "message": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"code": "MODEL_INFERENCE_FAILURE", "message": "SPAD inference failed."}) from exc
