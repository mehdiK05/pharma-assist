"""FastAPI service for medication-box recognition."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services import (
    MAX_IMAGE_BYTES,
    GeminiServiceError,
    ImageValidationError,
    RecognitionResult,
    load_catalogue,
    recognize_medication,
)


load_dotenv()

app = FastAPI(
    title="Medication Box Recognition API",
    version="0.1.0",
    description=(
        "Extracts visible medication-box text with Gemini and ranks a local catalogue. "
        "Scores are not probabilities and results are not medical advice."
    ),
)


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    gemini_model: str
    catalogue_items: int


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        count = len(load_catalogue())
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
                "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
                "catalogue_items": 0,
            },
        )
    return HealthResponse(
        status="ok",
        gemini_configured=bool(os.getenv("GEMINI_API_KEY")),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        catalogue_items=count,
    )


@app.post(
    "/recognize",
    response_model=RecognitionResult,
    responses={
        400: {"description": "Invalid image"},
        413: {"description": "Image too large"},
        429: {"description": "Gemini rate limit reached"},
        502: {"description": "Gemini error"},
        503: {"description": "Gemini not configured"},
    },
)
async def recognize(image: UploadFile = File(...)) -> RecognitionResult | JSONResponse:
    if image.content_type not in {"image/jpeg", "image/png"}:
        return error_response(400, "unsupported_image_type", "Upload a JPEG or PNG image.")

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    await image.close()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return error_response(
            413,
            "image_too_large",
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        return await run_in_threadpool(
            recognize_medication, image_bytes, image.content_type
        )
    except ImageValidationError as exc:
        return error_response(400, "invalid_image", str(exc))
    except GeminiServiceError as exc:
        return error_response(exc.http_status, exc.code, str(exc))
    except RuntimeError:
        return error_response(
            503,
            "catalogue_unavailable",
            "The local medication catalogue could not be loaded.",
        )
