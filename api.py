"""Unified FastAPI entry point for PharmaAssist text and image workflows."""

import os

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services import (
    MAX_IMAGE_BYTES,
    ChatResponse,
    ImageValidationError,
    RecognitionResult,
    RecognitionServiceError,
    ServiceError,
    load_medication_catalogue,
    process_message,
    recognize_medication,
)

app = FastAPI(
    title="PharmaAssist API",
    description=(
        "Multilingual pharmacy chat and medication-box image recognition "
        "with structured Gemini output."
    ),
    version="1.0.0",
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    gemini_model: str
    medication_catalogue_items: int


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                "message": error["msg"],
            }
        )
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "message": "Invalid request body.", "details": errors},
    )


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    try:
        catalogue_items = len(load_medication_catalogue())
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
                "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                "medication_catalogue_items": 0,
            },
        )
    return HealthResponse(
        status="ok",
        gemini_configured=bool(os.getenv("GEMINI_API_KEY")),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        medication_catalogue_items=catalogue_items,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return process_message(payload.message)


@app.post(
    "/recognize",
    response_model=RecognitionResult,
    responses={
        400: {"description": "Invalid image"},
        413: {"description": "Image too large"},
        429: {"description": "Gemini rate limit reached"},
        502: {"description": "Gemini provider error"},
        503: {"description": "Gemini or catalogue unavailable"},
    },
)
async def recognize(image: UploadFile = File(...)) -> RecognitionResult | JSONResponse:
    mime_type = image.content_type
    if mime_type not in {"image/jpeg", "image/png"}:
        await image.close()
        return JSONResponse(
            status_code=400,
            content={"error": "unsupported_image_type", "message": "Upload a JPEG or PNG image."},
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    await image.close()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": "image_too_large",
                "message": f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
            },
        )

    try:
        return await run_in_threadpool(recognize_medication, image_bytes, mime_type)
    except ImageValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_image", "message": str(exc)},
        )
    except RecognitionServiceError as exc:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.code, "message": str(exc)},
        )
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "catalogue_unavailable",
                "message": "The local medication catalogue could not be loaded.",
            },
        )
