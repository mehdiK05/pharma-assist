"""Gemini interpretation and pharmacy CSV lookup."""

from __future__ import annotations

import csv
import io
import os
import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rapidfuzz import fuzz

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PHARMACY_FILE = BASE_DIR / "pharmacy.csv"
CATALOGUE_FILE = BASE_DIR / "medications.csv"
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}


class Intent(str, Enum):
    CHECK_AVAILABILITY = "check_availability"
    ASK_PRICE = "ask_price"
    ASK_LOCATION = "ask_location"
    ASK_HOURS = "ask_hours"
    ASK_CONTACT = "ask_contact"
    REQUEST_MEDICATION = "request_medication"
    OTHER = "other"


class Language(str, Enum):
    DARIJA_ARABIC = "darija_arabic"
    DARIJA_LATIN = "darija_latin"
    ARABIC = "arabic"
    FRENCH = "french"
    MIXED = "mixed"
    OTHER = "other"


class Interpretation(BaseModel):
    intent: Intent = Field(description="The user's single main intent")
    medication_name: str | None = Field(
        default=None, description="Medication name only when explicitly stated"
    )
    strength: str | None = Field(
        default=None, description="Strength such as 1000 mg only when explicitly stated"
    )
    quantity: float | None = Field(
        default=None, description="Requested numeric quantity only when explicitly stated"
    )
    unit: str | None = Field(
        default=None, description="Quantity unit such as boxes only when explicitly stated"
    )
    language: Language = Field(description="Language/script used in the request")
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information required to handle the request but not supplied",
    )
    clarification_question: str | None = Field(
        default=None,
        description="A short clarification in the user's language, or null if none is needed",
    )


class ChatResponse(BaseModel):
    original_text: str
    interpretation: Interpretation
    pharmacy_data: dict[str, Any] | None = None
    clarification_question: str | None = None
    answer: str


class ServiceError(Exception):
    """An error safe for the API layer to translate into an HTTP response."""

    def __init__(self, message: str, status_code: int = 502, code: str = "provider_error"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


SYSTEM_PROMPT = """You classify and extract information from short pharmacy customer requests.
The request can be Moroccan Darija in Arabic or Latin/Arabizi script, Arabic, French, or mixed.

Return only the requested structured data. Never diagnose, recommend a dose, prescribe, suggest
a substitute, or claim that a medicine is available. Availability is checked later from local data.

Rules:
- Extract only facts explicitly present in the request. Never guess missing values.
- Preserve the written medication name. Normalize strength spacing only when obvious.
- check_availability means the user explicitly asks whether an item is in stock.
- ask_price, ask_location, ask_hours, and ask_contact are direct information questions.
- request_medication means the user wants a medicine but does not explicitly ask stock or price.
- Use other for greetings, medical advice, symptoms, diagnosis, dosage, prescriptions, substitutions,
  or anything outside these operational pharmacy intents.
- missing_information contains only fields needed to understand/fulfil the request.
- If the request is ambiguous or lacks essential information, ask one short clarification question
  in the same language/style as the user. Otherwise clarification_question must be null.
- A generic request such as "Bghit dwa" lacks medication_name and needs clarification.
"""


def interpret_message(message: str) -> Interpretation:
    """Make exactly one Gemini structured-output call."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    if not api_key:
        raise ServiceError(
            "GEMINI_API_KEY is not configured. Copy .env.example to .env and add your key.",
            status_code=503,
            code="configuration_error",
        )

    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
        interaction = client.interactions.create(
            model=model,
            system_instruction=SYSTEM_PROMPT,
            input=message,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": Interpretation.model_json_schema(),
            },
            generation_config={"max_output_tokens": 500},
            store=False,
        )
        if interaction.output_text:
            return Interpretation.model_validate_json(interaction.output_text)
        raise ServiceError(
            "Gemini returned no structured result.", code="invalid_provider_response"
        )
    except ServiceError:
        raise
    except errors.APIError as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status == 429:
            raise ServiceError(
                "Gemini rate limit or quota reached. Wait and try again.",
                status_code=429,
                code="rate_limit",
            ) from exc
        if status in {401, 403}:
            raise ServiceError(
                "Gemini rejected the API key or project access.",
                status_code=502,
                code="provider_authentication_error",
            ) from exc
        if status == 400:
            raise ServiceError(
                "Gemini rejected the model or request configuration. Check GEMINI_MODEL.",
                status_code=502,
                code="provider_request_error",
            ) from exc
        raise ServiceError(
            "Gemini is temporarily unavailable; no AI response was generated.",
            code="provider_error",
        ) from exc
    except (ValidationError, ValueError, TypeError) as exc:
        raise ServiceError(
            "Gemini returned invalid structured data; no result was fabricated.",
            code="invalid_provider_response",
        ) from exc
    except (httpx.NetworkError, httpx.TimeoutException, ConnectionError, TimeoutError, OSError) as exc:
        raise ServiceError(
            "Could not reach Gemini. Check the network and try again.",
            status_code=503,
            code="network_error",
        ) from exc


def _normalize(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)


def _strength_matches(requested: str, catalog_value: str) -> bool:
    wanted = _normalize(requested)
    actual = _normalize(catalog_value)
    if wanted == actual:
        return True
    # Darija requests often say "Doliprane 1000" without writing the implied mg.
    return wanted.isdigit() and actual.startswith(wanted) and actual[len(wanted) :] in {"mg", "g"}


def _read_rows() -> list[dict[str, str]]:
    try:
        with PHARMACY_FILE.open(encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except (OSError, csv.Error) as exc:
        raise ServiceError(
            "The pharmacy data could not be read.",
            status_code=500,
            code="data_error",
        ) from exc


def lookup_pharmacy(interpretation: Interpretation) -> dict[str, Any] | None:
    """Look up deterministic pharmacy data after interpretation."""
    rows = _read_rows()
    if not rows:
        return None

    if interpretation.intent in {Intent.ASK_LOCATION, Intent.ASK_HOURS, Intent.ASK_CONTACT}:
        row = rows[0]
        fields = {
            Intent.ASK_LOCATION: {"address": row["address"]},
            Intent.ASK_HOURS: {"hours": row["hours"]},
            Intent.ASK_CONTACT: {"phone": row["phone"]},
        }[interpretation.intent]
        return {
            "source": "pharmacy catalog",
            "pharmacy_name": row["pharmacy_name"],
            **fields,
        }

    medication_intents = {
        Intent.CHECK_AVAILABILITY,
        Intent.ASK_PRICE,
        Intent.REQUEST_MEDICATION,
    }
    if interpretation.intent not in medication_intents or not interpretation.medication_name:
        return None

    wanted_name = _normalize(interpretation.medication_name)
    matches = [row for row in rows if _normalize(row["medication_name"]) == wanted_name]
    if interpretation.strength:
        matches = [
            row for row in matches if _strength_matches(interpretation.strength, row["strength"])
        ]

    if not matches:
        return {
            "source": "pharmacy catalog",
            "matched": False,
            "message": "No exact catalog match was found.",
        }

    public_matches = []
    for row in matches:
        item: dict[str, Any] = {
            "medication_name": row["medication_name"],
            "strength": row["strength"],
            "unit": row["unit"],
        }
        if interpretation.intent in {Intent.CHECK_AVAILABILITY, Intent.REQUEST_MEDICATION}:
            item["in_stock"] = int(row["stock"]) > 0
            item["stock_count"] = int(row["stock"])
        if interpretation.intent == Intent.ASK_PRICE:
            item["price_mad"] = float(row["price_mad"])
        public_matches.append(item)

    return {
        "source": "pharmacy catalog",
        "matched": True,
        "pharmacy_name": matches[0]["pharmacy_name"],
        "items": public_matches,
    }


def build_answer(
    interpretation: Interpretation, pharmacy_data: dict[str, Any] | None
) -> str:
    """Build a concise chat reply from validated interpretation and catalogue data."""
    if interpretation.clarification_question:
        return interpretation.clarification_question

    language = interpretation.language
    is_arabic = language in {Language.ARABIC, Language.DARIJA_ARABIC}
    is_french = language == Language.FRENCH

    if interpretation.intent == Intent.OTHER:
        if is_arabic:
            return "يمكنني مساعدتك في توفر الأدوية، الأثمنة، العنوان، أوقات العمل ورقم الهاتف."
        if is_french:
            return "Je peux vous aider avec la disponibilité, les prix, l’adresse, les horaires et le contact de la pharmacie."
        return "N9dar n3awnk f disponibilité, taman, l'adresse, les horaires w numéro dyal pharmacie."

    if not pharmacy_data:
        if is_arabic:
            return "أحتاج إلى معلومات إضافية لفهم طلبك."
        if is_french:
            return "J’ai besoin de plus d’informations pour traiter votre demande."
        return "Khasni chi ma3loumat zayda bach nfhem talab dyalk."

    if pharmacy_data.get("matched") is False:
        name = interpretation.medication_name or "ce médicament"
        if is_arabic:
            return f"لم أجد تطابقاً دقيقاً لـ {name} في الكتالوج."
        if is_french:
            return f"Je n’ai trouvé aucune correspondance exacte pour {name} dans le catalogue."
        return f"Ma l9itch correspondance exacte l {name} f catalogue."

    intent = interpretation.intent
    if intent == Intent.ASK_LOCATION:
        return f"📍 **{pharmacy_data['pharmacy_name']}**\n\n{pharmacy_data['address']}"
    if intent == Intent.ASK_HOURS:
        return f"🕒 **Horaires**\n\n{pharmacy_data['hours']}"
    if intent == Intent.ASK_CONTACT:
        return f"☎️ **{pharmacy_data['pharmacy_name']}**\n\n{pharmacy_data['phone']}"

    lines: list[str] = []
    for item in pharmacy_data.get("items", []):
        label = f"**{item['medication_name']} {item['strength']}**"
        if intent == Intent.ASK_PRICE:
            lines.append(f"{label} — {item['price_mad']:.2f} MAD")
        else:
            state = "In stock" if item.get("in_stock") else "Out of stock"
            lines.append(f"{label} — {state} ({item.get('stock_count', 0)} {item['unit']})")
    return "\n\n".join(lines)


def process_message(message: str) -> ChatResponse:
    interpretation = interpret_message(message)
    pharmacy_data = lookup_pharmacy(interpretation)
    return ChatResponse(
        original_text=message,
        interpretation=interpretation,
        pharmacy_data=pharmacy_data,
        clarification_question=interpretation.clarification_question,
        answer=build_answer(interpretation, pharmacy_data),
    )


# Medication-box image recognition


class ImageValidationError(ValueError):
    """The uploaded file is not a supported, decodable image."""


class RecognitionServiceError(RuntimeError):
    """A Gemini vision call failed without producing recognition output."""

    def __init__(self, code: str, message: str, http_status: int = 502):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class MedicationExtraction(BaseModel):
    """Information visibly supported by an uploaded medication package."""

    model_config = ConfigDict(extra="ignore")

    brand_name: str | None = Field(default=None, description="Visible brand name, or null")
    active_ingredient: str | None = Field(
        default=None, description="Visible generic name or active ingredient, or null"
    )
    strength: str | None = Field(
        default=None, description="Visible strength including units, or null"
    )
    dosage_form: str | None = Field(default=None, description="Visible dosage form, or null")
    package_quantity: str | None = Field(
        default=None, description="Visible package quantity, or null"
    )
    visible_text: str | None = Field(
        default=None,
        description="Faithful transcription of readable Arabic and French text, or null",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields that are absent, hidden, or unreadable",
    )


class NormalizedMedication(BaseModel):
    brand_name: str | None = None
    active_ingredient: str | None = None
    strength: str | None = None
    dosage_form: str | None = None
    package_quantity: str | None = None


class MatchCandidate(BaseModel):
    catalogue_id: str
    brand_name: str
    brand_name_ar: str | None = None
    active_ingredient: str
    active_ingredient_ar: str | None = None
    strength: str
    dosage_form: str
    package_quantity: str
    brand_score: float
    ingredient_score: float | None = None
    strength_score: float | None = None
    ranking_score: float
    conflicts: list[str] = Field(default_factory=list)


class RecognitionResult(BaseModel):
    status: Literal["candidate_found", "needs_confirmation", "no_match"]
    model: str
    extracted: MedicationExtraction
    normalized: NormalizedMedication
    candidates: list[MatchCandidate]
    message: str
    ranking_note: str = "Scores rank catalogue similarity; they are not probabilities."
    medical_disclaimer: str = (
        "Identification aid only. Do not use this result for treatment, dosage, "
        "or substitution decisions."
    )


IMAGE_EXTRACTION_PROMPT = """
Inspect this single photo of medication packaging. Extract only information that is
actually visible and readable on the package, including Arabic and French text.

Rules:
- Never infer, complete, translate into a guessed product, or use outside knowledge.
- Preserve the visible spelling. If a field is absent, hidden, blurry, or uncertain,
  return null and include its field name in missing_fields.
- active_ingredient means the generic name or active ingredient.
- strength must include every visible number and unit that belongs to the medicine.
- visible_text is a faithful compact transcription of all readable package text.
- Do not provide medical advice, dosage instructions, or substitutions.
""".strip()

IMAGE_EXTRACTION_FIELDS = (
    "brand_name",
    "active_ingredient",
    "strength",
    "dosage_form",
    "package_quantity",
    "visible_text",
)


def validate_image(image_bytes: bytes, declared_mime_type: str | None) -> str:
    """Validate byte size, MIME type, decoding, dimensions, and actual format."""
    if not image_bytes:
        raise ImageValidationError("The uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    if declared_mime_type not in ALLOWED_IMAGE_TYPES:
        raise ImageValidationError("Only JPEG and PNG images are accepted.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            actual_format = image.format
            width, height = image.size
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The file could not be decoded as a valid image.") from exc

    if actual_format != ALLOWED_IMAGE_TYPES[declared_mime_type]:
        raise ImageValidationError("The file content does not match its declared image type.")
    if width < 20 or height < 20:
        raise ImageValidationError("Image dimensions are too small to inspect.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError("Image dimensions are too large (maximum 25 megapixels).")
    return declared_mime_type


def _clean_extracted_value(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text or text.lower() in {"null", "none", "unknown", "unreadable", "n/a"}:
        return None
    return text


def clean_extraction(extraction: MedicationExtraction) -> MedicationExtraction:
    values = {
        field: _clean_extracted_value(getattr(extraction, field))
        for field in IMAGE_EXTRACTION_FIELDS
    }
    missing = [field for field, value in values.items() if value is None]
    return MedicationExtraction(**values, missing_fields=missing)


def extract_image_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[MedicationExtraction, str]:
    """Make one Gemini vision call and validate its structured response."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    if not key:
        raise RecognitionServiceError(
            "gemini_not_configured",
            "GEMINI_API_KEY is not configured on the API server.",
            503,
        )

    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=selected_model,
            contents=[IMAGE_EXTRACTION_PROMPT, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=MedicationExtraction.model_json_schema(),
                temperature=0,
                max_output_tokens=2048,
            ),
        )
        if not response.text:
            raise RecognitionServiceError(
                "empty_model_response",
                "Gemini returned no text. Try a clearer image.",
            )
        extraction = MedicationExtraction.model_validate_json(response.text)
        return clean_extraction(extraction), selected_model
    except RecognitionServiceError:
        raise
    except ValidationError as exc:
        raise RecognitionServiceError(
            "invalid_model_response",
            "Gemini returned data that did not match the required schema.",
        ) from exc
    except errors.APIError as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status == 429:
            raise RecognitionServiceError(
                "gemini_rate_limited",
                "Gemini rate limit reached. Wait briefly and try again.",
                429,
            ) from exc
        if status in {400, 401, 403}:
            raise RecognitionServiceError(
                "gemini_auth_or_request_error",
                "Gemini rejected the API key, model, or request configuration.",
            ) from exc
        raise RecognitionServiceError(
            "gemini_api_error",
            "Gemini could not process the image. No recognition result was generated.",
        ) from exc
    except (httpx.NetworkError, httpx.TimeoutException, ConnectionError, TimeoutError, OSError) as exc:
        raise RecognitionServiceError(
            "network_error", "Could not reach Gemini. Check the network and try again.", 503
        ) from exc
    except Exception as exc:
        raise RecognitionServiceError(
            "gemini_api_error",
            "Gemini could not process the image. No recognition result was generated.",
        ) from exc


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_medication_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).translate(_ARABIC_DIGITS).lower().strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w%+./]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_medication_strength(value: object) -> str:
    text = normalize_medication_text(value).replace(",", ".")
    if not text:
        return ""
    text = text.replace("ميليغرام", "mg").replace("ميليجرام", "mg")
    text = text.replace("ملغ", "mg").replace("مغ", "mg")
    text = text.replace("ميكروغرام", "mcg").replace("مكغ", "mcg")
    text = text.replace("غرام", "g").replace("غم", "g")
    text = text.replace("ملليلتر", "ml").replace("مل", "ml")
    text = re.sub(r"\b(milligrammes?|milligrams?|mgr)\b", "mg", text)
    text = re.sub(r"\b(microgrammes?|micrograms?|mcg|µg|ug)\b", "mcg", text)
    text = re.sub(r"\b(grammes?|grams?|gr)\b", "g", text)
    return re.sub(r"\s+", "", text)


def _strength_tokens(value: object) -> set[tuple[float, str]]:
    text = normalize_medication_strength(value)
    tokens: set[tuple[float, str]] = set()
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(mcg|mg|g|%|ml)", text):
        amount = float(number)
        canonical_unit = unit
        if unit == "g":
            amount *= 1000
            canonical_unit = "mg"
        elif unit == "mcg":
            amount /= 1000
            canonical_unit = "mg"
        tokens.add((round(amount, 6), canonical_unit))
    return tokens


def medication_strength_similarity(extracted: object, catalogue: object) -> tuple[float, bool]:
    left = _strength_tokens(extracted)
    right = _strength_tokens(catalogue)
    if left and right:
        overlap = left & right
        if not overlap:
            return 0.0, True
        return 100.0 * len(overlap) / max(len(left), len(right)), False
    left_text = normalize_medication_strength(extracted)
    right_text = normalize_medication_strength(catalogue)
    if not left_text:
        return 0.0, False
    score = float(fuzz.ratio(left_text, right_text))
    return score, score < 55


def _best_medication_similarity(query: str, values: list[object]) -> float:
    normalized = [normalize_medication_text(value) for value in values]
    normalized = [value for value in normalized if value]
    if not query or not normalized:
        return 0.0
    return float(max(fuzz.WRatio(query, value) for value in normalized))


def _medication_ingredients_conflict(query: str, values: list[object]) -> bool:
    if not query:
        return False
    scores = [
        _best_medication_similarity(query, [value])
        for value in values
        if normalize_medication_text(value)
    ]
    return bool(scores) and max(scores) < 55


def load_medication_catalogue(path: Path = CATALOGUE_FILE) -> pd.DataFrame:
    required = {
        "id",
        "brand_name",
        "brand_name_ar",
        "active_ingredient",
        "active_ingredient_ar",
        "strength",
        "dosage_form",
        "package_quantity",
    }
    try:
        catalogue = pd.read_csv(path, dtype=str).fillna("")
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not load catalogue: {path}") from exc
    missing = required - set(catalogue.columns)
    if missing:
        raise RuntimeError(f"Catalogue is missing columns: {', '.join(sorted(missing))}")
    return catalogue


def normalize_extraction(extracted: MedicationExtraction) -> NormalizedMedication:
    return NormalizedMedication(
        brand_name=normalize_medication_text(extracted.brand_name) or None,
        active_ingredient=normalize_medication_text(extracted.active_ingredient) or None,
        strength=normalize_medication_strength(extracted.strength) or None,
        dosage_form=normalize_medication_text(extracted.dosage_form) or None,
        package_quantity=normalize_medication_text(extracted.package_quantity) or None,
    )


def rank_medication_catalogue(
    extracted: MedicationExtraction,
    catalogue: pd.DataFrame | None = None,
    limit: int = 3,
) -> tuple[str, list[MatchCandidate], NormalizedMedication, str]:
    data = catalogue if catalogue is not None else load_medication_catalogue()
    normalized = normalize_extraction(extracted)
    brand_query = normalized.brand_name or ""
    ingredient_query = normalized.active_ingredient or ""
    strength_query = normalized.strength or ""
    evidence_count = sum(bool(value) for value in (brand_query, ingredient_query, strength_query))

    if evidence_count == 0:
        return (
            "no_match",
            [],
            normalized,
            "No readable brand, active ingredient, or strength was available for matching.",
        )

    ranked: list[MatchCandidate] = []
    for _, row in data.iterrows():
        brand_score = _best_medication_similarity(
            brand_query, [row["brand_name"], row["brand_name_ar"]]
        )
        ingredient_values = [row["active_ingredient"], row["active_ingredient_ar"]]
        ingredient_score = (
            _best_medication_similarity(ingredient_query, ingredient_values)
            if ingredient_query
            else None
        )
        strength_score, strength_conflict = medication_strength_similarity(
            strength_query, row["strength"]
        )
        strength_score_value = strength_score if strength_query else None
        if not strength_query:
            strength_conflict = False

        ingredient_conflict = _medication_ingredients_conflict(
            ingredient_query, ingredient_values
        )
        components: list[tuple[float, float]] = []
        if brand_query:
            components.append((brand_score, 0.50))
        if ingredient_query and ingredient_score is not None:
            components.append((ingredient_score, 0.30))
        if strength_query and strength_score_value is not None:
            components.append((strength_score_value, 0.20))
        weighted = sum(score * weight for score, weight in components) / sum(
            weight for _, weight in components
        )

        conflicts: list[str] = []
        if ingredient_conflict:
            conflicts.append("active_ingredient")
        if strength_conflict:
            conflicts.append("strength")
        if conflicts:
            weighted = min(weighted, 49.0)

        ranked.append(
            MatchCandidate(
                catalogue_id=row["id"],
                brand_name=row["brand_name"],
                brand_name_ar=row["brand_name_ar"] or None,
                active_ingredient=row["active_ingredient"],
                active_ingredient_ar=row["active_ingredient_ar"] or None,
                strength=row["strength"],
                dosage_form=row["dosage_form"],
                package_quantity=row["package_quantity"],
                brand_score=round(brand_score, 1),
                ingredient_score=(
                    round(ingredient_score, 1) if ingredient_score is not None else None
                ),
                strength_score=(
                    round(strength_score_value, 1)
                    if strength_score_value is not None
                    else None
                ),
                ranking_score=round(weighted, 1),
                conflicts=conflicts,
            )
        )

    ranked.sort(key=lambda item: item.ranking_score, reverse=True)
    top = ranked[:limit]
    top_score = top[0].ranking_score if top else 0.0
    margin = top_score - top[1].ranking_score if len(top) > 1 else top_score

    if top_score < 50:
        status = "no_match"
        message = "No sufficiently similar catalogue entry was found."
    elif top_score >= 80 and evidence_count >= 2 and not top[0].conflicts and margin >= 5:
        status = "candidate_found"
        message = "A strong catalogue candidate was found; verify it against the physical box."
    else:
        status = "needs_confirmation"
        if top[0].conflicts:
            message = (
                "Possible candidates exist, but an ingredient or strength conflict prevents "
                "confident identification."
            )
        else:
            message = "Possible candidates exist, but more visible package details are needed."
    return status, top, normalized, message


def recognize_medication(image_bytes: bytes, mime_type: str) -> RecognitionResult:
    validated_mime = validate_image(image_bytes, mime_type)
    extracted, model = extract_image_with_gemini(image_bytes, validated_mime)
    status, candidates, normalized, message = rank_medication_catalogue(extracted)
    return RecognitionResult(
        status=status,
        model=model,
        extracted=extracted,
        normalized=normalized,
        candidates=candidates,
        message=message,
    )
