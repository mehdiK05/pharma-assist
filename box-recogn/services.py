"""Gemini extraction, image validation, and local catalogue matching."""

from __future__ import annotations

import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rapidfuzz import fuzz


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CATALOGUE_PATH = BASE_DIR / "medications.csv"
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_MIME_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
EXTRACTION_FIELDS = (
    "brand_name",
    "active_ingredient",
    "strength",
    "dosage_form",
    "package_quantity",
    "visible_text",
)


class ImageValidationError(ValueError):
    """The uploaded file is not a safe, supported image."""


class GeminiServiceError(RuntimeError):
    """A Gemini call failed without producing recognition output."""

    def __init__(self, code: str, message: str, http_status: int = 502):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class MedicationExtraction(BaseModel):
    """Only text visibly supported by the uploaded package."""

    model_config = ConfigDict(extra="ignore")

    brand_name: str | None = Field(
        default=None, description="Visible commercial/brand name, or null."
    )
    active_ingredient: str | None = Field(
        default=None,
        description="Visible generic name or active ingredient, or null.",
    )
    strength: str | None = Field(
        default=None, description="Visible strength including units, or null."
    )
    dosage_form: str | None = Field(
        default=None, description="Visible dosage form, or null."
    )
    package_quantity: str | None = Field(
        default=None, description="Visible package quantity, or null."
    )
    visible_text: str | None = Field(
        default=None,
        description="Faithful transcription of readable Arabic and French text, or null.",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields that are absent, hidden, or unreadable.",
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
        "Identification aid only. Do not use this result for treatment, dosage, or substitution decisions."
    )


EXTRACTION_PROMPT = """
Inspect this single photo of medication packaging. Extract only information that is
actually visible and readable on the package, including Arabic and French text.

Rules:
- Never infer, complete, translate into a guessed product, or use outside knowledge.
- Preserve the visible spelling. If a field is absent, hidden, blurry, or uncertain,
  return null and include its field name in missing_fields.
- active_ingredient means the generic name / active ingredient.
- strength must include every visible number and unit that belongs to the medicine.
- visible_text is a faithful compact transcription of all readable package text.
- Do not provide medical advice, dosage instructions, or substitutions.
""".strip()


def validate_image(image_bytes: bytes, declared_mime_type: str | None) -> str:
    """Validate size, declared type, decoding, dimensions, and actual format."""
    if not image_bytes:
        raise ImageValidationError("The uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    if declared_mime_type not in ALLOWED_MIME_TYPES:
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

    if actual_format != ALLOWED_MIME_TYPES[declared_mime_type]:
        raise ImageValidationError("The file content does not match its declared image type.")
    if width < 20 or height < 20:
        raise ImageValidationError("Image dimensions are too small to inspect.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError("Image dimensions are too large (maximum 25 megapixels).")
    return declared_mime_type


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text or text.lower() in {"null", "none", "unknown", "unreadable", "n/a"}:
        return None
    return text


def clean_extraction(extraction: MedicationExtraction) -> MedicationExtraction:
    """Normalize empty values and calculate missing fields server-side."""
    values = {
        field: _clean_optional(getattr(extraction, field)) for field in EXTRACTION_FIELDS
    }
    missing = [field for field, value in values.items() if value is None]
    return MedicationExtraction(**values, missing_fields=missing)


def extract_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[MedicationExtraction, str]:
    """Call Gemini once and validate its response. No fallback output is invented."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    if not key:
        raise GeminiServiceError(
            "gemini_not_configured",
            "GEMINI_API_KEY is not configured on the API server.",
            503,
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = client.models.generate_content(
            model=selected_model,
            contents=[EXTRACTION_PROMPT, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=MedicationExtraction.model_json_schema(),
                temperature=0,
                max_output_tokens=2048,
            ),
        )
        if not response.text:
            raise GeminiServiceError(
                "empty_model_response",
                "Gemini returned no text. Try a clearer image.",
                502,
            )
        extraction = MedicationExtraction.model_validate_json(response.text)
        return clean_extraction(extraction), selected_model
    except GeminiServiceError:
        raise
    except ValidationError as exc:
        raise GeminiServiceError(
            "invalid_model_response",
            "Gemini returned data that did not match the required schema.",
            502,
        ) from exc
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        text = str(exc).lower()
        if status == 429 or "429" in text or "resource_exhausted" in text:
            raise GeminiServiceError(
                "gemini_rate_limited",
                "Gemini rate limit reached. Wait briefly and try again.",
                429,
            ) from exc
        if status in {400, 401, 403}:
            raise GeminiServiceError(
                "gemini_auth_or_request_error",
                "Gemini rejected the API key, model, or request configuration.",
                502,
            ) from exc
        raise GeminiServiceError(
            "gemini_api_error",
            "Gemini could not process the image. No recognition result was generated.",
            502,
        ) from exc


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_text(value: object) -> str:
    """Normalize French/Latin and Arabic text for matching, not for display."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).translate(_ARABIC_DIGITS).lower().strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w%+./]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_strength(value: object) -> str:
    """Canonicalize common mass units while retaining multi-ingredient strengths."""
    text = normalize_text(value).replace(",", ".")
    if not text:
        return ""
    # Common Arabic packaging abbreviations and spellings.
    text = text.replace("ميليغرام", "mg").replace("ميليجرام", "mg")
    text = text.replace("ملغ", "mg").replace("مغ", "mg")
    text = text.replace("ميكروغرام", "mcg").replace("مكغ", "mcg")
    text = text.replace("غرام", "g").replace("غم", "g")
    text = text.replace("ملليلتر", "ml").replace("مل", "ml")
    text = re.sub(r"\b(milligrammes?|milligrams?|mgr)\b", "mg", text)
    text = re.sub(r"\b(microgrammes?|micrograms?|mcg|µg|ug)\b", "mcg", text)
    text = re.sub(r"\b(grammes?|grams?|gr)\b", "g", text)
    text = re.sub(r"\s+", "", text)
    return text


def _strength_tokens(value: object) -> set[tuple[float, str]]:
    text = normalize_strength(value)
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


def strength_similarity(extracted: object, catalogue: object) -> tuple[float, bool]:
    """Return a separate strength score and whether an explicit conflict exists."""
    left = _strength_tokens(extracted)
    right = _strength_tokens(catalogue)
    if left and right:
        overlap = left & right
        if not overlap:
            return 0.0, True
        return 100.0 * len(overlap) / max(len(left), len(right)), False
    left_text = normalize_strength(extracted)
    right_text = normalize_strength(catalogue)
    if not left_text:
        return 0.0, False
    score = float(fuzz.ratio(left_text, right_text))
    return score, score < 55


def _best_similarity(query: str, values: list[object]) -> float:
    normalized_values = [normalize_text(value) for value in values]
    normalized_values = [value for value in normalized_values if value]
    if not query or not normalized_values:
        return 0.0
    return float(max(fuzz.WRatio(query, value) for value in normalized_values))


def _ingredients_conflict(query: str, values: list[object]) -> bool:
    """Only mark conflict when no ingredient wording meaningfully overlaps."""
    if not query:
        return False
    scores = [_best_similarity(query, [value]) for value in values if normalize_text(value)]
    return bool(scores) and max(scores) < 55


def load_catalogue(path: Path = CATALOGUE_PATH) -> pd.DataFrame:
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


def normalized_extraction(extracted: MedicationExtraction) -> NormalizedMedication:
    return NormalizedMedication(
        brand_name=normalize_text(extracted.brand_name) or None,
        active_ingredient=normalize_text(extracted.active_ingredient) or None,
        strength=normalize_strength(extracted.strength) or None,
        dosage_form=normalize_text(extracted.dosage_form) or None,
        package_quantity=normalize_text(extracted.package_quantity) or None,
    )


def rank_catalogue(
    extracted: MedicationExtraction,
    catalogue: pd.DataFrame | None = None,
    limit: int = 3,
) -> tuple[str, list[MatchCandidate], NormalizedMedication, str]:
    """Rank catalogue rows, keeping conflicts from becoming confident matches."""
    data = catalogue if catalogue is not None else load_catalogue()
    normalized = normalized_extraction(extracted)
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
        brand_score = _best_similarity(
            brand_query, [row["brand_name"], row["brand_name_ar"]]
        )
        ingredient_values = [row["active_ingredient"], row["active_ingredient_ar"]]
        ingredient_score = (
            _best_similarity(ingredient_query, ingredient_values)
            if ingredient_query
            else None
        )
        strength_score, strength_conflict = strength_similarity(
            strength_query, row["strength"]
        )
        if not strength_query:
            strength_score_value: float | None = None
            strength_conflict = False
        else:
            strength_score_value = strength_score

        ingredient_conflict = _ingredients_conflict(ingredient_query, ingredient_values)
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
    elif (
        top_score >= 80
        and evidence_count >= 2
        and not top[0].conflicts
        and margin >= 5
    ):
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
    extracted, model = extract_with_gemini(image_bytes, validated_mime)
    status, candidates, normalized, message = rank_catalogue(extracted)
    return RecognitionResult(
        status=status,
        model=model,
        extracted=extracted,
        normalized=normalized,
        candidates=candidates,
        message=message,
    )


if __name__ == "__main__":
    # API-free smoke tests for the safety-critical local matching behavior.
    import unittest

    class MatchingTests(unittest.TestCase):
        def test_french_box_prefers_exact_strength(self) -> None:
            extracted = clean_extraction(
                MedicationExtraction(
                    brand_name="DOLIPRANE",
                    active_ingredient="Paracétamol",
                    strength="500 mg",
                )
            )
            status, candidates, _, _ = rank_catalogue(extracted)
            self.assertEqual(status, "candidate_found")
            self.assertEqual(candidates[0].catalogue_id, "MED001")

        def test_arabic_box(self) -> None:
            extracted = clean_extraction(
                MedicationExtraction(
                    brand_name="بانادول",
                    active_ingredient="باراسيتامول",
                    strength="٥٠٠ ملغ",
                )
            )
            status, candidates, _, _ = rank_catalogue(extracted)
            self.assertEqual(status, "candidate_found")
            self.assertEqual(candidates[0].catalogue_id, "MED003")

        def test_missing_text_is_no_match(self) -> None:
            status, candidates, _, _ = rank_catalogue(MedicationExtraction())
            self.assertEqual(status, "no_match")
            self.assertEqual(candidates, [])

        def test_conflicting_strength_blocks_confidence(self) -> None:
            extracted = MedicationExtraction(
                brand_name="Doliprane",
                active_ingredient="paracetamol",
                strength="250 mg",
            )
            status, candidates, _, _ = rank_catalogue(extracted)
            self.assertNotEqual(status, "candidate_found")
            self.assertIn("strength", candidates[0].conflicts)

        def test_unknown_brand_is_no_match(self) -> None:
            extracted = MedicationExtraction(brand_name="Totally Unknown Zyxq")
            status, _, _, _ = rank_catalogue(extracted)
            self.assertEqual(status, "no_match")

    unittest.main(verbosity=2)
