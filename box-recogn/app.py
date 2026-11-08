"""Production-style Streamlit UI for medication-box recognition."""

from __future__ import annotations

import hashlib
import html
import io
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError


load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024

st.set_page_config(
    page_title="Medication Box Recognition",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #102a2a;
        --muted: #617777;
        --teal: #087f73;
        --teal-dark: #075e57;
        --mint: #e8f7f3;
        --line: #dce8e5;
        --surface: #ffffff;
        --canvas: #f5f8f7;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 2%, rgba(46, 196, 157, .09), transparent 24rem),
            radial-gradient(circle at 92% 15%, rgba(50, 125, 190, .07), transparent 28rem),
            var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    h1, h2, h3, p, label { color: var(--ink); }
    h1, h2, h3 { letter-spacing: -0.025em; }

    .hero {
        background: linear-gradient(130deg, #073f3b 0%, #087f73 58%, #11947f 100%);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 24px;
        box-shadow: 0 18px 55px rgba(8, 70, 65, .18);
        color: white;
        overflow: hidden;
        padding: 34px 38px;
        position: relative;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -60px;
        top: -95px;
        border-radius: 50%;
        border: 48px solid rgba(255,255,255,.07);
    }

    .brand-row { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
    .brand-mark {
        width: 42px; height: 42px; display: grid; place-items: center;
        background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.24);
        border-radius: 13px; font-size: 22px;
    }
    .brand-name { font-size: 14px; font-weight: 750; letter-spacing: .02em; color: white; }
    .brand-tag { font-size: 11px; color: #bfe9e1; text-transform: uppercase; letter-spacing: .1em; }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.05; margin: 0 0 13px; max-width: 730px; }
    .hero p { color: #d8f2ed; font-size: 16px; line-height: 1.65; margin: 0; max-width: 720px; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 7px; margin-top: 24px;
        padding: 7px 11px; border-radius: 999px; background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.18); color: #e9fffa; font-size: 12px;
    }

    .steps {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
        margin: 18px 0 28px;
    }
    .step {
        display: flex; align-items: center; gap: 11px; padding: 13px 16px;
        border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,.76);
        color: var(--muted); font-size: 13px;
    }
    .step-number {
        flex: 0 0 auto; width: 26px; height: 26px; display: grid; place-items: center;
        border-radius: 8px; background: var(--mint); color: var(--teal-dark); font-weight: 800;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.92);
        border-color: var(--line);
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(31, 69, 65, .055);
    }

    [data-testid="stFileUploader"] section {
        background: #f8fcfb;
        border: 1.5px dashed #9fc8bf;
        border-radius: 16px;
        min-height: 156px;
        transition: border-color .2s ease, background .2s ease;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--teal); background: #f1faf7; }
    [data-testid="stFileUploader"] small { color: var(--muted); }

    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #087f73, #0a9b86);
        border: 0; border-radius: 12px; min-height: 48px; font-weight: 750;
        box-shadow: 0 8px 20px rgba(8,127,115,.2);
    }
    .stButton > button[kind="primary"]:hover { background: #076e65; }

    .eyebrow { color: var(--teal); font-size: 12px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 5px; }
    .section-title { font-size: 25px; font-weight: 800; color: var(--ink); line-height: 1.2; }
    .section-copy { color: var(--muted); font-size: 14px; line-height: 1.55; margin-top: 5px; }
    .upload-placeholder {
        min-height: 300px; display: grid; place-items: center; text-align: center;
        border-radius: 16px; border: 1px solid var(--line);
        background: linear-gradient(145deg, #f8fbfa, #edf6f3); color: var(--muted);
        padding: 28px;
    }
    .placeholder-icon { font-size: 40px; margin-bottom: 10px; }
    .file-meta { color: var(--muted); font-size: 12px; text-align: center; margin-top: -4px; }

    .status-card {
        border-radius: 18px; padding: 22px 24px; border: 1px solid var(--line);
        background: white; min-height: 137px;
    }
    .status-card.success { background: linear-gradient(145deg, #f1fbf7, #ffffff); border-color: #b7e1d5; }
    .status-card.warning { background: linear-gradient(145deg, #fff9eb, #ffffff); border-color: #eedda7; }
    .status-card.neutral { background: linear-gradient(145deg, #f5f7f7, #ffffff); }
    .status-pill {
        display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 10px;
        font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 850;
        margin-bottom: 13px;
    }
    .success .status-pill { color: #076c59; background: #d8f4eb; }
    .warning .status-pill { color: #8a5d00; background: #fff0bf; }
    .neutral .status-pill { color: #526969; background: #e9eeee; }
    .status-title { font-size: 19px; font-weight: 800; margin-bottom: 5px; }
    .status-copy { color: var(--muted); font-size: 13px; line-height: 1.5; }

    .metric-card {
        min-height: 137px; padding: 20px; border: 1px solid var(--line);
        border-radius: 18px; background: rgba(255,255,255,.94);
    }
    .metric-label { color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 9px; }
    .metric-value { color: var(--ink); font-size: 28px; font-weight: 850; letter-spacing: -.04em; }
    .metric-note { color: var(--muted); font-size: 11px; margin-top: 5px; }

    .field-card {
        min-height: 103px; border: 1px solid var(--line); border-radius: 14px;
        padding: 15px 16px; background: #fff; margin-bottom: 8px;
    }
    .field-label { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: .06em; }
    .field-value { color: var(--ink); font-size: 16px; font-weight: 720; margin-top: 7px; line-height: 1.35; }
    .field-empty { color: #94a4a4; font-size: 15px; font-weight: 500; }

    .candidate-head { display: flex; align-items: center; gap: 12px; }
    .rank {
        width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
        color: var(--teal-dark); background: var(--mint); font-weight: 850;
    }
    .candidate-name { font-size: 19px; font-weight: 830; color: var(--ink); }
    .candidate-sub { font-size: 13px; color: var(--muted); margin-top: 2px; }
    .score-big { text-align: right; font-size: 25px; font-weight: 850; color: var(--teal-dark); }
    .score-caption { text-align: right; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; }

    .safety {
        display: flex; gap: 13px; align-items: flex-start; padding: 16px 18px;
        border-radius: 14px; border: 1px solid #d7e5e2; background: #f8fbfa;
        color: #47605f; font-size: 13px; line-height: 1.55; margin-top: 22px;
    }
    .footer-note { text-align: center; color: #7b8d8c; font-size: 11px; margin-top: 28px; }

    div[data-testid="stAlert"] { border-radius: 14px; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] { border-radius: 10px; padding-left: 16px; padding-right: 16px; }

    @media (max-width: 760px) {
        .block-container { padding: 1rem .85rem 3rem; }
        .hero { padding: 25px 22px; border-radius: 18px; }
        .hero h1 { font-size: 2.05rem; }
        .steps { grid-template-columns: 1fr; }
        .status-card, .metric-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def escape(value: Any) -> str:
    return html.escape(str(value))


def format_bytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB"


@st.cache_data(ttl=10, show_spinner=False)
def api_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def validate_upload(uploaded: Any) -> tuple[bytes | None, Image.Image | None, str | None, str]:
    image_bytes = uploaded.getvalue()
    if not image_bytes:
        return None, None, "The selected file is empty.", ""
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None, None, f"Maximum upload size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB.", ""
    if uploaded.type not in {"image/jpeg", "image/png"}:
        return None, None, "Only JPEG and PNG images are accepted.", ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            if source.format not in {"JPEG", "PNG"}:
                return None, None, "The file content is not a JPEG or PNG image.", ""
            width, height = source.size
            if width < 20 or height < 20:
                return None, None, "The image dimensions are too small to inspect.", ""
            if width * height > 25_000_000:
                return None, None, "Image dimensions exceed the 25 megapixel limit.", ""
            preview = ImageOps.exif_transpose(source).copy()
            metadata = f"{source.format} · {width} × {height} · {format_bytes(len(image_bytes))}"
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None, None, "The selected file could not be decoded as an image.", ""
    return image_bytes, preview, None, metadata


def request_recognition(uploaded: Any, image_bytes: bytes) -> dict[str, Any] | None:
    try:
        response = requests.post(
            f"{API_URL}/recognize",
            files={"image": (uploaded.name, image_bytes, uploaded.type)},
            timeout=120,
        )
    except requests.ConnectionError:
        st.error(f"Could not reach the recognition API at {API_URL}. Start FastAPI and retry.")
        return None
    except requests.Timeout:
        st.error("Recognition timed out. Try a smaller or clearer image.")
        return None
    except requests.RequestException as exc:
        st.error(f"The API request failed: {exc}")
        return None

    try:
        payload = response.json()
    except ValueError:
        st.error(f"The API returned an invalid response (HTTP {response.status_code}).")
        return None
    if not response.ok:
        error = payload.get("error", {})
        st.error(error.get("message", f"Recognition failed (HTTP {response.status_code})."))
        if error.get("code"):
            st.caption(f"Reference: {error['code']}")
        return None
    return payload


def render_field(label: str, value: Any) -> None:
    if value:
        content = f'<div class="field-value" dir="auto">{escape(value)}</div>'
    else:
        content = '<div class="field-value field-empty">Not visible</div>'
    st.markdown(
        f'<div class="field-card"><div class="field-label">{escape(label)}</div>{content}</div>',
        unsafe_allow_html=True,
    )


def render_candidate(item: dict[str, Any], rank: int) -> None:
    with st.container(border=True):
        title_col, score_col = st.columns([4, 1])
        with title_col:
            arabic = f" · {escape(item['brand_name_ar'])}" if item.get("brand_name_ar") else ""
            st.markdown(
                f"""
                <div class="candidate-head">
                    <div class="rank">{rank}</div>
                    <div>
                        <div class="candidate-name" dir="auto">{escape(item['brand_name'])}{arabic}</div>
                        <div class="candidate-sub" dir="auto">{escape(item['active_ingredient'])} · {escape(item['strength'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with score_col:
            st.markdown(
                f'<div class="score-big">{item["ranking_score"]:.0f}</div>'
                '<div class="score-caption">ranking score</div>',
                unsafe_allow_html=True,
            )

        st.progress(min(float(item["ranking_score"]) / 100, 1.0))
        info_a, info_b, info_c = st.columns(3)
        info_a.metric("Brand", f'{item["brand_score"]:.0f} / 100')
        ingredient = item.get("ingredient_score")
        info_b.metric("Ingredient", f"{ingredient:.0f} / 100" if ingredient is not None else "Not read")
        strength = item.get("strength_score")
        info_c.metric("Strength", f"{strength:.0f} / 100" if strength is not None else "Not read")
        st.caption(f'{item["dosage_form"]} · {item["package_quantity"]} · Catalogue ID {item["catalogue_id"]}')
        if item.get("conflicts"):
            st.error("Conflict detected: " + ", ".join(item["conflicts"]).replace("_", " "))


def render_results(payload: dict[str, Any]) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Recognition report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Result overview</div>', unsafe_allow_html=True)

    status = payload["status"]
    status_meta = {
        "candidate_found": ("success", "Strong candidate", "✓"),
        "needs_confirmation": ("warning", "Review needed", "!"),
        "no_match": ("neutral", "No catalogue match", "—"),
    }
    style, label, symbol = status_meta.get(status, ("neutral", "Unknown status", "—"))
    candidates = payload.get("candidates", [])
    top_score = candidates[0]["ranking_score"] if candidates else 0
    extraction = payload["extracted"]
    core_fields = ["brand_name", "active_ingredient", "strength", "dosage_form", "package_quantity"]
    visible_fields = sum(bool(extraction.get(field)) for field in core_fields)

    overview, score, fields = st.columns([2.1, 1, 1])
    with overview:
        st.markdown(
            f"""
            <div class="status-card {style}">
                <div class="status-pill">{symbol}&nbsp;&nbsp;{label}</div>
                <div class="status-title">{escape(payload['message'])}</div>
                <div class="status-copy">Always verify the result against the physical package.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with score:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Top ranking score</div>
                <div class="metric-value">{top_score:.0f}<span style="font-size:15px;color:#78908e"> / 100</span></div>
                <div class="metric-note">Similarity, not probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fields:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Fields visible</div>
                <div class="metric-value">{visible_fields}<span style="font-size:15px;color:#78908e"> / 5</span></div>
                <div class="metric-note">Extracted from this image</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    match_tab, extraction_tab, details_tab = st.tabs(
        ["Likely matches", "Package text", "Technical details"]
    )
    with match_tab:
        st.markdown("#### Ranked catalogue candidates")
        st.caption("Candidates are ordered using brand, ingredient, and strength similarity.")
        if candidates:
            for index, candidate in enumerate(candidates, start=1):
                render_candidate(candidate, index)
        else:
            st.info("There is not enough readable information to propose catalogue candidates.")

    with extraction_tab:
        st.markdown("#### Information visible on the box")
        first_row = st.columns(3)
        with first_row[0]:
            render_field("Brand name", extraction.get("brand_name"))
        with first_row[1]:
            render_field("Active ingredient", extraction.get("active_ingredient"))
        with first_row[2]:
            render_field("Strength", extraction.get("strength"))
        second_row = st.columns(2)
        with second_row[0]:
            render_field("Dosage form", extraction.get("dosage_form"))
        with second_row[1]:
            render_field("Package quantity", extraction.get("package_quantity"))

        st.markdown("##### Visible text transcription")
        if extraction.get("visible_text"):
            st.code(extraction["visible_text"], language=None, wrap_lines=True)
        else:
            st.info("No additional text was reliably readable.")
        if extraction.get("missing_fields"):
            missing = ", ".join(field.replace("_", " ") for field in extraction["missing_fields"])
            st.caption(f"Missing or unreadable: {missing}")

    with details_tab:
        st.markdown("#### Processing details")
        detail_a, detail_b, detail_c = st.columns(3)
        detail_a.metric("Status", status.replace("_", " ").title())
        detail_b.metric("Model", payload["model"])
        detail_c.metric("Candidates shown", len(candidates))
        with st.expander("Normalized values used for matching"):
            normalized = payload.get("normalized", {})
            for key, value in normalized.items():
                st.text(f"{key.replace('_', ' ').title()}: {value or '—'}")
        st.info(payload["ranking_note"])

    st.markdown(
        '<div class="safety"><div style="font-size:20px">⚕</div><div><strong>Safety notice</strong><br>'
        f'{escape(payload["medical_disclaimer"])}</div></div>',
        unsafe_allow_html=True,
    )


# Sidebar is deliberately secondary: the primary workflow remains one clear action.
with st.sidebar:
    st.markdown("### System")
    health = api_health()
    if health:
        st.success("Recognition API online")
        st.caption(f"Model: {health.get('gemini_model', '—')}")
        st.caption(f"Catalogue: {health.get('catalogue_items', 0)} products")
        if not health.get("gemini_configured"):
            st.warning("Gemini API key is not configured")
    else:
        st.error("Recognition API offline")
        st.caption(f"Expected at {API_URL}")
    st.divider()
    st.markdown("### Privacy")
    st.caption("Uploaded images are sent to your FastAPI server and the configured Google Gemini API.")
    st.markdown("### Supported files")
    st.caption(f"JPEG or PNG · maximum {MAX_IMAGE_BYTES // (1024 * 1024)} MB")


st.markdown(
    """
    <section class="hero">
        <div class="brand-row">
            <div class="brand-mark">✚</div>
            <div><div class="brand-name">MedBox Vision</div><div class="brand-tag">Medication box recognition</div></div>
        </div>
        <h1>Identify the box.<br>Verify with confidence.</h1>
        <p>Read visible Arabic and French packaging with Gemini Vision, then compare it with a controlled local catalogue.</p>
        <div class="hero-badge">●&nbsp; Vision extraction &nbsp;·&nbsp; Local catalogue ranking</div>
    </section>
    <div class="steps">
        <div class="step"><span class="step-number">1</span><span>Upload one clear package photo</span></div>
        <div class="step"><span class="step-number">2</span><span>Extract only visible information</span></div>
        <div class="step"><span class="step-number">3</span><span>Review the top catalogue matches</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, preview_column = st.columns([1.02, 1], gap="large")
with upload_column:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Start recognition</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload package photo</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">For the best result, keep the brand, ingredient, and strength in focus.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose or drop an image",
            type=["jpg", "jpeg", "png"],
            help=f"JPEG or PNG, up to {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
            label_visibility="collapsed",
        )

        image_bytes: bytes | None = None
        preview_image: Image.Image | None = None
        image_error: str | None = None
        metadata = ""
        if uploaded is not None:
            image_bytes, preview_image, image_error, metadata = validate_upload(uploaded)
            if image_error:
                st.error(image_error)
            else:
                st.success("Image ready for recognition")
                st.caption(metadata)

        recognize_clicked = st.button(
            "Analyze medication box  →",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None or image_error is not None,
        )
        st.caption("Usually completes in a few seconds. Results require human verification.")

with preview_column:
    if preview_image is not None:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Image preview</div>', unsafe_allow_html=True)
            st.image(preview_image, width="stretch")
            st.markdown(f'<div class="file-meta">{escape(uploaded.name)} · {escape(metadata)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="upload-placeholder">
                <div><div class="placeholder-icon">▣</div>
                <strong>Your package preview appears here</strong><br>
                <span style="font-size:13px">Use an evenly lit, front-facing photo without glare.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if "recognition_result" not in st.session_state:
    st.session_state.recognition_result = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

current_signature = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
if current_signature != st.session_state.upload_signature:
    st.session_state.upload_signature = current_signature
    st.session_state.recognition_result = None

if recognize_clicked and uploaded is not None and image_bytes is not None:
    with st.spinner("Reading the package and comparing catalogue entries…"):
        st.session_state.recognition_result = request_recognition(uploaded, image_bytes)

if st.session_state.recognition_result:
    render_results(st.session_state.recognition_result)

st.markdown(
    '<div class="footer-note">Medication Box Recognition MVP · Arabic + French · '
    'No treatment, dosage, or substitution recommendations</div>',
    unsafe_allow_html=True,
)
