"""Unified PharmaAssist chat interface for text and medication-box images."""

from __future__ import annotations

import html
import io
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))

st.set_page_config(
    page_title="PharmaAssist",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #13243a;
        --muted: #66788a;
        --teal: #0b8f7b;
        --teal-dark: #087261;
        --mint: #eaf8f4;
        --blue: #3978f6;
        --line: #e3ebe9;
        --canvas: #f6f9f8;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% -5%, rgba(48, 184, 153, .09), transparent 28rem),
            radial-gradient(circle at 95% 10%, rgba(57, 120, 246, .06), transparent 25rem),
            var(--canvas);
        color: var(--navy);
    }
    .block-container { max-width: 980px; padding: 1.4rem 1.2rem 8rem; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: .35rem 0 1.2rem;
    }
    .brand { display: flex; align-items: center; gap: .78rem; }
    .brand-mark {
        width: 42px; height: 42px; display: grid; place-items: center;
        border-radius: 13px; color: white; font-size: 1.35rem; font-weight: 800;
        background: linear-gradient(135deg, #0c7567, #14aa8e);
        box-shadow: 0 8px 20px rgba(11, 143, 123, .22);
    }
    .brand-name { font-size: 1.18rem; font-weight: 800; letter-spacing: -.025em; }
    .brand-sub { color: var(--muted); font-size: .72rem; margin-top: -.08rem; }
    .top-status {
        display: inline-flex; align-items: center; gap: .45rem; padding: .42rem .72rem;
        border: 1px solid #cfe7df; border-radius: 999px; background: rgba(255,255,255,.8);
        color: #287363; font-size: .74rem; font-weight: 700;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #20b486; }

    .hero {
        position: relative; overflow: hidden; padding: 2.15rem 2.3rem;
        border-radius: 24px; color: white; margin-bottom: 1.3rem;
        background: linear-gradient(125deg, #102e43 0%, #0a665f 52%, #0b9a80 100%);
        box-shadow: 0 18px 50px rgba(19, 58, 67, .16);
    }
    .hero:after {
        content: ""; position: absolute; width: 260px; height: 260px;
        border: 42px solid rgba(255,255,255,.06); border-radius: 50%;
        right: -70px; top: -105px;
    }
    .hero-kicker { color: #9de5d6; font-size: .7rem; font-weight: 800; letter-spacing: .14em; }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.08; margin: .5rem 0 .7rem; letter-spacing: -.04em; }
    .hero p { color: #d6eeea; max-width: 650px; line-height: 1.6; margin: 0; font-size: .96rem; }
    .capabilities { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.2rem; }
    .capability {
        padding: .38rem .68rem; border-radius: 999px; font-size: .72rem;
        border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.09);
    }

    .section-kicker { color: var(--teal-dark); font-size: .7rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; margin: 1.15rem 0 .55rem; }
    div[data-testid="stChatMessage"] {
        border: 1px solid var(--line); border-radius: 18px; padding: .95rem 1rem;
        background: rgba(255,255,255,.88); box-shadow: 0 5px 18px rgba(25, 58, 52, .035);
        margin-bottom: .75rem;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #edf7f4;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #cfdfda; border-radius: 18px; background: white;
        box-shadow: 0 12px 35px rgba(22, 57, 49, .12);
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--teal); }
    [data-testid="stBottomBlockContainer"] { background: linear-gradient(transparent, var(--canvas) 24%); }

    .welcome-title { font-weight: 800; font-size: 1.02rem; margin-bottom: .35rem; }
    .welcome-copy { color: var(--muted); font-size: .88rem; line-height: 1.55; }
    .result-banner { border-radius: 14px; padding: .9rem 1rem; margin-bottom: .8rem; border: 1px solid; }
    .result-banner.success { background: #edf9f5; border-color: #bce5d8; }
    .result-banner.warning { background: #fff8e7; border-color: #efdda5; }
    .result-banner.neutral { background: #f3f6f6; border-color: #dce5e3; }
    .result-label { font-size: .66rem; text-transform: uppercase; font-weight: 850; letter-spacing: .1em; color: var(--teal-dark); }
    .result-title { font-weight: 780; margin-top: .25rem; line-height: 1.4; }

    .extract-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .55rem; margin: .65rem 0; }
    .extract-card { border: 1px solid var(--line); border-radius: 12px; background: white; padding: .7rem .75rem; min-height: 72px; }
    .extract-label { color: var(--muted); font-size: .62rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .extract-value { color: var(--navy); font-size: .84rem; font-weight: 700; margin-top: .33rem; overflow-wrap: anywhere; }
    .empty { color: #9aa8a5; font-weight: 500; }
    .candidate-name { font-size: 1rem; font-weight: 800; }
    .candidate-meta { color: var(--muted); font-size: .76rem; margin-top: .15rem; }
    .score { color: var(--teal-dark); font-size: 1.15rem; font-weight: 850; text-align: right; }
    .fine-print { color: #7b8a87; font-size: .69rem; text-align: center; margin-top: 1.4rem; }

    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; border-color: var(--line); background: white; }
    div[data-testid="stAlert"] { border-radius: 13px; }
    .stButton > button { border-radius: 12px; border-color: #dbe6e3; min-height: 2.65rem; font-weight: 650; }
    .stButton > button:hover { border-color: var(--teal); color: var(--teal-dark); }
    [data-baseweb="tab-list"] { gap: .35rem; }
    button[data-baseweb="tab"] { border-radius: 10px; }

    @media (max-width: 720px) {
        .block-container { padding: .8rem .75rem 7.5rem; }
        .hero { padding: 1.6rem 1.25rem; border-radius: 18px; }
        .hero h1 { font-size: 2rem; }
        .top-status { display: none; }
        .extract-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def escape(value: Any) -> str:
    return html.escape(str(value))


@st.cache_data(ttl=10, show_spinner=False)
def api_health() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def call_text_api(message: str) -> tuple[str, dict[str, Any]]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat", json={"message": message}, timeout=45
        )
        payload = response.json()
    except requests.Timeout:
        return "error", {"message": "The request timed out. Please try again."}
    except requests.ConnectionError:
        return "error", {"message": "PharmaAssist API is offline. Start FastAPI and retry."}
    except (requests.RequestException, ValueError):
        return "error", {"message": "The service returned an invalid response."}
    if not response.ok:
        return "error", payload
    return "text", payload


def call_image_api(
    file_name: str, image_bytes: bytes, mime_type: str
) -> tuple[str, dict[str, Any]]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/recognize",
            files={"image": (file_name, image_bytes, mime_type)},
            timeout=120,
        )
        payload = response.json()
    except requests.Timeout:
        return "error", {"message": "Image analysis timed out. Try a clearer image."}
    except requests.ConnectionError:
        return "error", {"message": "PharmaAssist API is offline. Start FastAPI and retry."}
    except (requests.RequestException, ValueError):
        return "error", {"message": "The service returned an invalid image response."}
    if not response.ok:
        return "error", payload
    return "image", payload


def safe_preview(image_bytes: bytes) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            return ImageOps.exif_transpose(source).copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None


def render_text_result(payload: dict[str, Any]) -> None:
    st.markdown(payload.get("answer", "Request understood."))
    interpretation = payload.get("interpretation", {})
    with st.expander("Understanding details"):
        first, second = st.columns(2)
        first.metric("Intent", str(interpretation.get("intent", "—")).replace("_", " ").title())
        second.metric("Language", str(interpretation.get("language", "—")).replace("_", " ").title())
        entity_cols = st.columns(4)
        for column, key in zip(
            entity_cols, ["medication_name", "strength", "quantity", "unit"]
        ):
            value = interpretation.get(key)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            column.metric(key.replace("_", " ").title(), value if value is not None else "—")
        if interpretation.get("missing_information"):
            st.caption(
                "Missing: "
                + ", ".join(
                    str(item).replace("_", " ")
                    for item in interpretation["missing_information"]
                )
            )
        with st.expander("Structured JSON"):
            st.json(payload)


def extraction_card(label: str, value: Any) -> str:
    content = escape(value) if value else '<span class="empty">Not visible</span>'
    return (
        '<div class="extract-card">'
        f'<div class="extract-label">{escape(label)}</div>'
        f'<div class="extract-value" dir="auto">{content}</div></div>'
    )


def render_image_result(payload: dict[str, Any]) -> None:
    status = payload.get("status", "no_match")
    style, label = {
        "candidate_found": ("success", "Strong candidate"),
        "needs_confirmation": ("warning", "Review needed"),
        "no_match": ("neutral", "No catalogue match"),
    }.get(status, ("neutral", "Analysis complete"))
    st.markdown(
        f'<div class="result-banner {style}"><div class="result-label">{label}</div>'
        f'<div class="result-title">{escape(payload.get("message", "Analysis complete."))}</div></div>',
        unsafe_allow_html=True,
    )

    match_tab, extracted_tab = st.tabs(["Likely matches", "Extracted package text"])
    with match_tab:
        candidates = payload.get("candidates", [])
        if not candidates:
            st.info("There is not enough matching information to suggest a catalogue item.")
        for rank, item in enumerate(candidates, start=1):
            with st.container(border=True):
                name_col, score_col = st.columns([4, 1])
                arabic = f" · {escape(item['brand_name_ar'])}" if item.get("brand_name_ar") else ""
                name_col.markdown(
                    f'<div class="candidate-name" dir="auto">#{rank} {escape(item["brand_name"])}{arabic}</div>'
                    f'<div class="candidate-meta">{escape(item["active_ingredient"])} · '
                    f'{escape(item["strength"])} · {escape(item["dosage_form"])}</div>',
                    unsafe_allow_html=True,
                )
                score_col.markdown(
                    f'<div class="score">{item["ranking_score"]:.0f}/100</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(float(item["ranking_score"]) / 100, 1.0))
                if item.get("conflicts"):
                    st.warning(
                        "Conflict: "
                        + ", ".join(item["conflicts"]).replace("_", " ")
                    )

    with extracted_tab:
        extracted = payload.get("extracted", {})
        cards = [
            extraction_card("Brand", extracted.get("brand_name")),
            extraction_card("Active ingredient", extracted.get("active_ingredient")),
            extraction_card("Strength", extracted.get("strength")),
            extraction_card("Dosage form", extracted.get("dosage_form")),
            extraction_card("Package quantity", extracted.get("package_quantity")),
        ]
        st.markdown('<div class="extract-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        if extracted.get("visible_text"):
            st.caption("Visible text")
            st.code(extracted["visible_text"], language=None, wrap_lines=True)
        if extracted.get("missing_fields"):
            st.caption("Not visible: " + ", ".join(extracted["missing_fields"]).replace("_", " "))

    with st.expander("Technical details"):
        st.caption(payload.get("ranking_note", ""))
        st.json(payload)
    st.caption(payload.get("medical_disclaimer", ""))


def render_message(message: dict[str, Any]) -> None:
    avatar = "👤" if message["role"] == "user" else "💊"
    with st.chat_message(message["role"], avatar=avatar):
        kind = message.get("kind")
        if message["role"] == "user":
            if message.get("text"):
                st.markdown(message["text"])
            if message.get("image_bytes"):
                preview = safe_preview(message["image_bytes"])
                if preview is not None:
                    st.image(preview, width=320)
                    st.caption(message.get("file_name", "Medication image"))
                else:
                    st.caption(f"Attached file: {message.get('file_name', 'image')}")
        elif kind == "welcome":
            st.markdown(
                '<div class="welcome-title">Welcome to PharmaAssist</div>'
                '<div class="welcome-copy">Ask about availability, prices, location, hours, or contact '
                'in Darija, Arabic, or French. You can also attach a clear medication-box photo to '
                'extract its visible details and compare it with the local catalogue.</div>',
                unsafe_allow_html=True,
            )
        elif kind == "text":
            render_text_result(message["payload"])
        elif kind == "image":
            render_image_result(message["payload"])
        else:
            st.error(message.get("payload", {}).get("message", "Something went wrong."))


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "kind": "welcome", "text": ""}
    ]
if "pending_text" not in st.session_state:
    st.session_state.pending_text = None

health = api_health()
status_copy = "API online" if health else "API offline"
st.markdown(
    f"""
    <div class="topbar">
        <div class="brand"><div class="brand-mark">✚</div><div>
            <div class="brand-name">PharmaAssist</div>
            <div class="brand-sub">Multilingual pharmacy intelligence</div>
        </div></div>
        <div class="top-status"><span class="status-dot"></span>{status_copy}</div>
    </div>
    <section class="hero">
        <div class="hero-kicker">TEXT + VISION, ONE ASSISTANT</div>
        <h1>Pharmacy help that understands how you speak.</h1>
        <p>Ask naturally in Darija, العربية, or Français — or attach a medication package photo for structured visual recognition.</p>
        <div class="capabilities">
            <span class="capability">✦ Multilingual chat</span>
            <span class="capability">▣ Package recognition</span>
            <span class="capability">⌁ Catalogue matching</span>
            <span class="capability">✓ Structured results</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ✚ PharmaAssist")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "kind": "welcome", "text": ""}
        ]
        st.rerun()
    st.divider()
    st.markdown("#### System status")
    if health:
        st.success("API connected")
        st.caption(f"Model: {health.get('gemini_model', '—')}")
        st.caption(
            f"Medication catalogue: {health.get('medication_catalogue_items', 0)} items"
        )
        if not health.get("gemini_configured"):
            st.warning("Gemini key is not configured")
    else:
        st.error("API unavailable")
        st.caption(API_BASE_URL)
    st.divider()
    st.markdown("#### Supported input")
    st.caption(f"Text up to 1,000 characters\n\nJPEG or PNG up to {MAX_IMAGE_MB} MB")
    st.markdown("#### Languages")
    st.caption("Darija Arabic · Darija Latin · Arabic · French · Mixed")
    st.divider()
    st.caption("PharmaAssist does not provide diagnosis, dosage, prescriptions, or substitutions.")

for chat_message in st.session_state.messages:
    render_message(chat_message)

has_user_message = any(message["role"] == "user" for message in st.session_state.messages)
if not has_user_message:
    st.markdown('<div class="section-kicker">Try a quick question</div>', unsafe_allow_html=True)
    example_cols = st.columns(3)
    examples = [
        ("Check Doliprane", "Salam wach 3andkom Doliprane 1000?"),
        ("Ask the price", "Ch7al taman dyal Augmentin?"),
        ("Opening hours", "Vous fermez à quelle heure ?"),
    ]
    for column, (label, value) in zip(example_cols, examples):
        if column.button(label, use_container_width=True):
            st.session_state.pending_text = value

submission = st.chat_input(
    "Ask a pharmacy question or attach a medication photo…",
    max_chars=1000,
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
    max_upload_size=MAX_IMAGE_MB,
)

submitted_text = ""
submitted_file = None
if submission:
    if isinstance(submission, str):
        submitted_text = submission.strip()
    else:
        submitted_text = submission.text.strip()
        submitted_file = submission.files[0] if submission.files else None
elif st.session_state.pending_text:
    submitted_text = st.session_state.pending_text
    st.session_state.pending_text = None

if submitted_text or submitted_file is not None:
    image_bytes = submitted_file.getvalue() if submitted_file is not None else None
    user_message = {
        "role": "user",
        "kind": "user",
        "text": submitted_text,
        "image_bytes": image_bytes,
        "file_name": submitted_file.name if submitted_file is not None else None,
    }
    st.session_state.messages.append(user_message)

    if submitted_file is not None and image_bytes is not None:
        kind, payload = call_image_api(
            submitted_file.name,
            image_bytes,
            submitted_file.type or "application/octet-stream",
        )
    else:
        kind, payload = call_text_api(submitted_text)
    st.session_state.messages.append(
        {"role": "assistant", "kind": kind, "payload": payload}
    )
    st.rerun()

st.markdown(
    '<div class="fine-print">PharmaAssist · Operational pharmacy information and package identification support</div>',
    unsafe_allow_html=True,
)
