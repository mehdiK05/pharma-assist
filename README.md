# PharmaAssist

PharmaAssist combines two pharmacy workflows in one conversational interface:

- multilingual text assistance for Darija (Arabic or Latin script), Arabic, French, and mixed requests;
- medication-box image extraction and local catalogue matching for JPEG/PNG uploads.

The Streamlit chat composer accepts either a normal text question or one image attachment. Text is sent to `POST /chat`; an attached image is sent to `POST /recognize`.

## What it does

### Text questions

PharmaAssist identifies these intents: availability, price, location, hours, contact, medication request, and other. It extracts the medication name, strength, quantity, unit, language, missing information, and a clarification question when needed. Gemini performs one structured interpretation call; pharmacy contact, prices, and stock are looked up locally in `pharmacy.csv`.

### Medication images

PharmaAssist validates a JPEG or PNG, asks Gemini to extract only visible package information, then compares the extraction with `medications.csv`. It normalizes Arabic/French text, digits, and strength units, ranks up to three candidates with RapidFuzz, and flags ingredient or strength conflicts. Ranking scores represent text similarity, not medical probability.

The application does not provide diagnosis, dosage advice, prescriptions, or substitutions.

## Architecture

```text
Streamlit chat (app.py)
       │
       ├── text ──► POST /chat ──► Gemini structured interpretation ──► pharmacy.csv
       │
       └── image ─► POST /recognize ─► image validation ─► Gemini vision extraction
                                                └──────► medications.csv + RapidFuzz
```

The root application is the integrated product. The `box-recogn/` directory remains as the original standalone recognition project for reference; it is not required to run PharmaAssist.

## Project files

- `app.py` — production-style Streamlit chat interface for text and images.
- `api.py` — unified FastAPI application with `/chat`, `/recognize`, and `/health`.
- `services.py` — text interpretation, conversational replies, image extraction, validation, normalization, and matching.
- `pharmacy.csv` — pharmacy contact, hours, stock, and prices.
- `medications.csv` — bilingual medication catalogue containing 19 package variants.
- `.env.example` — configuration template without credentials.
- `requirements.txt` — pinned dependencies.
- `PROJECT_DOCUMENTATION.md` — technical and presentation documentation.

## Install

Python 3.11+ is recommended.

```bash
cd "/home/mehdi/pharma assist"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
API_BASE_URL=http://127.0.0.1:8000
MAX_IMAGE_MB=10
```

Never commit `.env` or an API key.

## Run

Terminal 1 — API:

```bash
source .venv/bin/activate
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — chat UI:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open:

- PharmaAssist: <http://localhost:8501>
- API documentation: <http://127.0.0.1:8000/docs>
- Health status: <http://127.0.0.1:8000/health>

## API examples

Text question:

```bash
curl -sS -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Salam wach 3andkom Doliprane 1000?"}' \
  | python -m json.tool
```

Medication image:

```bash
curl -sS -X POST http://127.0.0.1:8000/recognize \
  -F "image=@/absolute/path/to/medication-box.jpg;type=image/jpeg" \
  | python -m json.tool
```

The image route accepts JPEG/PNG files up to `MAX_IMAGE_MB` and 25 megapixels. Uploaded images are sent by the API server to the configured Gemini provider.

## Local validation

The application can be compiled and its API opened without spending Gemini quota:

```bash
python -m py_compile app.py api.py services.py
curl http://127.0.0.1:8000/health
```

A valid Gemini key is required for real `/chat` and `/recognize` inference. Provider failures, invalid structured output, bad images, unavailable catalogues, missing configuration, and quota errors return explicit HTTP errors; the application does not invent a fallback AI result.
