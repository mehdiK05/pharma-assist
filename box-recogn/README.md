# Medication Box Recognition

A deliberately small MVP that reads visible Arabic/French medication-box text with
Google Gemini, then ranks likely products from a local CSV catalogue. It is an
identification aid only: scores are similarities, not probabilities, and it must
not be used for treatment, dosage, or substitution decisions.

## Architecture

Streamlit (`app.py`) previews an uploaded JPEG/PNG and sends it to FastAPI over
HTTP. FastAPI (`api.py`) validates the upload, then `services.py` requests strict
structured JSON from Gemini and independently validates the response. RapidFuzz
ranks `medications.csv` by brand, active ingredient, and strength; explicit
ingredient or strength conflicts prevent a confident result.

## Install and run

Python 3.10+ is recommended.

```bash
cd /home/mehdi/box-recogn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your Gemini key in `.env`:

```dotenv
GEMINI_API_KEY=replace_with_your_real_key
GEMINI_MODEL=gemini-3.1-flash-lite
```

Create a free key at the official [Google AI Studio API Keys
page](https://aistudio.google.com/app/apikey): sign in, accept the terms, choose or
create a project, select **Create API key**, and copy it into `.env`. Billing is not
required for a model that is available on the free tier. Google currently lists
free input/output tokens for `gemini-3.1-flash-lite`; quotas are enforced per
project as requests/minute, tokens/minute, and requests/day, vary by model/project,
and the authoritative active values are shown in [AI
Studio](https://aistudio.google.com/usage). Review the official [pricing
page](https://ai.google.dev/gemini-api/docs/pricing) and [rate-limit
guide](https://ai.google.dev/gemini-api/docs/rate-limits) before a demo because
availability and quotas can change.

Run these in two terminals (activate `.venv` in each):

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

```bash
streamlit run app.py --server.port 8501
```

Open <http://127.0.0.1:8501>. API documentation is at
<http://127.0.0.1:8000/docs>.

## API checks

Check local configuration and catalogue loading:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Exercise the real Gemini connection with a JPEG or PNG (this consumes API quota):

```bash
curl -s -X POST http://127.0.0.1:8000/recognize \
  -F "image=@/absolute/path/to/medication-box.jpg;type=image/jpeg" \
  | python -m json.tool
```

The API never substitutes made-up extraction data when Gemini fails. It returns a
structured `error` object for missing configuration, rate limits, upstream errors,
or invalid model output.

## Demo cases

Use photos you are permitted to send to Google:

1. A clear French package with brand, ingredient, and strength visible.
2. An Arabic package with the same fields visible.
3. A blurry or partly covered label; confirm missing values remain null and the
   status is conservative.
4. Similar product names at different strengths (for example, Doliprane 500 mg
   and 1000 mg).
5. A product absent from the CSV; expect `no_match` when similarity is low.

Do not present recognition accuracy claims until these cases have been tested on a
representative, labelled image set.

## Local tests (no Gemini request)

```bash
python services.py
```

The tests cover French accents, Arabic text/digits, unreadable input, conflicting
strengths, an unknown product, and same-brand strength disambiguation.

## Add catalogue products

Add one UTF-8 CSV row per package variant to `medications.csv`. Keep the existing
header and use a unique `id`; strength or package-size variants should be separate
rows. Fill Arabic columns when available, leave genuinely unavailable translations
blank, and restart the API after editing.

## Configuration

- `GEMINI_API_KEY`: secret key used only by FastAPI.
- `GEMINI_MODEL`: Gemini model ID; defaults to `gemini-3.1-flash-lite`.
- `API_URL`: FastAPI base URL used by Streamlit.
- `MAX_IMAGE_MB`: upload limit used by both UI and API; defaults to 10.

Gemini availability, pricing, and quotas can change. Check Google AI Studio for
the active limits on the specific project and model before a demo.
