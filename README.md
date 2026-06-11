# MoneyTalks — Rupiah Banknote Detector

MoneyTalks is a web application that recognises Indonesian Rupiah banknotes from a
live camera feed and announces the denomination out loud in Indonesian. It is built
as an accessibility aid for blind and visually-impaired users, who can point their
phone at a banknote and hear the amount spoken back to them.

The recognition engine fuses two signals:

1. **Computer vision** — ORB keypoint features + an HSV colour histogram are encoded
   into a Bag-of-Visual-Words (BoVW) representation, TF-IDF weighted, and classified
   by an SVM.
2. **OCR** — Tesseract reads the printed numerals on the note, and a fuzzy
   denomination matcher (Levenshtein-based) cross-checks the SVM's guess.

A decision-fusion layer combines both predictions to resolve the denominations that
are most easily confused (e.g. `2.000` vs `20.000`, `10.000` vs `100.000`).

---

## Features

- **Live detection** (`/`) — stream camera frames to `/api/detect` and get back the
  denomination, a confidence score, and a bounding box.
- **Indonesian text-to-speech** (`/api/tts`) — denomination spoken via gTTS
  (e.g. *"Lima Puluh Ribu Rupiah"*).
- **Confidence gating** — low-confidence detections prompt the user to move the note
  closer instead of guessing.
- **Admin panel** (`/admin`) — bcrypt-authenticated dashboard to:
  - browse and download stored scan images (date-filterable, ZIP export),
  - upload new model bundles (`.zip` of the three `.pkl` files),
  - hot-swap the deployed model **without restarting the server**.
- **Supabase backend** — scan images, scan metadata, admin accounts, and model
  versions are persisted to Supabase (Postgres + Storage).

## Supported denominations

| Label        | Amount (IDR) | Spoken                  |
|--------------|--------------|-------------------------|
| `idr_1000`   | 1.000        | Seribu Rupiah           |
| `idr_2000`   | 2.000        | Dua Ribu Rupiah         |
| `idr_5000`   | 5.000        | Lima Ribu Rupiah        |
| `idr_10000`  | 10.000       | Sepuluh Ribu Rupiah     |
| `idr_20000`  | 20.000       | Dua Puluh Ribu Rupiah   |
| `idr_50000`  | 50.000       | Lima Puluh Ribu Rupiah  |
| `idr_100000` | 100.000      | Seratus Ribu Rupiah     |

---

## Tech stack

- **Backend:** Flask, Flask-Login, Flask-WTF (CSRF)
- **CV / ML:** OpenCV, scikit-learn (SVM + KMeans BoVW), NumPy, joblib
- **OCR:** pytesseract (Tesseract OCR)
- **TTS:** gTTS
- **Storage / DB:** Supabase
- **Auth:** bcrypt
- **Testing:** pytest

---

## Project structure

```
MoneyTalks/
├── Web Application/
│   ├── app.py                 # Flask routes (guest + admin)
│   ├── inference.py           # CV + OCR pipeline & decision fusion
│   ├── supabase_client.py     # All DB / Storage operations
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/                # Deployed model bundle (.pkl files)
│   │   ├── svm_model.pkl
│   │   ├── bovw_dictionary.pkl
│   │   └── tfidf_scaler.pkl
│   ├── templates/             # index, test, admin/*
│   ├── static/css/
│   └── tests/                 # pytest suite
└── ML Model/
    └── models/
        ├── baseline/          # baseline model bundle
        └── proposed/          # proposed (improved) model bundle
```

---

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** installed on the host:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: install to `C:\Program Files\Tesseract-OCR\` (the path
    `inference.py` expects on Windows)
- A **Supabase** project (for the admin panel and scan persistence). The live
  detector and TTS work without it; Supabase-backed endpoints return `503` when
  unconfigured.

---

## Setup

```bash
cd "Web Application"

# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# then edit .env with your values
```

### Environment variables (`.env`)

```ini
# Supabase — Settings → API in your project
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_SERVICE_KEY=<your-service-role-key>

# Flask session signing key
# generate with: python -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=<your-flask-secret-key>
```

Optional:

- `CONFIDENCE_THRESHOLD` — minimum confidence to accept a detection (default `0.75`).

### Supabase schema

The app expects these tables and storage buckets:

**Tables**
- `ScannedMoney` — `id`, `denomination`, `confidence`, `image_path`, `scanned_at`
- `Administrator` — `id`, `email`, `password_hash` (bcrypt)
- `ModelVersions` — `id`, `version_string`, `file_path`, `uploaded_by`,
  `is_deployed`, `uploaded_at`

**Storage buckets**
- `scanned-images` — captured scan frames
- `model-files` — uploaded model `.zip` bundles

> Create an admin by inserting a row into `Administrator` with a bcrypt hash:
> `python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"`

---

## Running

```bash
cd "Web Application"
python app.py
```

The server starts at **http://localhost:5003**.

| Route           | Purpose                          |
|-----------------|----------------------------------|
| `/`             | Live banknote detector (main UI) |
| `/test`         | Detection test page              |
| `/admin/login`  | Admin login                      |
| `/admin`        | Admin dashboard                  |

> The dev server runs with `debug=True`. Use a production WSGI server
> (e.g. gunicorn) for deployment.

---

## API

| Method | Endpoint            | Description                                              |
|--------|---------------------|----------------------------------------------------------|
| POST   | `/api/detect`       | `frame` (image file) → `{ message, confidence, valid, box }` |
| POST   | `/api/upload-image` | Persist a scan to Supabase (`frame`, `denomination`, `confidence`) |
| GET    | `/api/tts?text=...` | Returns Indonesian speech audio (`audio/mpeg`)           |

---

## Model bundles

A model bundle is a `.zip` containing exactly three files:

- `svm_model.pkl` — trained SVM classifier
- `bovw_dictionary.pkl` — KMeans visual-word dictionary
- `tfidf_scaler.pkl` — TF-IDF transformer

Upload a bundle via **Admin → Models**, then click **Deploy**. The server validates
the archive, hot-swaps the in-memory models, and copies the new `.pkl` files into
`models/` so the change survives restarts.

The `ML Model/models/` directory ships a `baseline/` and a `proposed/` bundle for
comparison.

---

## Testing

```bash
cd "Web Application"
pytest
```

The suite covers OCR normalisation, Levenshtein distance, denomination candidate
extraction (exact + fuzzy), image preprocessing, BoVW histogram generation, ORB +
colour feature extraction, and the `predict_currency` output contract (with mocked
models).

---

## How detection works

1. Decode the incoming frame and locate the note via ORB keypoint density,
   producing a cropped region + normalised bounding box.
2. **SVM path:** extract ORB descriptors → BoVW histogram → TF-IDF → fuse with the
   HSV colour histogram → SVM `predict` + `predict_proba`.
3. **OCR path:** run Tesseract over the crop (and a 180°-rotated / full-frame
   fallback), normalise the text, and score denomination candidates — exact matches
   plus fuzzy Levenshtein matches.
4. **Fusion:** if OCR agrees with the SVM, confidence is boosted; if they disagree,
   denomination-specific rules decide whether OCR overrides the SVM (targeting the
   commonly-confused pairs).
5. Responses below `CONFIDENCE_THRESHOLD` ask the user to move the note closer.

---

## License

Academic project — BINUS University, Software Engineering.
