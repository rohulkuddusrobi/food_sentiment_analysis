# Customer Sentiment BI Dashboard

A Django dashboard that imports customer review data from CSV files, calculates
rating-derived sentiment analytics, and previews a pre-trained sentiment
ensemble on individual review texts.

The app is a local, single-user demo: it runs with `runserver`, SQLite, and a
bundled model artifact. It is not configured for production deployment.

## Contents

1. [Requirements](#1-requirements)
2. [Setup](#2-setup)
3. [Using the dashboard](#3-using-the-dashboard)
4. [CSV data schema](#4-csv-data-schema)
5. [Machine learning model](#5-machine-learning-model)
6. [scikit-learn compatibility](#6-scikit-learn-compatibility)
7. [Analytics definitions](#7-analytics-definitions)
8. [Performance notes](#8-performance-notes)
9. [Configuration and security](#9-configuration-and-security)
10. [Testing](#10-testing)
11. [Known limitations](#11-known-limitations)
12. [Demo checklist](#12-demo-checklist)
13. [Project structure](#13-project-structure)

## 1. Requirements

- Windows, macOS, or Linux
- Python 3.12 (the reference environment is Python 3.12.10)
- Packages from `requirements.txt` (Django 6.1.2, scikit-learn 1.9.1,
  numpy 2.5.3, scipy 1.18.1, joblib 1.6.0, pandas 3.0.6)

## 2. Setup

```powershell
# from the project root
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt   # macOS/Linux: pip install -r requirements.txt
venv\Scripts\python.exe manage.py migrate
venv\Scripts\python.exe manage.py runserver
```

Open <http://127.0.0.1:8000/>. The development database (`db.sqlite3`) starts
empty, so the dashboard shows its empty state until a CSV is uploaded.

To run the test suite:

```powershell
venv\Scripts\python.exe manage.py test
```

## 3. Using the dashboard

| Area | URL | What it shows |
| --- | --- | --- |
| Analytics (home) | `/` | KPI cards, sentiment doughnut, rating bars, monthly volume, sentiment-by-rating chart, generated insights, recent reviews |
| Data & ML - Upload | `/upload/` | CSV import with a per-file summary (processed / imported / duplicates / invalid / unrated) |
| Data & ML - Predict | `/predict/` | Single-text prediction from the saved ensemble plus the last 10 predictions |
| Review Explorer | `/reviews/` | Paginated, searchable, filterable review list with CSV export |
| Negative Feedback | `/negative/` | Rating-derived negative analysis: frequent terms, monthly negative volume, rating breakdown, insights |
| Exports | `/export/reviews.csv`, `/export/analytics.csv` | Streaming CSV downloads for the current filter |

Filters are GET parameters, so any filtered view can be bookmarked and shared:

- `q` - keyword over review text and summary
- `rating` - `1`..`5` or `none`
- `sentiment` - `positive`, `neutral`, `negative` (case-insensitive) or `none`
- `start_date` / `end_date` - inclusive calendar dates (`YYYY-MM-DD`)

### Importing the full dataset manually

1. Start the server and open `/upload/`.
2. Choose the CSV file and press **Upload and import**.
3. Wait for the summary card. For the reference dataset used during the audit
   (`Amazon_Reviews_Cleaned_Phase2.csv`, 21,055 data rows) the expected summary
   is: processed `21,055`, imported `20,407`, duplicates `648`, invalid `0`,
   unrated `0` (about 4 seconds).
4. Open `/` to see the populated dashboard.

Re-uploading the same file imports nothing new: duplicates are skipped by the
SHA-256 content hash of `product id + review text`.

The development database was left empty by the Phase 7 audit; the dataset is
imported only into throw-away temporary databases during verification.

## 4. CSV data schema

The first row must be a header row. Header matching is case-insensitive and
ignores surrounding whitespace. Each logical column accepts several header
names; the **first match in the order below wins**.

| Logical column | Accepted headers (in order) | Required | Notes |
| --- | --- | --- | --- |
| Review text | `Text`, `Review Text`, `Cleaned Review` | yes | Empty text rows are counted as invalid |
| Rating | `Score`, `Rating_Number`, `Rating` | no | Integer 1-5, `4.0`, or platform text such as `Rated 4 out of 5 stars` |
| Product id | `ProductId`, `Product Id`, `product_id` | no | Combined with the text for duplicate detection; empty is allowed |
| Review date | `Time`, `Review Date`, `review_date` | no | ISO 8601 (`2024-09-16T13:44:26.000Z`), unix seconds/millis, `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY-MM-DD HH:MM:SS` |
| Title / summary | `Summary`, `Review Title`, `Title` | no | Shown in the Review Explorer |

Import behaviour:

- UTF-8 with or without BOM, CRLF line endings, and quoted multi-line fields
  are supported; non-UTF-8 files fall back to Latin-1.
- Rows whose column count does not match the header are counted as invalid.
- Rows missing review text are counted as invalid.
- The file is streamed in 500-row batches; it is never loaded into memory as a
  whole and is never written into the project directory.
- The first 25 row errors are reported with their row numbers.
- The `Sentiment` column found in some datasets is deliberately ignored: the
  dashboard only derives sentiment from the rating (see below).

### Reference dataset

Verified against `Amazon_Reviews_Cleaned_Phase2.csv` (21,055 rows, 13 columns,
UTF-8 with BOM, 2007-2024):

| Check | Result |
| --- | --- |
| Processed / imported / duplicates | 21,055 / 20,407 / 648 |
| Invalid / unrated rows | 0 / 0 |
| Sentiment counts (derived) | Negative 14,157 - Neutral 825 - Positive 5,425 |
| Rating distribution (1-5) | 12,954 / 1,203 / 825 / 1,193 / 4,232 |
| Average rating | 2.14 |
| Earliest review date | 2007-08-27 |

All of the above were cross-checked with pandas computed independently from the
CSV, including the dashboard KPIs, charts, and the monthly volume series.

## 5. Machine learning model

- Artifact: `ml_models/sentiment_ensemble_model.joblib` (a dictionary holding a
  TF-IDF vectorizer, three classifiers, and the class labels).
- Ensemble: Logistic Regression + Linear SVM + SGD Classifier with hard voting;
  ties follow the saved class order.
- The application **loads the artifact lazily, caches it, and never retrains,
  rewrites, or replaces it.** Most tests use stand-in estimators; one
  integration test (`test_repeated_loads_reuse_cached_model`) loads the real
  artifact read-only, checks the cache, and runs a single prediction - it skips
  itself when the file is absent. Nothing in the codebase writes to the file
  (temporary artifacts are created under `TEMP` with `SENTIMENT_MODEL_PATH`
  overridden).
- The prediction page shows only the file name of the model, never the server
  path (this also applies to error messages).
- Ratings are never fed to the model: the model predicts from text only, and
  dashboard sentiment always comes from the rating.

## 6. scikit-learn compatibility

The artifact was trained with **scikit-learn 1.6.1**; the installed runtime is
**scikit-learn 1.9.1**. Loading it therefore emits five
`InconsistentVersionWarning` messages (one per unpickled estimator), and the
application surfaces a notice on `/predict/`:

> Model was trained with scikit-learn 1.6.1 but scikit-learn 1.9.1 is
> installed. Predictions may be unreliable; the saved model was not retrained
> or replaced.

Evidence collected during the audit (not a guarantee for future versions):

| Measurement | scikit-learn 1.6.1 | scikit-learn 1.9.1 |
| --- | --- | --- |
| Version warnings on load | 0 | 5 |
| Hard-vote labels for all 21,055 dataset texts | 14,055 Negative / 608 Neutral / 6,392 Positive | identical |
| Prediction fingerprint (SHA-256 of the label sequence) | `e9df66ac9a54806f` | `e9df66ac9a54806f` |
| Per-classifier label counts | Logistic 14,657/967/5,431 - LinearSVC 13,214/864/6,977 - SGD 12,448/118/8,489 | identical |

Conclusion for this dataset: **no prediction differences were observed** between
the training version and the installed version. scikit-learn still documents
cross-version loading as unsupported, so the warning stays in the UI.

To reproduce a matching environment (inference outputs only, verified here by
re-running the same comparison):

```powershell
python -m venv sk161
sk161\Scripts\pip.exe install scikit-learn==1.6.1 joblib
sk161\Scripts\python.exe <your prediction script>
```

Do not delete or overwrite `ml_models/sentiment_ensemble_model.joblib`; it is
the only trained artifact.

## 7. Analytics definitions

- Sentiment rule (fixed): rating 1-2 -> `Negative`, 3 -> `Neutral`,
  4-5 -> `Positive`; missing or out-of-range ratings have no sentiment and are
  excluded from sentiment percentages (they are reported as "missing rating").
- Percentage bases are stated in the UI: sentiment percentages use the count of
  reviews with a rating-derived sentiment.
- The monthly chart shows the most recent 24 months with a valid review date.
- Keyword search is case-insensitive over review text and summary.
- CSV exports: the reviews export carries the current Explorer filters; the
  analytics export carries the current home date filter. Cells starting with
  `=`, `+`, `-`, `@`, tab, or carriage return are prefixed with `'` so
  spreadsheets treat them as text.

## 8. Performance notes

Measured on a temporary database loaded with the real dataset (20,407 rows,
SQLite, median of 5 runs):

| Query | No index | With indexes | Change |
| --- | --- | --- | --- |
| Explorer first page (`ORDER BY created_at DESC LIMIT 20`) | 78.0 ms | 1.2 ms | -98.5% |
| Explorer deep page (offset 9,980) | 289.8 ms | 1.5 ms | -99.5% |
| Explorer count + sentiment-filtered slice | 80.1 ms | 21.5 ms | -73.2% |
| Date-range KPI aggregates | 168.1 ms | 34.7 ms | -79.3% |
| Monthly volume chart | 136.1 ms | 103.8 ms | -23.7% |
| Full dashboard assembly | 361.9 ms | 275.5 ms | -23.9% |
| Keyword search count (`LIKE '%term%'`) | 41.0 ms | 40.0 ms | -2.5% |

Because of those measurements the migration `0003` adds two indexes to
`Review`: `-created_at` (Explorer, previews, exports ordering) and
`review_date` (date-range filters and the monthly chart). `content_hash`
was already indexed for duplicate detection.

Page render times on the same data (test client, warm cache): `/` about
270-560 ms (12 queries), `/reviews/` about 21 ms, `/reviews/?page=500` about
19 ms, `/negative/?q=delivery` about 385 ms, `/export/reviews.csv` streams
about 11.5 MB in under a second.

Keyword search stays a full-table scan (see limitations).

## 9. Configuration and security

Settings read three optional environment variables; the defaults keep the local
demo working without any configuration:

| Variable | Default | Effect |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | insecure development key shipped in `config/settings.py` | Django signing key |
| `DJANGO_DEBUG` | `true` | Set to `0`/`false`/`no`/`off` to disable debug |
| `DJANGO_ALLOWED_HOSTS` | empty | Comma-separated hosts; **required when debug is off** |

Notes from the audit:

- This project is **not a git repository**; nothing has been committed. The
  provided `.gitignore` already excludes `venv/`, `db.sqlite3`, `.env*`,
  `*.csv`, `*.joblib` (model artifacts), `media/`, and `__pycache__/`.
- `.env` files are ignored but also **not read automatically**; set variables in
  the shell when you need them.
- Both POST forms (`/upload/`, `/predict/`) include CSRF tokens; all other forms
  are GET search/filter forms.
- The SQLite database, uploads, and export downloads contain user data; exports
  are streamed from the database and are not written to disk.
- The admin site is available at `/admin/` but no user accounts exist; create a
  superuser only if you need it (`python manage.py createsuperuser`).
- Static and media files are only served by Django while debug is on.
- The compatibility warning and import errors are rendered as plain text (Django
  auto-escaping); the model path is reduced to its file name in both the status
  block and flash messages.
- `python manage.py check --deploy` reports 1 error and 7 warnings
  (development console email backend, insecure shipped key, debug on, empty
  `ALLOWED_HOSTS`, no HSTS/SSL redirect/secure cookies). All of them are
  expected for a local demo and are not fixed on purpose.
- This is a local academic demo. Do not expose it to the internet with the
  shipped key and debug enabled.

## 10. Testing

```powershell
venv\Scripts\python.exe manage.py test           # 197 tests
venv\Scripts\python.exe manage.py check
venv\Scripts\python.exe manage.py makemigrations --check --dry-run
```

The suite covers sentiment mapping, CSV import (including the reference dataset
headers, duplicates, malformed rows, batching, encodings), analytics and
insights, filtering, pagination, negative analysis, CSV exports and formula
sanitisation, prediction history, model status/error handling, message
rendering, and UI navigation.

Phase 7 additionally verified, against the real dataset in temporary databases
(the development database was not touched):

- full import summary and pandas cross-checks for counts, sentiment
  distribution, rating distribution, average rating, dates, and the monthly
  series (27/27 checks passed),
- page rendering for all routes plus a filtered, paginated Explorer request,
- end-to-end prediction through the web form with the real artifact (positive,
  negative, and neutral examples, history stored, warning rendered, no server
  path exposed),
- the scikit-learn 1.6.1 vs 1.9.1 comparison described above.

Pages were exercised with Django's test client. **No manual browser walkthrough
was performed**, so the final visual screenshots still need to be captured after
the dataset is imported.

## 11. Known limitations

1. **Model/runtime version mismatch** - loading a 1.6.1 model on 1.9.1 is
   unsupported by scikit-learn. Predictions were identical for this dataset,
   but the warning must stay until the environment or the artifact is aligned.
2. **Keyword search has no full-text index** - `q` uses `LIKE '%term%'` over
   text and summary, which scans the table (about 40 ms at 20k rows; grows
   linearly with the data).
3. **Sentiment is rating-derived only** - the dataset's own `Sentiment` column
   and the model's predictions are never mixed into dashboard numbers, so
   analytics and the ML tool intentionally answer different questions.
4. **Monthly chart window** - only the last 24 months are charted (the dataset
   spans 184 months); the underlying data is not truncated.
5. **Model quality is out of scope** - the artifact is fixed, and some
   neutral-sounding sentences were observed to be voted `Negative` (for
   example "Received the order today, everything seems to be in order.").
6. **Empty development database** - the demo database holds 0 rows; the import
   has to be repeated after cloning/resetting the project.
7. **No upload size cap** - the importer streams rows, but a very large file
   still spends time in the request; there is no configurable maximum size.
8. **Local-only security posture** - debug on by default, insecure shipped key,
   empty `ALLOWED_HOSTS`, SQLite, no authentication on the dashboard itself.
9. **Duplicate `Models/` folder** - an identical copy of the artifact exists in
   `Models/`; the application only uses `ml_models/`. Both are excluded from
   version control by `*.joblib`.
10. **No visual verification** - functional checks used the Django test client;
    screenshots for the thesis write-up are still outstanding.

## 12. Demo checklist

Ordered steps for the thesis demonstration:

1. `python -m venv venv` and `venv\Scripts\pip.exe install -r requirements.txt`
2. `venv\Scripts\python.exe manage.py migrate`
3. `venv\Scripts\python.exe manage.py test` - expect `197 tests ... OK`
4. `venv\Scripts\python.exe manage.py runserver`
5. Open `/` - empty-state message, zeroed KPIs, no charts data
6. Open `/upload/`, import the dataset - expect processed 21,055, imported
   20,407, duplicates 648, invalid 0, unrated 0
7. Open `/` again - total 20,407 reviews; Negative 14,157, Neutral 825,
   Positive 5,425; average rating 2.14; rating bars
   12,954 / 1,203 / 825 / 1,193 / 4,232; monthly window 2022-10 to 2024-09;
   four generated insights
8. Apply a date range on `/` and confirm the KPIs, charts, insights, and the
   analytics export all narrow together
9. Open `/reviews/` - filter by sentiment and keyword, page through, check the
   match count and the "showing X-Y of Z" text
10. Download `/export/reviews.csv` and `/export/analytics.csv`; open them in a
    spreadsheet and confirm text cells beginning with `=` stay inert
11. Open `/negative/` - frequent terms, monthly negative volume, rating
    breakdown, and the method note
12. Open `/predict/` - confirm the model status card and the scikit-learn
    1.6.1/1.9.1 warning; submit one positive, one negative, and one neutral
    text and check the labels plus the history table
13. Capture screenshots (upload summary, home, explorer, negative, predict)
    - **the final screenshots cannot be taken until step 6 has been done**
14. Stop and report results; do not claim a visual check that was not performed

## 13. Project structure

```
config/                 Django project (settings, urls, wsgi/asgi)
dashboard/
  analytics.py          KPI, chart, and insight calculations
  csv_import.py         streaming batched CSV import with header aliases
  csv_export.py         streaming CSV exports with formula sanitisation
  forms.py              upload, prediction, and filter forms
  ml_service.py         lazy model loading, status, hard-vote prediction
  models.py             Review and PredictionHistory models, indexes
  negative_analysis.py  negative review analysis and frequent terms
  queries.py            shared ORM filter helpers
  views.py              page views and CSV download endpoints
  tests.py              197 tests
  migrations/           0001 initial, 0002 prediction history, 0003 indexes
ml_models/              sentiment_ensemble_model.joblib (trained artifact)
static/css/             dashboard stylesheet
templates/              base layout and page templates
db.sqlite3              development database (starts empty)
requirements.txt        pinned runtime dependencies
.gitignore              excludes venv, database, datasets, artifacts, .env
```
