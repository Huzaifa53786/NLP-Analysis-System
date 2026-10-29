# Open Text Lab

A self-hosted, general-purpose NLP application built with Python, FastAPI, pandas, and scikit-learn. Explore text, train a classifier on your own labeled CSV, inspect held-out evaluation metrics, and classify new text. No account, external NLP API, fixed taxonomy, or pre-trained dataset is required.

## Features

- Text cleanup with Unicode normalization, whitespace normalization, and URL/email masking.
- Document, character, word, vocabulary, and frequent-term summaries.
- Configurable text and label columns for your own labeled CSV data.
- TF-IDF unigram and bigram features with Logistic Regression, Linear SVM, or Multinomial Naive Bayes.
- Reproducible stratified holdout evaluation with accuracy, per-label precision/recall/F1, and a confusion matrix.
- Batch prediction for up to 100 items, with probability confidence where supported by the selected estimator.
- JSON API, interactive browser interface, health endpoint, Docker image, and CI workflow.
- Request limits and input validation. Text is processed in memory and is not persisted by the application.

## Run locally

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[test]"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. API docs are at <http://127.0.0.1:8000/docs>; health check at <http://127.0.0.1:8000/health>.

## Train your own model

Prepare a UTF-8 CSV with a text column and a label column. Each usable row needs both values. Provide at least 10 examples across two or more labels; each label needs at least two examples for a stratified split. More balanced, representative examples generally produce more useful evaluations. Select the columns and estimator in the app, review the holdout metrics, then use the Predict tab.

`static/sample.csv` is a small example dataset to demonstrate the workflow, not a benchmark or model-quality promise. The train/test split is reproducible, but a single split on a small dataset can vary substantially. Avoid putting sensitive or regulated data into an instance you do not control. No data is stored by this app, but the server operator may independently collect infrastructure logs.

## JSON API

Analyze:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"texts":["The delivery was fast.","My package is late."],"top_n":10}'
```

Train with the contents of a CSV file as a JSON string:

```json
{
  "csv": "text,label\nHelpful and fast,positive\nBroken and late,negative\n...",
  "text_column": "text",
  "label_column": "label",
  "model": "Logistic Regression",
  "test_size": 0.2
}
```

Send the payload to `POST /api/train`. Pass the returned `model_id` to `POST /api/predict/{model_id}` with `{"texts":["A new message"]}`. Supported models are `Logistic Regression`, `Linear SVM`, and `Naive Bayes`. Models are held in process memory, only the five most recently used models are retained, and they disappear at restart. Run one server worker when using the built-in UI so training and predictions reach the same model registry. For a multi-instance production deployment, add a shared model store and user isolation/authentication to match your operating environment.

## Docker

```bash
docker compose up --build
```

Or build and run directly:

```bash
docker build -t open-text-lab .
docker run --rm -p 8000:8000 open-text-lab
```

The container runs as a non-root user and binds the app to port 8000. Set `HOST`/`PORT` through your platform's container settings as needed. The included deployment is intentionally a single-process instance with in-memory models; use persistent shared storage, authentication, and request throttling if your deployment needs cross-user privacy boundaries or long-lived models.

## Tests

```bash
python -m pip install -e ".[test]"
pytest
```

The test suite covers text normalization, descriptive analysis, CSV validation, end-to-end model training and prediction, input errors, API behavior, and the served UI.

## GitHub deployment

Push this folder to a GitHub repository. The GitHub Actions workflow installs the package and runs the test suite on pushes and pull requests. For deployment, use any Python 3.11+ host or build the included Docker image; set its start command to `uvicorn main:app --host 0.0.0.0 --port 8000`. No external credentials or secrets are required for the base app.

## Project layout

```text
main.py                      ASGI application entry point
nlp_text_analysis/core.py    Reusable NLP and model training functions
nlp_text_analysis/api.py     FastAPI routes and process-local model registry
static/index.html            Browser interface
static/sample.csv            Example labeled dataset
tests/                       Core and HTTP API tests
Dockerfile                   Container build
.github/workflows/ci.yml     GitHub Actions checks
```

## License

MIT. See [LICENSE](LICENSE).
