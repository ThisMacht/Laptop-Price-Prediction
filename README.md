# Laptop Price Prediction

A machine-learning web application that estimates a laptop's market price from its hardware specifications and condition. Users can enter specifications manually or describe a laptop in natural language; the app returns a point estimate and an uncertainty-aware price range in million VND. It can also compare several laptop listings and rank their value.

## Model overview

The production model is a tuned **CatBoost regressor** trained on 7,296 cleaned laptop listings. The feature pipeline converts 14 user-facing fields—such as brand, model, RAM, storage, CPU, GPU, condition, and warranty—into 86 model features.

- Target: laptop price
- Output unit: million VND
- Production model: `models/final_laptop_price_model_full_data.joblib`
- Holdout reference: RMSE 5.91, MAE 3.37, and R² 0.85
- Natural-language input: Gemini extracts structured specifications before local prediction

The reported metrics come from the held-out evaluation performed before the final model was retrained on the full dataset. Predictions are estimates, not guaranteed resale prices; uncertainty is generally higher for expensive or unusual laptops.

## Project structure

```text
app/
├── backend/             # Python HTTP API and prediction service
└── frontend/            # HTML, CSS, and JavaScript interface
models/                  # Production model, schema, and interval config
src/                     # Encoding, feature, and prediction modules
notebooks/               # Data preparation and modeling experiments
data/                    # Raw, intermediate, and processed datasets
tests/                   # Automated tests
```

## Requirements

- Python 3.10 or newer
- `pip`
- A Gemini API key only for natural-language prediction and comparison

## Setup

Clone the repository and move into it:

```bash
git clone <repository-url>
cd Laptop-Price-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For natural-language input, create the backend environment file:

```bash
cp app/backend/.env.example app/backend/.env
```

Then replace the placeholder in `app/backend/.env`:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

The manual specification form does not require Gemini or an API key.

## Run the backend and frontend together

From the repository root, run:

```bash
python app/backend/server.py
```

Open <http://127.0.0.1:8000> in a browser. The same server provides both the frontend and API, so this is the recommended development setup.

Check that the backend is ready:

```bash
curl http://127.0.0.1:8000/api/health
```

Stop the server with `Ctrl+C`.

## Run the frontend and backend separately

Start the backend from the repository root:

```bash
python app/backend/server.py
```

In a second terminal, serve the frontend:

```bash
cd app/frontend
python -m http.server 3000
```

Then open <http://127.0.0.1:3000>. The frontend automatically connects to the backend at `http://127.0.0.1:8000`.

Do not open `index.html` directly with a `file://` URL; use one of the HTTP server commands above.

## API examples

Manual prediction:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "manual",
    "raw_features": {
      "brand": "Dell",
      "model": "Inspiron",
      "ram_gb": 16,
      "storage_gb": 512,
      "storage_type": "SSD",
      "screen_size_inch": 15.6,
      "cpu_text": "Intel Core i5-1235U",
      "cpu_brand": "Intel",
      "cpu_family": "Intel Core i5",
      "cpu_generation": 12,
      "cpu_suffix": "U",
      "gpu_text": "Intel Iris Xe",
      "condition": "good",
      "warranty_status": "expired"
    }
  }'
```

Natural-language prediction (requires `GEMINI_API_KEY`):

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "text",
    "description": "Dell Inspiron 15, Core i5-1235U, 16GB RAM, 512GB SSD, used"
  }'
```

## Tests

Run the test suite from the repository root:

```bash
pytest -q
```

## License

See [LICENSE](LICENSE).
