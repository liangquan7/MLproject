# Dating App Match Outcome Predictor

A machine learning web app that predicts dating app match outcomes using a FastAPI backend and Next.js frontend.

## How to Run the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/liangquan7/MLproject.git
cd MLproject
```

### 2. Set up Python environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost fastapi uvicorn pydantic flaml
```

### 3. Generate model files

Run these from the root project folder:

```bash
python preprocessing_pipeline.py
python model_training.py
```

This will generate:

```text
preprocessing_artifacts.pkl
trained_models.pkl
```

These files are required for the backend to work.

### 4. Start the backend

```bash
uvicorn app:app --reload --port 8000
```

Backend URL:

```text
http://localhost:8000
```

### 5. Start the frontend

Open a new terminal:

```bash
cd dating-predictor
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

### 6. Use the app

Open:

```text
http://localhost:3000
```

Fill in the form and click **Predict My Match Outcome**.

## Notes

- Keep the backend running on port `8000`.
- Keep the frontend running on port `3000`.
- If the frontend shows “Failed to fetch”, make sure the backend is running.
- The `.pkl` model files are generated locally and are not stored in GitHub.
