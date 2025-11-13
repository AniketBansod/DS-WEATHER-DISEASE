# Deploy to Render (Docker)

This project is containerized and ready to deploy on Render using the included `Dockerfile`.

## One-time setup

1. Push this branch (`ani`) to GitHub.
2. Ensure `models/` contains:
   - `weather_disease_model.joblib`
   - `feature_names.joblib`
   - `label_encoder.joblib`

## Option A — Render Blueprint (render.yaml)

1. In Render, click New > Blueprint.
2. Select this repo and branch `ani`.
3. Review the plan (Free) and create resources.
4. First build may take several minutes.

## Option B — Manual Web Service from Dockerfile

1. In Render, click New > Web Service.
2. Select this repo, branch `ani`, and choose Environment: Docker.
3. Use default build; Render will use `Dockerfile`.
4. Auto-Deploy: Yes.

## Local test (optional)

```powershell
# Build (ensure Docker Desktop is running)
docker build -t ds-weather-disease .
# Run locally on 8501
docker run --rm -p 8501:8501 ds-weather-disease
```

Then open http://localhost:8501.

## Notes

- Streamlit binds to `0.0.0.0` and uses `$PORT` provided by Render.
- Container uses minimal `requirements-prod.txt` to keep image small and builds fast.
- Large directories (`data/`, `outputs/`, `notebooks/`) are excluded via `.dockerignore`.
