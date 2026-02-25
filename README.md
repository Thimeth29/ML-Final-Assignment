# churn-mlops-project (Starter)

This is a **starter skeleton** for the Machine Learning Final Assignment:
**End-to-End Customer Churn Prediction with Full MLOps Pipeline** (Git + DVC + MLflow + Airflow + DAGsHub + API + Docker).

✅ This repo contains:
- A clean folder structure
- Safe, commented **templates** (not a finished solution)
- Placeholders for DVC pipeline, MLflow logging, Airflow DAG, and FastAPI API

## Quick start (Windows)
1) Create and activate venv
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Put the dataset here:
- `data/raw/telco_customer_churn_data.csv`

3) Initialize Git + DVC
```bat
git init
dvc init
dvc add data\raw\telco_customer_churn_data.csv
git add .
git commit -m "init structure + dvc tracking"
```

4) Run preprocessing (after you implement TODOs)
```bat
python src\preprocessing.py --input data\raw\telco_customer_churn_data.csv --outdir data\processed
```

5) Train + evaluate (after you implement TODOs)
```bat
python src\train.py --train data\processed\train.csv --model_out models\best_model.pkl
python src\evaluate.py --test data\processed\test.csv --model models\best_model.pkl --outdir reports
```

6) Run API (after you implement)
```bat
uvicorn api.main:app --reload
```
Then open:
- http://127.0.0.1:8000/docs

## Suggested branch names
- feature/data
- feature/model
- feature/dvc
- feature/mlflow-dagshub
- feature/airflow
- feature/api-docker

## Notes
- Airflow is best run via Docker on Windows. See `airflow/README_AIRFLOW_DOCKER.md`.
- This starter intentionally includes TODO markers to guide your group work.
