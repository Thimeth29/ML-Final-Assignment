"""Airflow DAG (template)

Tasks required by assignment:
1) Data ingestion
2) Data validation
3) Feature engineering (preprocessing)
4) Model training
5) Model evaluation
6) Model registration

This template uses BashOperator to call your Python scripts inside the Airflow container.
It assumes your repo is mounted to: /opt/airflow/project (see docker-compose).
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

DEFAULT_ARGS = {"owner": "team", "retries": 0}

with DAG(
    dag_id="churn_mlops_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["churn", "mlops"],
) as dag:

    # TODO: optionally call src/data_ingestion.py to produce a quality report
    ingest = BashOperator(
        task_id="data_ingestion",
        bash_command="python /opt/airflow/project/src/data_ingestion.py --input /opt/airflow/project/data/raw/telco_customer_churn_data.csv --report_out /opt/airflow/project/reports/data_quality_report.json",
    )

    preprocess = BashOperator(
        task_id="preprocessing",
        bash_command="python /opt/airflow/project/src/preprocessing.py --input /opt/airflow/project/data/raw/telco_customer_churn_data.csv --outdir /opt/airflow/project/data/processed",
    )

    train = BashOperator(
        task_id="training",
        bash_command="python /opt/airflow/project/src/train.py --train /opt/airflow/project/data/processed/train.csv --model_out /opt/airflow/project/models/best_model.pkl",
    )

    evaluate = BashOperator(
        task_id="evaluation",
        bash_command="python /opt/airflow/project/src/evaluate.py --test /opt/airflow/project/data/processed/test.csv --model /opt/airflow/project/models/best_model.pkl --outdir /opt/airflow/project/reports",
    )

    # TODO: implement real model registration (MLflow Model Registry or "copy + tag best run")
    register = BashOperator(
        task_id="model_registration",
        bash_command="echo 'TODO: register model (MLflow registry or tagging)';",
    )

    ingest >> preprocess >> train >> evaluate >> register
