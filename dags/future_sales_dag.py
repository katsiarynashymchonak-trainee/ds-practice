from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import time
import os

from pipe.src.future_sales_inno_ds_project.pipeline import (
    prepare_data,
    train_and_evaluate,
    log_to_neptune,
    initialize_neptune_run,
)
from pipe.src.future_sales_inno_ds_project.config import (
    USE_OPTUNA, OPTUNA_SPACES, HYPEROPT_SPACES, LOG_TARGET,
)

import pandas as pd
from dotenv import load_dotenv

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

default_args = {
    "owner": "ekaterina",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="future_sales_pipeline",
    default_args=default_args,
    description="Multi-step DAG for future sales prediction",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["future_sales", "ml"],
) as dag:

    dvc_pull = BashOperator(
        task_id="dvc_pull_data",
        bash_command="dvc pull data/ --jobs 4 --quiet",
        cwd="/opt/airflow"
    )

    def _prepare_data(**context):
        start = time.time()
        x, y, x_test = prepare_data()
        x.to_parquet("/tmp/x.parquet")
        y.to_parquet("/tmp/y.parquet")
        x_test.to_parquet("/tmp/x_test.parquet")
        context['ti'].xcom_push(key='x_path', value="/tmp/x.parquet")
        context['ti'].xcom_push(key='y_path', value="/tmp/y.parquet")
        context['ti'].xcom_push(key='x_test_path', value="/tmp/x_test.parquet")
        logger.info(f"Data preparation completed in {time.time() - start:.2f} seconds")

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=_prepare_data,
    )

    def _train_and_evaluate(**context):
        start = time.time()
        x = pd.read_parquet(context['ti'].xcom_pull(key='x_path'))
        y = pd.read_parquet(context['ti'].xcom_pull(key='y_path'))
        x_test = pd.read_parquet(context['ti'].xcom_pull(key='x_test_path'))

        if LOG_TARGET:
            from pipe.src.future_sales_inno_ds_project.data_preparation.data_preprocessor import DataPreprocessor
            y = DataPreprocessor.log_transform(y)

        run = initialize_neptune_run(NEPTUNE_API_TOKEN)
        spaces = OPTUNA_SPACES if USE_OPTUNA else HYPEROPT_SPACES
        trainer, rmse = train_and_evaluate(run, x, y, x_test, spaces)

        context['ti'].xcom_push(key='trainer', value=trainer)
        context['ti'].xcom_push(key='rmse', value=rmse)
        context['ti'].xcom_push(key='neptune_run_id', value=run["sys/id"].fetch())
        logger.info(f"Model training completed in {time.time() - start:.2f} seconds")

    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=_train_and_evaluate,
    )

    def _log_to_neptune(**context):
        from neptune import init_run
        run_id = context['ti'].xcom_pull(key='neptune_run_id')
        run = init_run(project="katsiaryna.shymchonak/Future-sales", api_token=NEPTUNE_API_TOKEN, with_id=run_id)
        trainer = context['ti'].xcom_pull(key='trainer')
        rmse = context['ti'].xcom_pull(key='rmse')
        log_to_neptune(run, trainer, rmse)
        run.stop()
        logger.info("Neptune logging completed")

    log = PythonOperator(
        task_id="log_to_neptune",
        python_callable=_log_to_neptune,
    )

    dvc_pull >> prepare >> train >> log
