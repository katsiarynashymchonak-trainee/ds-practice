# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.utils.dates import days_ago
# from pipe.src.model_entrypoint import get_neptune_token
# from pipe.src.stages import (
#     prepare_data, load_data, train_model, evaluate_model,
#     predict, log_metadata, log_environment, log_dvc
# )
#
# with DAG(
#     dag_id="future_sales_pipeline",
#     start_date=days_ago(1),
#     schedule_interval="@monthly",
#     catchup=False,
# ) as dag:
#
#     def init_run():
#         token = get_neptune_token()
#         run = neptune.init_run(project="...", api_token=token)
#         return run
#
#     def run_prepare():
#         prepare_data()
#
#     def run_load(**kwargs):
#         x, y, x_test = load_data()
#         kwargs['ti'].xcom_push(key="x", value=x)
#         kwargs['ti'].xcom_push(key="y", value=y)
#         kwargs['ti'].xcom_push(key="x_test", value=x_test)
#
#     def run_train(**kwargs):
#         run = kwargs['ti'].xcom_pull(task_ids="init_run")
#         x = kwargs['ti'].xcom_pull(key="x")
#         y = kwargs['ti'].xcom_pull(key="y")
#         trainer = train_model(x, y, run)
#         kwargs['ti'].xcom_push(key="trainer", value=trainer)
#
#     def run_eval(**kwargs):
#         run = kwargs['ti'].xcom_pull(task_ids="init_run")
#         trainer = kwargs['ti'].xcom_pull(key="trainer")
#         x = kwargs['ti'].xcom_pull(key="x")
#         y = kwargs['ti'].xcom_pull(key="y")
#         rmse = evaluate_model(trainer, x, y, run)
#         kwargs['ti'].xcom_push(key="rmse", value=rmse)
#
#     def run_predict(**kwargs):
#         trainer = kwargs['ti'].xcom_pull(key="trainer")
#         x_test = kwargs['ti'].xcom_pull(key="x_test")
#         predict(trainer, x_test)
#
#     def run_log_meta(**kwargs):
#         trainer = kwargs['ti'].xcom_pull(key="trainer")
#         run = kwargs['ti'].xcom_pull(task_ids="init_run")
#         rmse = kwargs['ti'].xcom_pull(key="rmse")
#         log_metadata(trainer, run, rmse)
#
#     def run_log_env(**kwargs):
#         run = kwargs['ti'].xcom_pull(task_ids="init_run")
#         log_environment(run)
#
#     def run_log_dvc(**kwargs):
#         run = kwargs['ti'].xcom_pull(task_ids="init_run")
#         log_dvc(run)
#         run.stop()
#
#     # Define tasks
    # init_run_task = PythonOperator(task_id="init_run", python_callable=init_run)
    # prepare_task = PythonOperator(task_id="prepare_data", python_callable=run_prepare)
    # load_task = PythonOperator(task_id="load_data", python_callable=run_load)
    # train_task = PythonOperator(task_id="train_model", python_callable=run_train)
    # eval_task = PythonOperator(task_id="evaluate_model", python_callable=run_eval)
    # predict_task = PythonOperator(task_id="predict", python_callable=run_predict)
    # log_meta_task = PythonOperator(task_id="log_metadata", python_callable=run_log_meta)
    # log_env_task = PythonOperator(task_id="log_environment", python_callable=run_log_env)
    # log_dvc_task = PythonOperator(task_id="log_dvc", python_callable=run_log_dvc)
#
#     # Set dependencies
    # init_run_task >> prepare_task >> load_task >> train_task >> eval_task >> predict_task
    # predict_task >> log_meta_task >> log_env_task >> log_dvc_task
