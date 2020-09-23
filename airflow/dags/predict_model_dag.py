from datetime import timedelta

from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG
from scripts.classifier import XgbClassifier

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False
}
dag = DAG(
    'predict_model_dag',
    default_args=default_args,
    description='predict model DAG',
    schedule_interval=timedelta(days=1),
)

preprocess_predict_data = PythonOperator(
    task_id='preprocess_predict_data',
    provide_context=True,
    python_callable=XgbClassifier.preprocess_dataset,
    op_kwargs={'usecase': 'predict'},
    dag=dag,
)

generate_predictions = PythonOperator(
    task_id='predict_task',
    provide_context=True,
    python_callable=XgbClassifier.predict_model,
    dag=dag,
)

preprocess_predict_data >> generate_predictions