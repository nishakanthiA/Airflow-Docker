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
    'train_model_dag',
    default_args=default_args,
    description='train model DAG',
    schedule_interval=None,
)

preprocess_train_data = PythonOperator(
    task_id='preprocess_task',
    provide_context=True,
    python_callable=XgbClassifier.preprocess_dataset,
    op_kwargs={'usecase': 'train'},
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_task',
    provide_context=True,
    python_callable=XgbClassifier.train_model,
    dag=dag,
)

preprocess_train_data >> train_model