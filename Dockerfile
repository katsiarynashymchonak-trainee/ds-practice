# Use the official Python 3.11 slim image as base
FROM python:3.11-slim

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install dvc[gs,gdrive]
RUN pip install --no-cache-dir "apache-airflow==2.7.2" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.2/constraints-3.11.txt"

RUN pip check || true
RUN pip cache purge

# Create DAGs directory and copy DAG
RUN mkdir -p $AIRFLOW_HOME/dags
COPY dags/future_sales_dag.py $AIRFLOW_HOME/dags/

EXPOSE 8080

# Let docker-compose decide what to run
CMD ["bash"]
