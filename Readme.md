# NYC Yellow Taxi Trip Data Pipeline

This project implements a data pipeline for processing and analyzing NYC Yellow Taxi trip data. It includes data ingestion, cleaning, feature engineering, machine learning model training, and a REST API for fare amount prediction.

## Architecture

The project follows the following architecture:

1. **Data Ingestion**: The `kafkaProducer.py` script reads data from a Parquet file and publishes it to a Kafka topic.

2. **Data Processing**: The `sparkConsumer.py` script consumes data from the Kafka topic, performs data cleaning and feature engineering using PySpark, and writes the processed data to a PostgreSQL database.

3. **Model Training**: The `sparkML.py` script reads the processed data from the PostgreSQL database, trains a Random Forest Regression model using PySpark MLlib, and saves the trained model. It also uses MLflow for experiment tracking and logging.

4. **Prediction API**: The `main.py` script implements a FastAPI endpoint that loads the trained model, preprocesses input data, and returns fare amount predictions.

## Tech Stack

- Apache Kafka: Distributed streaming platform for data ingestion
- Apache Spark (PySpark): Distributed processing framework for data processing and machine learning
- PostgreSQL: Relational database for storing processed data
- MLflow: Platform for managing the machine learning lifecycle
- FastAPI: Web framework for building the prediction API
- pandas: Data manipulation library for handling input data in the API

## Setup and Running the Project

1. Start the Kafka server and create the necessary topic.

2. Start the PostgreSQL server and create a database named `nyc`.

3. Run the `kafkaProducer.py` script to publish data to the Kafka topic:
   ```
   python kafkaProducer.py
   ```

4. Run the `sparkConsumer.py` script to consume data from Kafka, process it, and write to PostgreSQL:
   ```
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 --driver-class-path postgresql-42.7.4.jar --jars postgresql-42.7.4.jar sparkConsumer.py
   ```

5. Run the `sparkML.py` script to train the machine learning model:
   ```
   spark-submit --driver-class-path postgresql-42.7.4.jar --jars postgresql-42.7.4.jar sparkML.py
   ```

6. Start the MLflow server to view the experiment runs:
   ```
   mlflow ui
   ```

7. Run the `main.py` script to start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

## Model Metrics

The trained Random Forest Regression model achieved the following metrics:

- Test RMSE: `2.6372`
- Test MAE: `0.5148`
- Test R-squared: `0.9800`

These metrics indicate good performance in predicting the fare amount based on the given features.

Here's the list of all Feature Importances:
- fare_amount: `0.3570`
- total_amount: `0.2598`
- trip_distance: `0.1704`
- trip_duration: `0.0533`
- ratecodeid: `0.0391`
- tip_amount: `0.0377`
- improvement_surcharge: `0.0370`
- dolocationid: `0.0108`
- pulocationid: `0.0055`
- payment_type: `0.0033`
- vendorid: `0.0011`
- passenger_count: `0.0008`
- pickup_timeofday_encoded: `0.0004`
- fare_per_mile: `0.0002`

## API Usage

To get fare amount predictions, send a POST request to the `/predict/` endpoint with a CSV file containing the trip data. The API will preprocess the data, apply the trained model, and return the predicted fare amounts.