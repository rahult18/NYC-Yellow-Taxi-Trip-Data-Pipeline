# command to start the backend server
# uvicorn main:app --reload

# import all libraries
from fastapi import FastAPI, File, UploadFile
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, hour, dayofweek, udf, unix_timestamp
from pyspark.sql.types import DoubleType, StringType
import pandas as pd
import io
import uvicorn

# initialising the Fast API 
app = FastAPI()
# creating the spark session
spark = SparkSession.builder.appName("nycTaxi-Prediction").getOrCreate()
# loading the saved model
model = PipelineModel.load("/Users/rahul/PycharmProjects/nycTaxi/savedModels/")


# user defined function to categorize time of day
@udf(returnType=StringType())
def getTimeOfDay(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 16:
        return 'afternoon'
    elif 16 <= hour < 22:
        return 'evening'
    else:
        return 'late night'


def preprocessData(df):
    # converting columns to appropriate data types
    numericCols = ['vendorid', 'ratecodeid', 'pulocationid', 'dolocationid', 'passenger_count', 'trip_distance',
                   'fare_amount', 'tip_amount', 'improvement_surcharge', 'total_amount', 'payment_type']
    for column in numericCols:
        # handling null values
        df = df.withColumn(column, when(col(column).isNull(), 0).otherwise(col(column).cast(DoubleType())))

    # feature engineering
    df = df.withColumn("fare_per_mile",
                       when(col("trip_distance") == 0, 0).otherwise(col("fare_amount") / col("trip_distance")))
    df = df.withColumn("trip_duration", (
                unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60)
    df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
    df = df.withColumn("pickup_day", dayofweek(col("tpep_pickup_datetime")))
    df = df.withColumn("pickup_timeofday", getTimeOfDay(col("pickup_hour")))

    # selecting only the features used in the model
    feature_cols = ['vendorid', 'ratecodeid', 'pulocationid', 'dolocationid', 'passenger_count', 'trip_distance',
                    'fare_amount', 'tip_amount', 'improvement_surcharge', 'total_amount', 'trip_duration',
                    'payment_type', 'fare_per_mile', 'pickup_timeofday']

    return df.select(feature_cols)


# creating a route for model's predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # reading the CSV file from frontend
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    # converting pandas df to spark dataframe
    sparkDF = spark.createDataFrame(df)
    # preprocessing the data
    preprocessedDF = preprocessData(sparkDF)
    # fetch predictions
    predictions = model.transform(preprocessedDF)
    # returning only the fare_amount and prediction columns
    result_df = predictions.select('fare_amount', 'prediction')
    # converting spark df back to pandas df
    result_pandas = result_df.toPandas()
    return result_pandas.to_dict(orient='records')


if __name__ == "__main__":
    # running this only when main is executed
    uvicorn.run(app, host="0.0.0.0", port=8000)
