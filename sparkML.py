# command to run this file
# spark-submit --driver-class-path postgresql-42.7.4.jar --jars postgresql-42.7.4.jar sparkML.py

# command to start mlflow
# mlflow ui

# importing all libraries
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
import mlflow
import mlflow.spark

# setting up the logger
logging.basicConfig(filename="/Users/rahul/PycharmProjects/nycTaxi/logs/sparkML.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# creating the spark session
spark = SparkSession.builder \
    .appName("nycTaxi-ML") \
    .config("spark.jars.packages", "/Users/rahul/PycharmProjects/nycTaxi/postgresql-42.7.4.jar") \
    .getOrCreate()

logger.info("Spark session initialized")

# loading data from DB
jdbcURL = "jdbc:postgresql://localhost:5432/nyc"
tableName = "cleaned_tripdata"
username = "rahul"
password = ""

# creating a spark DF to fetch the data from DB
df = spark.read \
    .format("jdbc") \
    .option("url", jdbcURL) \
    .option("dbtable", tableName) \
    .option("user", username) \
    .option("password", password) \
    .load()

logger.info("Data loaded from PostgreSQL")

# removing all unnecessary columns
colsToDrop = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag', 'congestion_surcharge',
                   'airport_fee', 'mta_tax', 'extra', 'tolls_amount', 'dropoff_timeofday']
df = df.drop(*colsToDrop)

logger.info("Unnecessary columns removed")

# converting string columns to appropriate types
numericCols = ['vendorid', 'ratecodeid', 'pulocationid', 'dolocationid', 'passenger_count', 'trip_distance',
                   'fare_amount', 'tip_amount', 'improvement_surcharge', 'total_amount', 'trip_duration',
                   'payment_type']

# handling null values
for column in numericCols:
    df = df.withColumn(column, when(col(column).isNull(), 0).otherwise(col(column).cast(DoubleType())))

logger.info("Columns converted to appropriate types and null values handled")

# defining categorical columns
categoricalCols = ['pickup_timeofday']
# creating stages for the pipeline
stages = []

# string indexer and one hot encoder for categorical columns
for categoricalCol in categoricalCols:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + '_index', handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_encoded"])
    stages += [stringIndexer, encoder]

# preparing features
featureCols = numericCols + [c + "_encoded" for c in categoricalCols] + ['fare_per_mile']
assembler = VectorAssembler(inputCols=featureCols, outputCol="assembled_features", handleInvalid="keep")
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")
stages += [assembler, scaler]
logger.info("Pipeline stages created")

# log a single row of the data
sampleRow = df.limit(1).toPandas().to_dict('records')[0]
logger.info(f"Sample row of data: {sampleRow}")

# split the data
train, test = df.randomSplit([0.8, 0.2], seed=42)
logger.info("Data split into train and test sets")

# defining the Random Forest model
rf = RandomForestRegressor(labelCol="fare_amount", featuresCol="features", numTrees=100, maxDepth=10)

# setting up MLflow
mlflow.set_experiment("NYC Taxi Fare Prediction")
logger.info("MLflow experiment set up")


# function to evaluate model and log metrics
def evaluate_logMetrics(predictions, dataType):
    # creating the evaluators
    rmseEvaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
    maeEvaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mae")
    r2Evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")
    
    # computing the metrics
    rmse = rmseEvaluator.evaluate(predictions)
    mae = maeEvaluator.evaluate(predictions)
    r2 = r2Evaluator.evaluate(predictions)

    # logging the metrics
    logger.info(f"{dataType} RMSE: {rmse}")
    logger.info(f"{dataType} MAE: {mae}")
    logger.info(f"{dataType} R-squared: {r2}")

    mlflow.log_metric(f"{dataType}_rmse", rmse)
    mlflow.log_metric(f"{dataType}_mae", mae)
    mlflow.log_metric(f"{dataType}_r2", r2)
    return rmse, mae, r2


# function to train and evaluate the model
def train_evalModel(model, model_name):
    with mlflow.start_run():
        # assembling the pipeline
        rfPipeline = Pipeline(stages=stages + [model])
        logger.info(f"Starting training for {model_name}")

        # fitting the model
        pipelineModel = rfPipeline.fit(train)

        # logging the model's parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("numTrees", model.getNumTrees())
        mlflow.log_param("maxDepth", model.getMaxDepth())

        # making predictions on train data
        trainPredictions = pipelineModel.transform(train)
        # evaluate and log metrics for train data
        trainRMSE, trainMAE, trainR2 = evaluate_logMetrics(trainPredictions, "train")

        # making predictions on test data
        testPredictions = pipelineModel.transform(test)
        # evaluate and log metrics for test data
        testRMSE, testMAE, testR2 = evaluate_logMetrics(testPredictions, "test")

        # logging the model to mlflow
        mlflow.spark.log_model(pipelineModel, "model")
        logger.info("Model training and evaluation completed")

        return testRMSE, testMAE, testR2, pipelineModel


# training and evaluating the Random Forest model
rfRMSE, rfMAE, rfR2, rfModel = train_evalModel(rf, "RandomForest")

# fetching feature importances from the model
featureImportances = rfModel.stages[-1].featureImportances
featureNames = featureCols

# printing and log feature importances
logger.info("Feature Importances:")
for feature, importance in sorted(zip(featureNames, featureImportances), key=lambda x: x[1], reverse=True):
    logger.info(f"{feature}: {importance}")

# saving the trained model
modelSavePath = "/Users/rahul/PycharmProjects/nycTaxi/savedModels/"
rfModel.save(modelSavePath)
logger.info(f"Model saved to: {modelSavePath}")

# stopping the spark session
spark.stop()

logger.info("Spark session stopped")
