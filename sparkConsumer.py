# command to run this file
# spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 --driver-class-path postgresql-42.7.4.jar --jars postgresql-42.7.4.jar sparkConsumer.py

# importing all libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, DoubleType, BooleanType
from pyspark.sql.functions import from_json, col, to_timestamp, unix_timestamp, lit, when, hour, dayofweek, udf

# creating the spark session with kafka and postgresql configuration
spark = SparkSession.builder \
    .appName("nycTaxi") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
    .config("spark.jars.packages", "/Users/rahul/PycharmProjects/nycTaxi/postgresql-42.7.4.jar") \
    .getOrCreate()

# configure spark logging
log4jLogger = spark._jvm.org.apache.log4j
rootLogger = log4jLogger.LogManager.getRootLogger()
rootLogger.setLevel(log4jLogger.Level.INFO)

# create file appender for logging
fileAppender = log4jLogger.FileAppender()
fileAppender.setName("FileLogger")
fileAppender.setFile("logs/sparkConsumer.log")
fileAppender.setLayout(log4jLogger.PatternLayout("%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n"))
fileAppender.setThreshold(log4jLogger.Level.INFO)
fileAppender.activateOptions()

# adding file appender to root logger
rootLogger.addAppender(fileAppender)
# creating logger
logger = spark._jvm.org.apache.log4j.LogManager.getLogger("nycTaxi")
logger.info("Starting nycTaxi Spark application")

# defining the schema for the data to be read from Kafka
schema = StructType([
    StructField("VendorID", StringType()),
    StructField("tpep_pickup_datetime", StringType()),
    StructField("tpep_dropoff_datetime", StringType()),
    StructField("passenger_count", DoubleType()),
    StructField("trip_distance", DoubleType()),
    StructField("RatecodeID", StringType()),
    StructField("store_and_fwd_flag", StringType()),
    StructField("PULocationID", StringType()),
    StructField("DOLocationID", StringType()),
    StructField("payment_type", StringType()),
    StructField("fare_amount", DoubleType()),
    StructField("extra", DoubleType()),
    StructField("mta_tax", DoubleType()),
    StructField("tip_amount", DoubleType()),
    StructField("tolls_amount", DoubleType()),
    StructField("improvement_surcharge", DoubleType()),
    StructField("total_amount", DoubleType()),
    StructField("congestion_surcharge", DoubleType()),
    StructField("Airport_fee", DoubleType())
])

# creating a streaming df from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "yellow_tripdata_2024-05") \
    .load()

# parsing the Kafka messages into a structured df
parsedDF = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
logger.info("Parsed Kafka messages into structured DataFrame")


# user defined function to categorize time of day
@udf(returnType=StringType())
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 16:
        return 'afternoon'
    elif 16 <= hour < 22:
        return 'evening'
    else:
        return 'late night'


# cleaning and transforming the data
cleanedDF = parsedDF \
    .withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"), "yyyy-MM-dd'T'HH:mm:ss")) \
    .withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"), "yyyy-MM-dd'T'HH:mm:ss")) \
    .withColumn("trip_duration",
                (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60) \
    .withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
    .withColumn("dropoff_hour", hour(col("tpep_dropoff_datetime"))) \
    .withColumn("pickup_day", dayofweek(col("tpep_pickup_datetime"))) \
    .withColumn("dropoff_day", dayofweek(col("tpep_dropoff_datetime"))) \
    .withColumn("pickup_timeofday", time_of_day(col("pickup_hour"))) \
    .withColumn("dropoff_timeofday", time_of_day(col("dropoff_hour"))) \
    .withColumn("pickup_is_weekend", when(col("pickup_day").isin(6, 7), lit(True)).otherwise(lit(False))) \
    .withColumn("fare_per_mile", when(col("trip_distance") == 0, 0).otherwise(col("fare_amount") / col("trip_distance"))) \
    .filter(col("passenger_count") != 0)

logger.info("Created cleaned DataFrame with additional features")


# function to write each batch of data to DB
def writeToDB(batch_df, batch_id):
    table_name = "cleaned_tripdata"
    jdbc_url = "jdbc:postgresql://localhost:5432/nyc"
    username = "rahul"
    password = ""

    try:
        # connecting to DB
        conn = spark._jvm.java.sql.DriverManager.getConnection(jdbc_url, username, password)

        # checking if the table exists
        stmt = conn.prepareStatement(
            f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')")
        rs = stmt.executeQuery()
        rs.next()
        table_exists = rs.getBoolean(1)
        rs.close()
        stmt.close()

        # creating the table if it does not exist
        if not table_exists:
            logger.info(f"Table {table_name} does not exist. Creating it!")
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            for field in cleanedDF.schema.fields:
                if isinstance(field.dataType, TimestampType):
                    create_table_sql += f"{field.name} TIMESTAMP,\n"
                elif isinstance(field.dataType, IntegerType):
                    create_table_sql += f"{field.name} INTEGER,\n"
                elif isinstance(field.dataType, DoubleType):
                    create_table_sql += f"{field.name} DOUBLE PRECISION,\n"
                elif isinstance(field.dataType, BooleanType):
                    create_table_sql += f"{field.name} BOOLEAN,\n"
                else:
                    create_table_sql += f"{field.name} VARCHAR(255),\n"
            create_table_sql = create_table_sql.rstrip(",\n") + "\n)"
            stmt = conn.prepareStatement(create_table_sql)
            stmt.executeUpdate()
            stmt.close()
            logger.info(f"Table {table_name} created successfully")

        # writing the batch data to DB
        row_count = batch_df.count()
        logger.info(f"Batch {batch_id}: Writing {row_count} rows to DB")

        batch_df.write \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", table_name) \
            .option("user", username) \
            .option("password", password) \
            .mode("append") \
            .save()

        logger.info(f"Batch {batch_id}: Written {row_count} rows to DB")

    except Exception as e:
        logger.error(f"An error occurred in batch {batch_id}: {str(e)}")

    finally:
        if conn:
            conn.close()


# starting the streaming query
query = cleanedDF.writeStream \
    .outputMode("append") \
    .foreachBatch(writeToDB) \
    .start()

logger.info("Started streaming query")

# waiting for the streaming query to terminate
query.awaitTermination()

logger.info("Streaming query terminated")
