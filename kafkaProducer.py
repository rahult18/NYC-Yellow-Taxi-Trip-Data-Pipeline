import pandas as pd
from kafka import KafkaProducer
import json
import pyarrow.parquet as pq
import os
from datetime import datetime
import time

# fetching current time now
start = time.time()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


# kafka config
# this serializer receives the row as a dict and then converts into a JSON dump
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v, cls=DateTimeEncoder).encode('utf-8'))
path = '/Users/rahul/PycharmProjects/nycTaxi/yellow_tripdata_2024-05.parquet'

topic = os.path.splitext(os.path.basename(path))[0]

# reading the parquet file data
data = pq.read_table(path)
df = data.to_pandas()

# debug info to verify the column names
print("Columns in the DataFrame:", df.columns.tolist())
print()

noOfRows = 300000
# sending each row as a dict to the serializer before writing to the topic
for index, row in df.head(noOfRows).iterrows():
    message = row.to_dict()
    producer.send(topic, value=message)
    if (index + 1) % 100000 == 0:
        print(f"Published {index + 1} messages to topic {topic}")

producer.flush()
producer.close()

print(f"\nFinished publishing data to topic {topic}")

# fetching current time now
end = time.time()
print(f"\n Took {end - start:.2f} seconds to publish {noOfRows} messages (rows) to {topic}")