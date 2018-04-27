import argparse
import datetime
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pyspark.sql import SparkSession

def filtering(data, A, B):
  base = []
  for a, b in zip(A, B):
    mask = (data['timestamp'] > a) & (data['timestamp'] < b)
    base.append(data.loc[mask])
  base = pd.concat(base)
  return base

if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Split the accelerometer and GPS  parquet data based on \
    the answering label'
  )

  parser.add_argument('--path', help='directory of the parquet file')

  parser.add_argument('--label', help='label of the transportation mode:' +
    '{Car, Auto, Bike}')

  parser.add_argument('--window', type=int, default=15,
  help='the window size in minutes: {5, 10, 20}')

  args = parser.parse_args()

  spark = SparkSession.builder \
    .master('local') \
    .appName('split') \
    .config('spark.executor.memory', '4gb') \
    .config("spark.cores.max", "4") \
    .getOrCreate()

  data = pq.ParquetDataset(str(args.path) + "questionnaireanswers.parquet/")
  data = data.read().to_pandas()
  data = data.fillna(0)

  aux = data.loc[data["answerstringb"] == str(args.label), "questiontimestamp"]

  A = aux - datetime.timedelta(minutes=args.window)
  B = aux + datetime.timedelta(minutes=args.window)

  data = pq.ParquetDataset(str(args.path) + "accelerometerevent.parquet/")
  data = data.read().to_pandas()
  data = data.drop(['day'], axis=1)

  base = filtering(data, A, B)
  pd.DataFrame.to_csv(base, str(args.label) + "_acc.csv")

  data = spark.read.load(str(args.path) + "locationeventpertime.parquet/")
  data = data.toPandas()

  data = data[["timestamp", "point"]]

  base = filtering(data, A, B)
  pd.DataFrame.to_csv(base, str(args.label) + "_gps.csv")
