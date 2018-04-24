import sys, os
import argparse
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Split the accelerometer and GPS  parquet data based on \
    the answering label'
  )

  parser.add_argument('--path', help='directory of the parquet file')

  parser.add_argument('--label', help='label of the transportation mode:' + 
    '{Car, Auto, Bike}')

  path = os.path.dirname(os.path.realpath(os.path.basename(__file__)))
  args = parser.parse_args()

  data = pq.ParquetDataset(str(args.path) + "/questionnaireanswers.parquet/")
  data = data.read().to_pandas()
  data = data.fillna(0)

  aux = data.loc[data["answerstringb"] == str(args.label), "questiontimestamp"]

  aux = aux.index.values

  A = data.iloc[aux-1, 1]
  B = data.iloc[aux, 1]

  data = pq.ParquetDataset(str(args.path) + "accelerometerevent.parquet/")
  data = data.read().to_pandas()
  data = data.fillna(0)

  base = []
  for a, b in zip(A, B):
    mask = (data['timestamp'] > a) & (data['timestamp'] <= b)
    base.append(data.loc[mask])

  base = pd.concat(base)
  base = base.drop(['day'], axis=1)

  pd.DataFrame.to_csv(base, str(args.label) + "_acc_bruto.csv")

  data = pq.ParquetDataset(str(args.path) + "locationeventpertime.parquet/")
  data = data.read().to_pandas()
  data = data.fillna(0)

  base = []
  for a, b in zip(A, B):
    mask = (data['timestamp'] > a) & (data['timestamp'] <= b)
    base.append(data.loc[mask])

  base = pd.concat(base)
  base = base.drop(['day'], axis=1)

  pd.DataFrame.to_csv(base, str(args.label) + "_gps_bruto.csv")
