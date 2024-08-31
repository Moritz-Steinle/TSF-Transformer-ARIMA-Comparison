import os

import influxdb_client
import pandas as pd

from config import config
from data.process import dtype

org = "uulm"


def fetch(sample_interval: str = config.row_sample_interval):
    client = influxdb_client.InfluxDBClient(
        url=config.influxdb_url, token=config.influxdb_token, org=org
    )
    query_api = client.query_api()
    query = f"""from(bucket: "plant_sensor_data")
        |> range(start: {config.range_start}, stop: {config.range_stop})
        |> filter(fn: (r) => r["_measurement"] == "ESP32_DEVKIT_V1_S001")
        |> filter(fn: (r) => r["_field"] == "moisture")
        |> aggregateWindow(every: {sample_interval}, fn: last)
        |> keep(columns: ["_time", "_value", "_field"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", "moisture"])
        |> yield(name: "mean")"""

    dataframes = query_api.query_data_frame(org=org, query=query)
    write_dataframe_to_csv(dataframes, filename=f"{sample_interval}.csv")
    print(f"Fetched {len(dataframes)} rows in {sample_interval} intervals")
    return dataframes


def write_dataframe_to_csv(dataframes, filename):
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename_with_path = os.path.join(dirname, filename)
    dataframes.to_csv(filename_with_path, index=False)


def read_file(resolution: str):
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename_with_path = os.path.join(dirname, f"{resolution}.csv")
    return pd.read_csv(
        filename_with_path,
        sep=",",
        encoding="utf-8",
        dtype=dtype,
        parse_dates=["_time"],
    )
