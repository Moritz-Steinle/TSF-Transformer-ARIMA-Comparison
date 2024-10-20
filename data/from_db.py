import os

import influxdb_client
import pandas as pd

from config import config

org = "uulm"


def fetch(
    resolution: str,
    start_date="2024-06-18T06:00:00Z",
    end_date="2024-09-24T00:00:00Z",
):
    client = influxdb_client.InfluxDBClient(
        url=config.influxdb_url, token=config.influxdb_token, org=org
    )
    query_api = client.query_api()
    query = f"""from(bucket: "plant_sensor_data")
        |> range(start: {start_date}, stop: {end_date})
        |> filter(fn: (r) => r["_measurement"] == "ESP32_DEVKIT_V1_S001")
        |> filter(fn: (r) => r["_field"] == "moisture")
        |> aggregateWindow(every: {resolution}, fn: last)
        |> keep(columns: ["_time", "_value", "_field"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", "moisture"])
        |> yield(name: "mean")"""

    dataframes = query_api.query_data_frame(org=org, query=query)
    write_dataframe_to_csv(dataframes, resolution=resolution)
    print(f"Fetched {len(dataframes)} rows in {resolution} intervals")
    return dataframes


def write_dataframe_to_csv(dataframes, resolution: str):
    filename = f"influxdb-{resolution}.csv"
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename_with_path = os.path.join(dirname, filename)
    dataframes.to_csv(filename_with_path, index=False)


def read_file(resolution: str):
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename_with_path = os.path.join(dirname, f"influxdb-{resolution}.csv")
    dtype = {"result": str, "table": str, "_time": str, "moisture": "float64"}
    return pd.read_csv(
        filename_with_path,
        sep=",",
        encoding="utf-8",
        dtype=dtype,
        parse_dates=["_time"],
    )
