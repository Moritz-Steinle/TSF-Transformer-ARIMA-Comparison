import os

import influxdb_client
from pandas import DataFrame

from config import config


def fetch_from_influxdb(
    target_filename: str,
    start_date="2024-06-18T06:00:00Z",
    end_date="2024-09-24T00:00:00Z",
) -> DataFrame:
    """
    Fetches data from an InfluxDB database and writes it to a CSV file.
    Args:
        target_filename (str): The name of the CSV file to write the data to.
        start_date (str, optional): The start date for the data query in ISO 8601 format.
            Defaults to "2024-06-18T06:00:00Z".
        end_date (str, optional): The end date for the data query in ISO 8601 format.
            Defaults to "2024-09-24T00:00:00Z".
    Returns:
        DataFrame: A pandas DataFrame containing the fetched data.
    Raises:
        Exception: If there is an error in querying the InfluxDB or writing the data to the CSV file.
    """
    client = influxdb_client.InfluxDBClient(
        url=config.influxdb_url, token=config.influxdb_token, org=config.influxdb_org
    )
    query_api = client.query_api()
    query = f"""from(bucket: "plant_sensor_data")
        |> range(start: {start_date}, stop: {end_date})
        |> filter(fn: (r) => r["_measurement"] == "ESP32_DEVKIT_V1_S001")
        |> filter(fn: (r) => r["_field"] == "moisture")
        |> aggregateWindow(every: {target_filename}, fn: last)
        |> keep(columns: ["_time", "_value", "_field"])
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", "moisture"])
        |> yield(name: "mean")"""

    dataset = query_api.query_data_frame(org=config.influxdb_org, query=query)
    write_dataframe_to_csv(dataset, target_filename=target_filename)
    print(f"Fetched {len(dataset)} rows to {target_filename}")
    return dataset


def write_dataframe_to_csv(dataset: DataFrame, target_filename: str) -> None:
    """
    Writes a pandas DataFrame to a CSV file to the data-files directory.
    Parameters:
        dataset (DataFrame): The DataFrame to be written to a CSV file.
        target_filename (str): The name of the target CSV file (without the .csv extension).
    Returns:
        None
    """
    target_filename = f"{target_filename}.csv"
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename_with_path = os.path.join(dirname, config.data_file_path, target_filename)
    dataset.to_csv(filename_with_path, index=False)
