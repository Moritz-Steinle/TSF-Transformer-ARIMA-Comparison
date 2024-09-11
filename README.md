# Time Series Forecasting comparsion of AI to classic statistical approaches 

This project compares the relatively new approach of TSF with a transformer model to the prominent ARIMA forecasting.
The real-world data used is the ground dryness of a potted plant that is watered roughly every 6 days.
This results in a one-dimensional, seasonal dataset. 

# Dependencies
- Python <= 3.9: Incompatible due to language features from 3.10
- Python >= 3.11: Not compatible due to pytorch-forecasting not compatible with > 3.10.

# Installation
1. Clone the repository
2. Create venv: `python -m venv <your-venv-name>`
3. Activate venv: `. <your-venv-name>/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the code: `python main.py`
6. On finish, deactivate `deactivate`

# Main File Usage

This README section provides an overview of the main file in our Python project, which contains several functions for training and evaluating time series models.

## Available Functions

### Transformer Models

1. `influx_transformer()`: Trains and evaluates a transformer model on our real-life scenario data.
2. `sawtooth_transformer()`: Trains and evaluates a transformer model on a mock dataset. The contained sawtooth function resembles the real-life data
3. `tutorial_transformer()`: Trains and evaluates a transformer model based on the [tutorial](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html) for the transformer.
4. `evaluate_saved_transformer(dataset, model_path)`: Evaluates a saved transformer model on a given dataset.

### ARIMA Models

1. `influx_arima()`: Trains and evaluates an ARIMA model on the real-life scenario data.
2. `sawtooth_arima()`: Trains and evaluates an ARIMA model on the mock sawtooth data
3. `run_arima_comparison()`: Currently only available with database access. Runs ARIMA comparisons on the real-life scenario data with different resolutions.


## Data
`Data` provides csv files with different resolutions. Use `fetch_data_from_db()` to fetch different data.
InfluxDB credentials are required. 
### InfluxDB credentials
The credentials can be set by renaming `credentials_EXAMPLE.ini` to `credentials.ini` and adding the credentials to the file.

## Notes

- The transformer functions use custom dataloader parameters and hyperparameters specific to the dataset. 
    It is recommended to set the flag `should_run_hyperparameter_study=True` when using different datasets.
- The same is true for the ARIMA functions with the flag `should_find_best_order=True` 

For more detailed information about each function's parameters and behavior, please refer to the function docstrings or the project's full documentation.
