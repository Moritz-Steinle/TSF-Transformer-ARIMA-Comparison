# Time Series Forecasting comparsion of AI to classic statistical approaches 

This project compares the relatively new approach of TSF with a transformer model to the prominent ARIMA forecasting.

# Main project parts
- main: Project entry point
- config.ini: Configurate all app parameters. Most important is `max_epochs` to determine training length
- transformer/controller: Main access to AI functionality
- data/fetch_from_db: To fetch specific data from the InfluxDB

# How to run the project
1. Clone the repo
2. Create venv: `python -m venv <your-venv-name>`
3. Activate venv: `. <your-venv-name>/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the code: `python main.py`
6. On finish, deactivate `deactivate`