# **Comparison of ARIMA and Transformer in Time Series Forecasting**

This project evaluates the time series forecasting capabilities of deep learning-based Transformers against statistical ARIMA models using two types of datasets: empirical and synthetic.

### **Models Used**
- **Transformer:** [PyTorch-Forecasting Temporal Fusion Transformer](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html)
- **ARIMA:** [Statsmodels SARIMAX](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

---

### **Datasets**
1. **Empirical Data:**  
   - Collected through an experimental setup monitoring soil dryness levels in a potted plant.  
   - Contains ~840,000 data points collected over 97 days in 10-second steps.  

2. **Synthetic Data:**  
   - Artificially generated to approximate the empirical data using a sawtooth function.

---

## **Dependencies**
- **Python 3.10:** Required due to compatibility constraints.  
  - Python < 3.10: Incompatible due to missing language features.  
  - Python > 3.10: Incompatible with `pytorch-forecasting`.

---

## **Installation**
1. Clone the repository:  
   ```bash
   git clone <repository-url>
   ```
2. Create a virtual environment:  
   ```bash
   python -m venv <your-venv-name>
   ```
3. Activate the virtual environment:  
   - Linux/macOS:  
     ```bash
     source <your-venv-name>/bin/activate
     ```  
   - Windows:  
     ```cmd
     <your-venv-name>\Scripts\activate
     ```  
4. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
5. Run the project:  
   ```bash
   python main.py
   ```
6. Deactivate the virtual environment when done:  
   ```bash
   deactivate
   ```

---

## **Main File Usage**

The primary script contains functions for training and evaluating models. Results, including plots, are saved in `{model}/results`.

### **Available Functions**
1. **`transformer_empirical()`**: Train and evaluate the Transformer on empirical data.  
2. **`transformer_synthetic()`**: Train and evaluate the Transformer on synthetic data.  
3. **`arima_empirical()`**: Train and evaluate the ARIMA model on empirical data.  
4. **`arima_synthetic()`**: Train and evaluate the ARIMA model on synthetic data.  
5. **`plot_series()`**: Visualize a given data series.  
6. **`analyse_series()`**: Analyze key characteristics of a data series.

---

## **Modules Overview**

### **Data**
- Fetch datasets from an `InfluxDB` and save them as CSV files.  
- Load and preprocess datasets (e.g., normalization, interpolation).  
- Analyze dataset properties (e.g., stationarity, season length).  
- Empirical data is stored in `/data-files`.

### **ARIMA**
- Train and evaluate the ARIMA model, saving results to `./results`.  
- Determine optimal ARIMA parameters using [`auto_arima`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html).

### **Transformer**
- Train and evaluate the Transformer model, saving results to `./results`.  
- Optimize model parameters using [`Optuna`](https://optuna.org/).  
- Save and load trained models.

### **Log**
- Log the following:
  - Model parameters.  
  - Dataset characteristics.  
  - Predicted values and error metrics (e.g., MAE, RMSE, MASE, SMAPE).  
  - Prediction plots.  
  - Computation times.

---

## **Notes**
- Configure paths in `config.ini`.  
- Set InfluxDB credentials by renaming `credentials_EXAMPLE.ini` to `credentials.ini` and updating the file with your credentials.
