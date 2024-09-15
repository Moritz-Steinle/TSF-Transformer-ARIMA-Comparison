# Research questions
- Graphs of data
- Graphs of predictions
- Effect of data normalisation
- Effect of data modification by copy pasting seasons with little datapoints
- Runtime of pdq calculation (`arima.data.find_best_order`)
- Runtime of hyperparameter calculation (`transformer.data.run_hyperparameter_study`)
- Training runtime
- Prediction accuracy

# Experiment Results
## ARIMA
### Sawtooth 1000 intervals in (0,1)
Best model:  ARIMA(4,0,1)(2,0,0)[9]          
Total fit time: 1876.938 seconds

