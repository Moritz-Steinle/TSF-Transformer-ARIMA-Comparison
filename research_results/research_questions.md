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
### Sawtooth (0,1)x10
{
    "label": "Sawtooth",
    "order": [
        5,
        0,
        1
    ],
    "seasonal_order": [
        2,
        0,
        0,
        9
    ],
    "mean_squared_error": 0.0000006811,
    "runtime": [
        1.7918200493,
        47.1459891796
    ],
    "length_train_dataset": 70,
    "length_test_dataset": 20,
    "prediction": "{\"70\":0.7999995121,\"71\":0.8999990215,\"72\":0.1000003716,\"73\":0.2000004047,\"74\":0.300000146,\"75\":0.4000000195,\"76\":0.4999999194,\"77\":0.5999995785,\"78\":0.699999365,\"79\":0.7999993853,\"80\":0.8999992322,\"81\":0.1000002718,\"82\":0.2000001729,\"83\":0.3000000406,\"84\":0.3999997782,\"85\":0.4999996468,\"86\":0.5999995846,\"87\":0.6999993715,\"88\":0.7999986792,\"89\":0.8999980991}"
}
### Sawtooth (0,1)x1000
{
    "label": "Sawtooth",
    "order": [
        4,
        0,
        1
    ],
    "seasonal_order": [
        2,
        0,
        0,
        9
    ],
    "mean_squared_error": 0.0000002454,
    "runtime": [
        41.6610975266,
        1876.938
    ],
    "length_train_dataset": 8980,
    "length_test_dataset": 20,
    "prediction": "{\"8980\":0.799999912,\"8981\":0.8999998339,\"8982\":0.1000001508,\"8983\":0.2000000585,\"8984\":0.2999999821,\"8985\":0.3999999167,\"8986\":0.4999998446,\"8987\":0.599999784,\"8988\":0.6999997299,\"8989\":0.7999996744,\"8990\":0.8999996252,\"8991\":0.0999999612,\"8992\":0.1999999145,\"8993\":0.2999998706,\"8994\":0.3999998276,\"8995\":0.4999997843,\"8996\":0.5999997421,\"8997\":0.6999997001,\"8998\":0.7999995698,\"8999\":0.8999994498}"
}
