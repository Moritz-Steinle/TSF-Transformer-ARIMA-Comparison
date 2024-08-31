import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer


def make_prediction(model: TemporalFusionTransformer, val_dataloader):
    predictions = model.predict(
        val_dataloader,
        return_x=True,
        mode="raw",
    )
    network_input = predictions.x
    network_output = predictions.output
    model.plot_prediction(network_input, network_output, idx=0, add_loss_to_title=True)
    print(predictions)
    plt.show()
    return predictions
