import keras
from Data_manager import DataManager
import Model
import numpy as np

class WeightCallback(keras.callbacks.Callback):
    """Custom Callback, um nach jeder Epoche die Gewichte speichern zu
    können, um die besten Gewichte für die Gewichtsdivergenz ermitteln
    zu können
    """
    def __init__(self):
        super().__init__()
        self.epoch_weights = []

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoche {epoch} ended")
        self.epoch_weights.append([np.copy(w) for w in self.model.get_weights()])

def sgd_train(epochs, batch_size):
    """Training der SGD-Baseline. Dabei wird die selbe Architektur wie 
    für die Clients verwendet.
    """
    dm_sgd = DataManager()

    x_train = dm_sgd.train_data
    y_train = dm_sgd.train_labels

    x_test = dm_sgd.test_data
    y_test = dm_sgd.test_labels

    sgd_model = Model.create_cnn_model()

    weight_history = WeightCallback()
    history = sgd_model.fit(
            x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.1, 
            verbose=0,
            callbacks=[weight_history]
        )
    
    
    return history, weight_history.epoch_weights
