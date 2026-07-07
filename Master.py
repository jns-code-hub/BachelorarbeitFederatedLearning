import numpy as np
from keras import models

class Master:
    
    def __init__(self, model:models.Sequential):
        self.model = model
        self.velocity = None

    def get_global_weights(self):
        return self.model.get_weights()
    
    def set_global_weights(self, weights):
        self.model.set_weights(weights)
    
    def aggregate(self, local_weights:list, client_data_counts:list, algo, beta:float):
        if algo == 0:
            self.FedAvg(local_weights, client_data_counts)
        if algo == 1:
            self.FedAvgM(local_weights, client_data_counts, beta)
    
    def FedAvg(self, local_weights:list, client_data_counts:list):
        """Hier werden alle Gewichte gemittelt (vgl. McMahan FedAvg).
        Dazu wird zuerst die Gesamtzahl der Datenpunkte ermittelt 
        und dann eine Gewichts Matrix erstellt, welche die gleiche Form 
        hat wie das Modell (Orientierung an Node 0).
        Im Anschluss werden die lokalen Gewichte durchgegangen und  
        """
        total_data_count = sum(client_data_counts)
        
        new_weights = [np.zeros_like(w) for w in local_weights[0]]

        for i in range(len(local_weights)):
            local_data_prop = client_data_counts[i] / total_data_count

            # Matrizenaddition der Gewichtslayer
            # 
            for layer_index in range(len(new_weights)):
                new_weights[layer_index] += local_weights[i][layer_index] * local_data_prop

        self.set_global_weights(new_weights)

        return new_weights  

    def FedAvgM(self, local_weights:list, client_data_counts:list, momentum_beta:float):
        total_data_count = sum(client_data_counts)
        w_old = self.get_global_weights()

        new_weights = [np.zeros_like(w) for w in local_weights[0]]

        # Durchschnitt der Gewichte berechnen
        for i in range(len(local_weights)):
            local_data_prop = client_data_counts[i] / total_data_count
            for layer_index in range(len(new_weights)):
                new_weights[layer_index] += local_weights[i][layer_index] * local_data_prop

        if self.velocity is None:
            self.velocity = [np.zeros_like(w) for w in w_old]
        
        final_weights = [np.zeros_like(w) for w in w_old]

        for i in range(len(w_old)):
            delta_w_layer = w_old[i] - new_weights[i]
            self.velocity[i] = momentum_beta * self.velocity[i] + delta_w_layer

            final_weights[i] = w_old[i] - self.velocity[i]

        self.set_global_weights(final_weights)

        return final_weights

    def evaluate(self, x_test, y_test):
        """Dürchführen der Server Evaluation mit den Testdaten"""
        print("Evaluation")
        history = self.model.evaluate(x_test, y_test)

        return history