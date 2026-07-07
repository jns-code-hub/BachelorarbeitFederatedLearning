import tensorflow as tf
import keras

class CustomModel(keras.Model):
    """Implementierung eines neuen Models, um die train_step-Methode
    verändern zu können.
    """
    def __init__(self, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.global_weights = None

    def train_step(self, data):
        """Überschreiben der Standard train_step Methode, um FedProx
        implementieren zu können. Wird auch bei FedAvg verwendet, 
        allerdings mu=0 -> FedAvg
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            base_loss = self.compute_loss(y=y, y_pred=y_pred)

            prox_loss = 0.0
            if self.global_weights is not None:
                for local_var, global_var in zip(self.trainable_variables, self.global_weights):
                    prox_loss += tf.reduce_sum(tf.square(local_var - global_var))
            
            total_loss = base_loss + (self.mu / 2.0) * prox_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        
        results["loss"] = total_loss 
        results["base_loss"] = base_loss # Optional: zum Vergleich in der Arbeit!
    
        return results
    
class Node:
    """Hier wird die Node (Client) implementiert. Dazu werden zunächst
    die grundlegenden Parameter festgelegt und eine Methode zum 
    trainieren implementiert.
    """
    def __init__(self, node_id, local_model:keras.models.Sequential, x_local, y_local, local_epochs=5, batch_size=32):
        self.node_id = node_id
        self.x_local = x_local
        self.y_local = y_local
        self.epochs = local_epochs
        self.batch_size = batch_size
        inputs = local_model.inputs
        outputs = local_model.outputs
        self.model = CustomModel(mu=0.0, inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=local_model.optimizer, 
            loss=local_model.loss, 
            metrics=['accuracy']
        )

    def local_train(self, global_weights, mu=0.0):
        """Hier wird das lokale Modell trainiert. Dazu ändert das lokale
        Modell die Gewichte auf die globalen Gewichte und trainiert 
        anschließend. Am Ende werden dann die neuen Gewichte, die Anzahl 
        der Trainingsdaten und die History zurückgegeben.
        """
        self.model.mu = mu
        self.model.global_weights = [tf.constant(w) for w in global_weights]
        
        self.model.set_weights(global_weights)
        
        history = self.model.fit(
            self.x_local, self.y_local, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=0
        )

        return self.model.get_weights(), len(self.x_local), history