from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical

class DataManager:

    def __init__(self):
        # Laden der Trainingsdaten / Testdaten
        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = fashion_mnist.load_data()

        # Dimensionen ändern, damit es in Conv2D passt.
        # -1 sagt Numpy, dass die neue shape mit der alten kompatibel sein soll
        self.train_data = self.train_data.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        self.test_data = self.test_data.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        # OneHotEncoding der Labels
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)


    def get_test_data(self):
        """Zurückgeben der Testdaten"""
        return self.test_data, self.test_labels


    def create_partitions(self, num_nodes, label_alpha, quantity_alpha, seed=42):
        """Partitionierung der Daten anhand von Label Alpha und 
        Quantity Alpha.
        """

        #Metadaten festlegen, Seed, Jede Klasse hat 6000 Bilder, 
        # Gesamtmenge der Daten berechnen (Fashion Mnist = 60.000)
        np.random.seed(seed)
        DATA_PER_CLASS = 6000
        total_data = len(self.train_data)

        # Labels sind one hot encoded, müssen aber als Zahl vorliegen
        y_train_simple = np.argmax(self.train_labels, axis=1)

        print(f"Total Data: {total_data}")
        
        # 1. Quantity Proportions der Clients berechnen
        quantity_proportions = np.random.dirichlet([quantity_alpha] * num_nodes)
        quantity_proportions = (quantity_proportions * total_data).astype(int)
        print(f"Sum of Quantity Proportions before filling: {np.sum(quantity_proportions)}")
        
        # 1b. Quantity Proportions auffüllen, falls nicht alle 60.000
        # zugewiesen wurden Differenz wird der Node mit der höchsten 
        # Datenzahl zugewiesen
        diff = total_data - np.sum(quantity_proportions)
        quantity_proportions[np.argmax(quantity_proportions)] += diff
        print(f"Quantity Props: {quantity_proportions}")
        print(f"Sum of Quantity Props: {sum(quantity_proportions)}")

        # 2. Label Proportions berechnen
        # Zunächst die "ideale Matrix" erstellen, ungeachtet der 
        # tatsächlichen Datenmengen pro Label
        ideal_matrix = np.zeros((num_nodes, 10))
        for i in range(num_nodes):
            label_proportions = np.random.dirichlet([label_alpha] * 10)
            # Prozentwert aller Labels der Node i * Anzahl der Gesamtdatenmenge der Node i
            ideal_matrix[i] = label_proportions * quantity_proportions[i] 

        # 2b. Forderung der Clients runterskalieren auf die tatsächliche 
        # Datenmenge pro Klasse
        for k in range(len(np.unique(y_train_simple))):
            # Gesamtforderung der Clients für Klasse k ermitteln
            cumulated_for_k = ideal_matrix[:,k].sum()
            # Falls mehr Datenpunkte gefordert werden, als in k überhaupt existieren...
            if cumulated_for_k > DATA_PER_CLASS:
                # Skalierungsfaktor ermitteln
                factor = DATA_PER_CLASS / cumulated_for_k
                # Skalierungsfaktor anwenden -> Forderungen prozentual reduzieren
                ideal_matrix[:,k] *= factor

        # 3. Finale Matrix erstellen
        # Vorherige Matrix mit int werten speichern -> Kommastellen werden abgeschnitten -> Es werden nicht alle 60.000 Bilder verteilt
        final_matrix = ideal_matrix.astype(int)

        print(f"Sum of final Matrix before filling: {sum(final_matrix.sum(axis=0))}")

        # Anzahl Daten pro Klasse ermitteln (Array der Spaltensummen)
        current_class_counts = final_matrix.sum(axis=0)
        # Array mit Differenz, wie viel Daten pro Klasse "ungenutzt" sind
        class_slots_left = DATA_PER_CLASS - current_class_counts
        # Anzahl Daten pro Node (Array der Zeilensummen)
        current_node_counts = final_matrix.sum(axis=1)
        # Array mit Differenz, wie viele Daten einer Node fehlen
        node_slots_left = quantity_proportions - current_node_counts

        #3b. Auffüllen der Nodes mit übrigen Daten
        for n in range(num_nodes):
            if node_slots_left[n] > 0:
                # Filtern nach Klassen, die noch ungenutzte Daten haben
                available_classes = np.where(class_slots_left > 0)[0]

                if len(available_classes) > 0:
                    for c in available_classes:
                        # Aktueller Node Anteil geben
                        # Entweder so viel wie sie insgesamt noch 
                        # braucht oder so viele, wie von der 
                        # aktuellen Klasse noch übrig sind
                        take = min(node_slots_left[n], class_slots_left[c])
                        final_matrix[n,c] += take
                        class_slots_left[c] -= take
                        node_slots_left[n] -= take
        
        # 3c. Durch rundung kann ein kleiner Rest übrig bleiben
        # diesen Rest der ersten Node geben, die noch Platz hat
        remaining_total_diff = total_data - final_matrix.sum()
        if remaining_total_diff > 0:
            for n in range(num_nodes):
                for c in range(10):
                    if remaining_total_diff > 0 and class_slots_left[c] > 0:
                        final_matrix[n, c] += 1
                        class_slots_left[c] -= 1
                        remaining_total_diff -= 1

        print(f"Sum of final Matrix after filling: {sum(final_matrix.sum(axis=0))}")
        print(final_matrix)

        # 4. Slicen der Daten
        class_indices = {}
        #4b. Shufflen der Daten einer Klasse
        for k in range(len(np.unique(y_train_simple))):
            k_indexes = np.where(y_train_simple == k)[0]
            np.random.shuffle(k_indexes)
            class_indices[k] = k_indexes.tolist()
        
        node_indices = [[] for _ in range(num_nodes)]

        for n in range(num_nodes):
            for c in range(len(np.unique(y_train_simple))):
                count = final_matrix[n,c]
                if count > 0:
                    take = class_indices[c][:count]
                    node_indices[n].extend(take)
                    class_indices[c] = class_indices[c][count:]
        
        node_datasets = []
        for n in range(num_nodes):
            x_data = self.train_data[node_indices[n]]
            y_data = self.train_labels[node_indices[n]]
            node_datasets.append((x_data, y_data))

        return node_datasets