# Bachelorarbeit - Digitaler Anhang
Der Digitale Anhang zur Bachelorarbeit *Empirische Untersuchung des Einflusses von Non-IID Daten auf das Konvergenzverhalten im Federated Learning* umfasst die gesamte Implementierung der Simulationsumgebung, alle Ergebnisdaten in Rohform, alle generierten Plots und eine Anleitung zum Durchführen der Experimente.

Folgend wird dargelegt, wie der digitale Anhang aufgebaut ist.

## Implementierung
Die Implementierung umfasst drei Teilaspekte, Simulationsumgebung, Ergebnisgrafiken, sonstige Grafiken, die im Folgenden beschrieben werden.

### Simulationsumgebung
Die Implementierung der Simulationsumgebung erstreckt sich über mehrere Dateien. So wird die Verarbeitung des Datensatzes in `Data_manager.py` vorgenommen, während Server und Client in `Master.py` beziehungsweise `Node.py` implementiert wurden. Das Modell und die SGD-Baseline wurden in `Modell.py` und `SGD_Baseline.py` implementiert. 

Die zentrale Ablaufsteuerung befindet sich in `Simulation.ipynb`. 

### Ergebnisgrafiken
Die Implementierung aller Grafiken, die im Ergebnisteil der Arbeit zu finden sind, befindet sich in `plots.ipynb`. Alle in der Arbeit verwendeten Plots befinden sich im Ordner `results/plots`. Zur besseren Übersichtlichkeit wurden die Grafiken manuell umbenannt und dorthin verschoben.

### Sonstige Grafiken
Grafiken, die nicht Teil der Ergebnisse sind, wurden in `helper_plots.py` generiert. Dazu gehören *Abbildung 2: Grafische Darstellung verschiedener $\alpha$ in Anlehnung an Hsu et al. (2019)*, *Abbildung 3: Exemplarische Darstellung der Bilder des Fashion-MNIST-Datensatzes* und *Abbildung 4: Vergleich von Quantity-$\alpha$ und Label-$\alpha$*.

## Ergebnisse
Die Rohdaten der Versuche befinden sich unter `results/`. Der Ordner enthält für jeden *Seed* einen Unterordner, welcher wiederum einen Ordner pro Szenario enthält. Die Namensgebung folgt dabei folgendem Muster: `<Aggregation>(FedAvg=0, FedAvgM=1)_Label_<Label-α>_Quantity_<Quantity-α>_MU<μ-Wert>_Beta_<β-Wert>`. 

## Installation
Da die Implementierung vollständig auf einem Linux System stattfand, beziehen sich die folgenden Schritte ebenfalls auf Linux Systeme. Auf Windows Systemen kann die Installation abweichen.

### Voraussetzungen
- Python 3.10.x mit pip
- Entwicklungsumgebung, es wird Visual Studio Code empfohlen
- Klonen des GitHub Repositories `git clone https://github.com/jns-code-hub/BachelorArbeit-FederatedLearning.git`

### Erstellung der Python Umgebung
Zur Erstellung der Python Umgebung müssen folgende Schritte im Basisverzeichnis durchgeführt werden:
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

### Optional: Cuda installieren
Um das Training auf der GPU durchzuführen wird *Cuda* benötigt. Die Anleitung zur Installation befindet sich unter *https://www.tensorflow.org/install/pip*. Tensorflow mit Cuda ist allerdings nur für Linux oder WSL verfügbar.

Dieser Schritt kann ausgelassen werden, das Training wird dann auf der CPU durchgeführt, was allerdings deutlich langsamer sein kann.

### Konfigurieren und ausführen
Wurde alles installiert kann `Simulation.ipynb` geöffnet werden. Die konfigurierbaren Parameter befinden sich direkt zu Beginn der Datei und können beliebig angepasst werden. Die in den Versuchen verwendeten Belegungen sind der Arbeit zu entnehmen.

Nach der Konfiguration kann das Notebook gestartet werden. Bei Visual Studio Code funktioniert dies mit dem Button *Run All*.

Hinweis: Als *ipynb-Kernel* muss die Python Installation der virtuellen Umgebung (venv) ausgewählt sein. In Visual Studio Code geht das oben rechts, wenn das Notebook geöffnet ist.


