"""Hier ist der Code zur Erstellung der Plots aus der Arbeit, die keine 
Ergebnisgrafiken sind. Dieser Code ist vollständig losgelöst von der 
Simulationsumgebung. 
"""
import matplotlib.pyplot as plt
from Data_manager import DataManager
import numpy as np


plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)

def alphas_grundlagen(NUM_NODES=20, NUM_CLASSES=10, seed=42):
    """Diese Methode wurde genutzt um die Funktionsweise des Alpha 
    Parameters in Kapitel 2 darzustellen.
    """
    np.random.seed(seed)

    alphas = [10000, 100, 1, 0.0004]
    titles = [r"$\alpha \to \infty$", 
              r"$\alpha$ = 100", 
              r"$\alpha$ = 1",
              r"$\alpha \to 0$"]
    
    fig, axes = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
    axes = axes.flatten()

    client_ids = [f"{i+1}" for i in range(NUM_NODES)]
    colors = plt.get_cmap("tab10", NUM_CLASSES)

    for i, alpha in enumerate(alphas):
        proportions = np.random.dirichlet([alpha] * NUM_CLASSES, NUM_NODES)
        
        bottom = np.zeros(NUM_NODES)
        for k in range(NUM_CLASSES):
            axes[i].bar(client_ids, proportions[:, k], bottom=bottom, 
                        color=colors(k), label=f"Klasse {k}" if i == 0 else "")
            bottom += proportions[:, k]
        
        axes[i].set_title(titles[i], fontsize=12)
        if i >= 2:
            axes[i].set_xlabel("Clients")
        if i % 2 == 0:
            axes[i].set_ylabel("Anteil $P(y)$")

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=NUM_CLASSES, title="Klassen")
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig("dirichlet_grundlagen_theorie.pdf", bbox_inches='tight')
    plt.show()



def all_alphas_methodik(NUM_NODES=10, SEED=10):
    """Diese Methode wurde genutzt um die Abbildung der Non-IID-Daten 
    Simulation aus Kapitel 3 zu erzeugen
    """
    dm = DataManager()

    alphas = [(10000, 10000), (10000, 1), (1, 10000), (1, 1)]
    titles = [
        r"Label-$\alpha \to \infty$" "\n" r"Quantity-$\alpha \to \infty$",
        r"Label-$\alpha \to \infty$" "\n" r"Quantity-$\alpha = 1$",
        r"Label-$\alpha = 1$" "\n" r"Quantity-$\alpha \to \infty$",
        r"Label-$\alpha = 1$" "\n" r"Quantity-$\alpha = 1$"
    ]
    
    NUM_CLASSES = 10
    colors = plt.colormaps["tab10"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.flatten() 

    for idx, (l, q) in enumerate(alphas):
        partitions = dm.create_partitions(num_nodes=NUM_NODES, label_alpha=l, quantity_alpha=q, seed=SEED)
        num_nodes = len(partitions)

        stats = np.zeros((num_nodes, NUM_CLASSES))
        
        for node_idx, (x, y) in enumerate(partitions):
            labels = np.argmax(y, axis=1)
            for label in labels:
                stats[node_idx, label] += 1
                
        bottom = np.zeros(num_nodes)
        for c in range(NUM_CLASSES):
            axes[idx].bar(range(num_nodes), stats[:, c], bottom=bottom, color=colors(c))
            bottom += stats[:, c]
            
        axes[idx].set_title(titles[idx], pad=10)
        
        axes[idx].set_xticks(range(num_nodes))
        axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
        
        if idx >= 2:
            axes[idx].set_xlabel("Node ID")
        if idx % 2 == 0:
            axes[idx].set_ylabel("Anzahl Bilder")

    handles = [plt.Rectangle((0,0),1,1, color=colors(c)) for c in range(NUM_CLASSES)]
    labels_legend = [f"Klasse {c}" for c in range(NUM_CLASSES)]
    fig.legend(handles, labels_legend, loc='lower center', bbox_to_anchor=(0.5, 0.0005), 
               ncol=5, title="Klassen", frameon=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig("dirichlet_methodik.pdf", bbox_inches='tight')
    plt.show()


def data_examples():
    """Diese Methode generiert eine Abbildung, die exemplarisch 
    Datenpunkte aus dem Fashion-MNIST-Datensatz darstellt.
    """
    dm = DataManager()

    x_test, y_test = dm.get_test_data()

    img1 = x_test[0]
    img_1_label = f"Klasse: 9 / Ankle Boot"
    img2 = x_test[1000]
    img_2_label = f"Klasse: 0 / T-Shirt/Top"
    img3 = x_test[2000]
    img_3_label = f"Klasse: 8 / Bag"

    fig,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ax1.imshow(img1)
    ax1.set_title(img_1_label)
    ax2.imshow(img2)
    ax2.set_title(img_2_label)
    ax3.imshow(img3)
    ax3.set_title(img_3_label)
    plt.tight_layout()
    plt.savefig("data_examples.pdf", bbox_inches="tight")
    plt.show()

#all_alphas_methodik()
#alphas_grundlagen()
data_examples()