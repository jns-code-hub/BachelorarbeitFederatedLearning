# Architekturen der Modelle
from keras import layers, models,optimizers

def create_cnn_model():
    """Hier wird die Architektur des Modells festgelegt. Diese ist für
    Server, Client und SGD-Baseline identisch.
    """

    model = models.Sequential()
    opt = optimizers.SGD(learning_rate=0.01)

    model.add(layers.Conv2D(32,(5,5), padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(5,5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.summary()
    #model.save('model.keras')
    return model

create_cnn_model()