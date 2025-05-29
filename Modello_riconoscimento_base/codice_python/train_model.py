import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Parametri ---
PREPROCESSED_DATA_FILE = "preprocessed_dataset.npz"
# --- Fine Parametri ---

def load_data(file_path):
    """Carica i dati pre-elaborati."""
    print(f"Caricamento dati da: {file_path}")
    with np.load(file_path) as data:
        train_images = data['train_images']
        train_labels = data['train_labels']
        val_images = data['val_images']
        val_labels = data['val_labels']
        class_names = data['class_names']
        # Le dimensioni sono salvate come array, prendiamo il primo elemento
        img_width = int(data['img_width'][0])
        img_height = int(data['img_height'][0])

    print("Dati caricati con successo.")
    print(f"  Forma immagini di addestramento: {train_images.shape}")
    print(f"  Forma etichette di addestramento: {train_labels.shape}")
    print(f"  Forma immagini di validazione: {val_images.shape}")
    print(f"  Forma etichette di validazione: {val_labels.shape}")
    print(f"  Nomi delle classi: {class_names}")
    print(f"  Dimensioni immagini (H, W): ({img_height}, {img_width})")
    return train_images, train_labels, val_images, val_labels, class_names, img_height, img_width

def build_model(input_shape, num_classes):
    """Definisce un semplice modello CNN."""
    print(f"\nCostruzione del modello con input_shape: {input_shape} e num_classes: {num_classes}")

    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape), # 16 filtri, kernel 3x3
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'), # 32 filtri, kernel 3x3
        layers.MaxPooling2D((2, 2)),

        # Potresti aggiungere un altro blocco Conv/Pool se le dimensioni dell'immagine sono più grandi (es. 96x96)
        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(32, activation='relu'), # Un layer Dense intermedio
        layers.Dense(num_classes, activation='softmax') # Output layer: softmax per classificazione multi-classe
                                                      # Se fosse binaria (2 classi), potresti usare 1 neurone e 'sigmoid'
    ])

    # Compila il modello
    # Per la classificazione multi-classe con etichette intere, usa 'sparse_categorical_crossentropy'
    # Se le etichette fossero one-hot encoded, useresti 'categorical_crossentropy'
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Stampa un riassunto dell'architettura del modello
    return model

def plot_training_history(history):
    """Visualizza l'andamento di accuracy e loss durante l'addestramento."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

def main():
    # Carica i dati pre-elaborati
    train_images, train_labels, val_images, val_labels, class_names, img_height, img_width = load_data(PREPROCESSED_DATA_FILE)

    input_shape = (img_height, img_width, 1) # Altezza, Larghezza, Canali (1 per scala di grigi)
    num_classes = len(class_names)

    # Costruisci il modello
    model = build_model(input_shape, num_classes)

    # Addestra il modello
    print("\nInizio addestramento del modello...")
    # NUM_EPOCHS determina per quante volte il modello vedrà l'intero dataset di addestramento.
    # Inizia con un numero moderato (es. 10-20) e poi puoi aumentarlo se necessario.
    # BATCH_SIZE è il numero di campioni elaborati prima di aggiornare i pesi del modello.
    NUM_EPOCHS = 15
    BATCH_SIZE = 32

    history = model.fit(train_images, train_labels,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_images, val_labels))

    print("Addestramento completato.")

    # Salva il modello Keras addestrato (formato .keras)
    model.save("hand_gesture_model.keras")
    print(f"\nModello Keras addestrato salvato come: hand_gesture_model.keras")

    # Visualizza la cronologia dell'addestramento
    plot_training_history(history)

if __name__ == '__main__':
    # Assicurati che matplotlib sia installato se non l'hai già fatto:
    # pip install matplotlib (nel terminale di PyCharm con .venv_arch_tf attivo)
    main()