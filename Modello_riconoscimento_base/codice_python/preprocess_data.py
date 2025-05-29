import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Opzionale, per visualizzare qualche immagine

# --- Parametri di Pre-elaborazione ---
DATASET_PATH = "dataset"  # Cartella principale contenente le sottocartelle delle classi
IMG_WIDTH = 96           # Larghezza desiderata per le immagini ridimensionate
IMG_HEIGHT = 96          # Altezza desiderata per le immagini ridimensionate
# Per TinyML, potresti provare anche più piccolo, es. 48x48 o 64x64,
# ma 96x96 è un buon compromesso iniziale.

# Nomi delle classi (devono corrispondere esattamente ai nomi delle tue cartelle nel dataset)
# Aggiungi o modifica in base alle classi che hai raccolto
CLASS_NAMES = ["mano_alzata", "mano_abbassata"]
# --- Fine Parametri ---

def load_and_preprocess_images(dataset_path, class_names, img_width, img_height):
    """
    Carica le immagini dal dataset, le ridimensiona, le converte in scala di grigi,
    le normalizza e crea le etichette.
    """
    images = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"ATTENZIONE: La cartella per la classe '{class_name}' non è stata trovata in '{dataset_path}'. Salto.")
            continue

        print(f"Caricamento immagini per la classe: '{class_name}' (etichetta: {class_index})")
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                # Leggi l'immagine a colori (verrà convertita dopo)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  ATTENZIONE: Impossibile leggere l'immagine {img_path}. Salto.")
                    continue

                # 1. Ridimensionamento
                img_resized = cv2.resize(img, (img_width, img_height))

                # 2. Conversione in Scala di Grigi
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

                images.append(img_gray)
                labels.append(class_index)
            except Exception as e:
                print(f"  ERRORE durante l'elaborazione di {img_path}: {e}")

    if not images: # Se nessuna immagine è stata caricata
        print("ERRORE: Nessuna immagine caricata. Controlla DATASET_PATH e i nomi delle classi.")
        return None, None

    # Converti le liste in array NumPy
    images_np = np.array(images)
    labels_np = np.array(labels)

    # 3. Normalizzazione dei Pixel (scala 0-1)
    # Espandi le dimensioni per il canale (necessario per TensorFlow/Keras con CNN)
    # Da (num_samples, height, width) a (num_samples, height, width, 1)
    images_np = np.expand_dims(images_np, axis=-1).astype('float32') / 255.0

    print(f"\nCaricamento completato.")
    print(f"Numero totale di immagini caricate: {len(images_np)}")
    print(f"Forma dell'array delle immagini: {images_np.shape}") # Dovrebbe essere (num_immagini, IMG_HEIGHT, IMG_WIDTH, 1)
    print(f"Forma dell'array delle etichette: {labels_np.shape}")   # Dovrebbe essere (num_immagini,)

    return images_np, labels_np

def main():
    print("Avvio pre-elaborazione dati...")

    images, labels = load_and_preprocess_images(DATASET_PATH, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT)

    if images is None or labels is None:
        return # Termina se il caricamento fallisce

    # Opzionale: visualizza alcune immagini pre-elaborate per verifica
    num_to_show = 5
    if len(images) >= num_to_show:
        print(f"\nVisualizzazione di {num_to_show} immagini pre-elaborate (in scala di grigi e normalizzate):")
        plt.figure(figsize=(10, 5))
        for i in range(num_to_show):
            plt.subplot(1, num_to_show, i + 1)
            # Rimuovi la dimensione del canale per la visualizzazione con plt.imshow per scala di grigi
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.title(f"Etichetta: {labels[i]}\n({CLASS_NAMES[labels[i]]})")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # 4. Divisione del Dataset in Training e Validation set
    # test_size=0.2 significa che il 20% dei dati andrà al validation set, l'80% al training.
    # random_state assicura che la divisione sia la stessa ogni volta che esegui lo script.
    # stratify=labels è utile per assicurarsi che la proporzione delle classi sia simile in entrambi i set.
    try:
        train_images, val_images, train_labels, val_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("\nDataset diviso con successo:")
        print(f"Immagini di addestramento: {train_images.shape}, Etichette di addestramento: {train_labels.shape}")
        print(f"Immagini di validazione: {val_images.shape}, Etichette di validazione: {val_labels.shape}")

        # 5. Salva i dati pre-elaborati (opzionale ma consigliato)
        # In questo modo non devi riprocessare tutto ogni volta che vuoi addestrare.
        # Puoi salvarli in un formato compresso NumPy (.npz) o file separati.
        processed_data_path = "preprocessed_dataset.npz"
        np.savez_compressed(processed_data_path,
                            train_images=train_images,
                            train_labels=train_labels,
                            val_images=val_images,
                            val_labels=val_labels,
                            class_names=CLASS_NAMES, # Salva anche i nomi delle classi
                            img_width=np.array([IMG_WIDTH]), # Salva anche le dimensioni usate
                            img_height=np.array([IMG_HEIGHT]))
        print(f"\nDati pre-elaborati e divisi salvati in: {processed_data_path}")

    except ValueError as e:
        print(f"\nERRORE durante la divisione del dataset: {e}")
        print("Questo può accadere se hai troppo poche immagini in una delle classi rispetto al numero di divisioni (splits).")
        print("Assicurati di avere almeno un certo numero di campioni per ogni classe (es. >5-10 per classe per poter fare lo split).")


if __name__ == '__main__':
    # Assicurati di avere matplotlib installato se vuoi visualizzare le immagini:
    # pip install matplotlib
    # (eseguilo nel terminale di PyCharm con .venv_arch_tf attivo se non l'hai già)
    main()