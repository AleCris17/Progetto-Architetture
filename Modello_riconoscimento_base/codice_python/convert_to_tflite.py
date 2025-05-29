import tensorflow as tf
import numpy as np # Lo useremo se decidiamo di fare quantizzazione INT8

# --- Parametri ---
KERAS_MODEL_PATH = "hand_gesture_model.keras"  # Percorso al modello Keras salvato
TFLITE_MODEL_PATH = "hand_gesture_model.tflite" # Nome del file per il modello TFLite

# Opzionale: per la quantizzazione INT8, avremo bisogno di un piccolo subset dei dati di training
# PREPROCESSED_DATA_FILE = "preprocessed_dataset.npz" # Se vuoi fare quantizzazione INT8
# NUM_CALIBRATION_IMAGES = 100 # Numero di immagini da usare per la calibrazione INT8
# --- Fine Parametri ---

def main():
    # Carica il modello Keras addestrato
    print(f"Caricamento del modello Keras da: {KERAS_MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print("Modello Keras caricato con successo.")
    except Exception as e:
        print(f"ERRORE durante il caricamento del modello Keras: {e}")
        print("Assicurati che il file 'hand_gesture_model.keras' esista e sia stato salvato correttamente dallo script di addestramento.")
        return

    # Converti il modello in TensorFlow Lite
    # Creiamo un convertitore dall'oggetto modello Keras
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # === OPZIONI DI OTTIMIZZAZIONE (Quantizzazione) ===
    # Per TinyML, la quantizzazione è molto importante per ridurre le dimensioni del modello
    # e migliorare la velocità di inferenza sui microcontrollori.
    # Puoi scegliere tra diverse strategie. Iniziamo con la conversione base (solo float32).
    # Poi puoi decommentare e provare le altre.

    # 1. Conversione Base (Default, tensori in Float32)
    # Non ci sono ottimizzazioni particolari qui, ma è il punto di partenza.
    # converter.optimizations = [] # Già il default se non specificato

    # 2. Quantizzazione Dinamica dei Pesi (Opzionale, semplice da applicare)
    # Riduce le dimensioni del modello quantizzando solo i pesi a 8-bit. L'attivazione è ancora float.
    # print("\nApplicazione della quantizzazione dinamica dei pesi...")
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. Quantizzazione Float16 (Opzionale, buon compromesso)
    # Converte pesi e attivazioni in float16. Richiede hardware che supporti float16 o emulazione.
    # Riduce le dimensioni del modello di circa la metà rispetto a float32.
    # print("\nApplicazione della quantizzazione Float16...")
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]

    # 4. Quantizzazione Intera (INT8) - Massima ottimizzazione per TinyML (Più Complessa)
    # Converte pesi e attivazioni in interi a 8-bit. Richiede un dataset rappresentativo per la calibrazione.
    # Questa offre la maggiore riduzione delle dimensioni e spesso la migliore accelerazione su hardware compatibile.
    # print("\nApplicazione della quantizzazione INT8 completa...")
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] # Abilita la quantizzazione

    # # Carica un subset dei dati di training per la calibrazione
    # try:
    #     with np.load(PREPROCESSED_DATA_FILE) as data:
    #         calibration_images = data['train_images'][:NUM_CALIBRATION_IMAGES]
    #     print(f"Usate {len(calibration_images)} immagini per la calibrazione INT8.")
    # except FileNotFoundError:
    #     print(f"ERRORE: File {PREPROCESSED_DATA_FILE} non trovato. Necessario per la calibrazione INT8.")
    #     print("Assicurati di aver eseguito prima lo script di pre-elaborazione.")
    #     return
    # except KeyError:
    #     print(f"ERRORE: 'train_images' non trovate in {PREPROCESSED_DATA_FILE}. Necessario per la calibrazione INT8.")
    #     return

    # def representative_dataset_gen():
    #     for value in calibration_images:
    #         # Ogni 'value' è un'immagine, deve essere wrappata in una lista e avere il tipo corretto
    #         yield [np.array(value, dtype=np.float32)] # TensorFlow si aspetta un float32 per il dataset rappresentativo

    # converter.representative_dataset = representative_dataset_gen
    # # Forza l'input e l'output del modello TFLite a essere interi (comune per microcontrollori)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8  # o tf.uint8
    # converter.inference_output_type = tf.int8 # o tf.uint8
    # --- Fine Opzioni di Ottimizzazione ---

    # Esegui la conversione
    try:
        tflite_model = converter.convert()
        print("\nConversione in TensorFlow Lite completata.")
    except Exception as e:
        print(f"ERRORE durante la conversione in TensorFlow Lite: {e}")
        # Se usi la quantizzazione INT8 e fallisce, spesso è un problema con
        # il representative_dataset_gen o con operatori non supportati per INT8.
        return

    # Salva il modello TFLite su file
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Modello TensorFlow Lite salvato come: {TFLITE_MODEL_PATH}")
    print(f"Dimensioni del modello TFLite: {len(tflite_model) / 1024:.2f} KB")

if __name__ == '__main__':
    main()