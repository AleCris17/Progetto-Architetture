import cv2
import numpy as np
import tensorflow as tf
import time
import signal
import sys
import os

# --- Parametri ---
TFLITE_MODEL_PATH = "hand_gesture_model.tflite"
RTSP_URL = "rtsp://localhost:8554/webcam_stream" # Il tuo URL RTSP

# Carica i nomi delle classi e le dimensioni dell'immagine dal file .npz
# per assicurare consistenza con l'addestramento.
PREPROCESSED_DATA_FILE = "preprocessed_dataset.npz"

try:
    with np.load(PREPROCESSED_DATA_FILE) as data:
        CLASS_NAMES = data['class_names']
        IMG_WIDTH = int(data['img_width'][0])
        IMG_HEIGHT = int(data['img_height'][0])
    print(f"Caricati nomi classi: {CLASS_NAMES}, IMG_HEIGHT: {IMG_HEIGHT}, IMG_WIDTH: {IMG_WIDTH}")
except FileNotFoundError:
    print(f"ERRORE: File '{PREPROCESSED_DATA_FILE}' non trovato.")
    print("Questo file è necessario per ottenere i nomi delle classi e le dimensioni delle immagini usate per l'addestramento.")
    print("Assicurati di aver eseguito prima lo script 'preprocess_data.py'.")
    sys.exit(1)
except KeyError as e:
    print(f"ERRORE: Chiave mancante ({e}) nel file '{PREPROCESSED_DATA_FILE}'.")
    print("Assicurati che il file .npz contenga 'class_names', 'img_width' e 'img_height'.")
    sys.exit(1)

# Dimensioni per la visualizzazione
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
# --- Fine Parametri ---

# --- Gestione Uscita con Ctrl+C ---
stop_program = False
def signal_handler(sig, frame_signal):
    print("\nSegnale di interruzione (Ctrl+C) ricevuto.")
    global stop_program
    stop_program = True
signal.signal(signal.SIGINT, signal_handler)
# --- Fine Gestione Uscita con Ctrl+C ---

def preprocess_frame(frame, target_height, target_width):
    """Pre-elabora un singolo frame come fatto per l'addestramento."""
    img_resized = cv2.resize(frame, (target_width, target_height))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = np.expand_dims(img_gray, axis=-1).astype(np.float32) / 255.0
    # Il modello TFLite si aspetta un batch di immagini, quindi aggiungiamo una dimensione batch (1)
    return np.expand_dims(img_normalized, axis=0)


def main():
    global stop_program
    # Carica il modello TFLite e alloca i tensori.
    print(f"Caricamento del modello TFLite da: {TFLITE_MODEL_PATH}")
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors() # Fondamentale!
        print("Modello TFLite caricato e tensori allocati.")
    except Exception as e:
        print(f"ERRORE durante il caricamento del modello TFLite: {e}")
        print("Assicurati che il file '.tflite' esista e sia valido.")
        return

    # Ottieni i dettagli dei tensori di input e output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input tensor details: {input_details}")
    print(f"  Output tensor details: {output_details}")

    # Verifica che le dimensioni di input del modello TFLite corrispondano
    # a IMG_HEIGHT, IMG_WIDTH (dopo aver aggiunto batch e canali)
    # input_shape atteso è (1, IMG_HEIGHT, IMG_WIDTH, 1)
    expected_input_shape = (1, IMG_HEIGHT, IMG_WIDTH, 1)
    if tuple(input_details[0]['shape']) != expected_input_shape:
        print(f"ATTENZIONE: La forma dell'input del modello TFLite {input_details[0]['shape']} "
              f"non corrisponde alla forma attesa {expected_input_shape} "
              f"basata su IMG_HEIGHT/IMG_WIDTH dal file .npz.")
        # Potresti voler terminare o gestire questo caso, ma per ora continuiamo.

    print(f"\nTentativo di connessione allo stream RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"Errore: Impossibile connettersi allo stream RTSP: {RTSP_URL}")
        sys.exit(1)

    print("Connesso allo stream RTSP. Avvio inferenza...")
    print("Premi 'q' nella finestra del video per uscire (o Ctrl+C nel terminale).")

    font = cv2.FONT_HERSHEY_SIMPLEX

    while not stop_program:
        ret, frame_bgr = cap.read() # frame_bgr perché OpenCV legge in formato BGR
        if not ret:
            print("Impossibile leggere il frame. Stream terminato?")
            stop_program = True
            continue

        # 1. Pre-elabora il frame catturato
        input_data = preprocess_frame(frame_bgr, IMG_HEIGHT, IMG_WIDTH)

        # 2. Imposta il tensore di input
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 3. Esegui l'inferenza
        interpreter.invoke()

        # 4. Ottieni i risultati dell'output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # output_data è un array di probabilità, es. [[0.1, 0.8, 0.1]] per 3 classi

        predicted_class_index = np.argmax(output_data[0])
        prediction_confidence = output_data[0][predicted_class_index]

        try:
            predicted_class_name = CLASS_NAMES[predicted_class_index]
        except IndexError:
            predicted_class_name = "Classe Sconosciuta"


        # Prepara il frame per la visualizzazione
        display_frame = cv2.resize(frame_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Scrivi la predizione sul frame
        text = f"{predicted_class_name} ({prediction_confidence*100:.1f}%)"
        cv2.putText(display_frame, text, (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Test Modello TFLite - Webcam RTSP", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Uscita richiesta con 'q'.")
            stop_program = True

    cap.release()
    cv2.destroyAllWindows()
    print("\nRisorse rilasciate. Test terminato.")

if __name__ == '__main__':
    main()