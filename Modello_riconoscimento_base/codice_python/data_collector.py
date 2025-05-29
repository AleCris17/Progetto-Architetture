import cv2
import time
import signal
import sys
import os

# --- Configurazione ---
RTSP_URL = "rtsp://localhost:8554/webcam_stream" # Verificalo con il tuo setup FFmpeg

# Chiedi all'utente per quale gesto/stato sta raccogliendo i dati
# Assicurati che il nome della cartella sia valido (senza spazi o caratteri speciali complessi)
gesture_name = input("Inserisci il nome della classe per cui stai raccogliendo i dati (es. 'mano_alzata', 'mano_abbassata'): ").strip().lower().replace(" ", "_")

DATASET_PATH = "dataset" # Cartella principale per tutte le classi
GESTURE_PATH = os.path.join(DATASET_PATH, gesture_name) # Sottocartella per la classe corrente

# Crea le cartelle se non esistono
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    print(f"Creata cartella dataset principale: {DATASET_PATH}")
if not os.path.exists(GESTURE_PATH):
    os.makedirs(GESTURE_PATH)
    print(f"Creata cartella per la classe '{gesture_name}': {GESTURE_PATH}")
else:
    print(f"Salvataggio frame per la classe '{gesture_name}' in: {GESTURE_PATH}")

# Dimensioni per la visualizzazione durante la raccolta
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Contatore per i frame salvati (inizia contando i file esistenti per non sovrascrivere)
saved_frame_count = 0
while True:
    img_name_check = os.path.join(GESTURE_PATH, f"{gesture_name}_{saved_frame_count:04d}.png")
    if not os.path.exists(img_name_check):
        break
    saved_frame_count += 1
print(f"Il prossimo frame verrà salvato come {gesture_name}_{saved_frame_count:04d}.png")
# --- Fine Configurazione ---

# --- Gestione Uscita con Ctrl+C ---
stop_program = False
def signal_handler(sig, frame_signal):
    print("\nSegnale di interruzione (Ctrl+C) ricevuto.")
    global stop_program
    stop_program = True
signal.signal(signal.SIGINT, signal_handler)
# --- Fine Gestione Uscita con Ctrl+C ---

print(f"Tentativo di connessione allo stream RTSP: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(f"Errore: Impossibile connettersi allo stream RTSP all'URL: {RTSP_URL}")
    print("Controlla che Mediamtx e FFmpeg siano in esecuzione e che l'URL sia corretto.")
    sys.exit(1)

print(f"Connesso correttamente allo stream RTSP.")
print(f"Mostrando frame a {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print("Premi 's' per salvare il frame corrente.")
print("Premi 'q' per uscire (o Ctrl+C nel terminale).")

last_save_time = time.time()
min_interval_between_saves = 0.2 # Secondi (es. 5 FPS max per il salvataggio)

while not stop_program:
    ret, frame = cap.read() # Legge il frame dalla sorgente RTSP (dovrebbe essere 640x480 da FFmpeg)
    if not ret:
        print("Impossibile leggere il frame. Lo stream potrebbe essersi interrotto.")
        stop_program = True
        continue

    # Ridimensiona solo per la visualizzazione, se necessario
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow(f"Raccolta Dati: '{gesture_name}' - Premi 's' per salvare, 'q' per uscire", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        current_time = time.time()
        if (current_time - last_save_time) > min_interval_between_saves:
            img_name = os.path.join(GESTURE_PATH, f"{gesture_name}_{saved_frame_count:04d}.png")
            # Salva il frame ORIGINALE catturato da FFmpeg (640x480)
            cv2.imwrite(img_name, frame)
            print(f"Salvato: {img_name}")
            saved_frame_count += 1
            last_save_time = current_time
        else:
            print("Salvataggio troppo ravvicinato, attendi un istante.")

    elif key == ord('q'):
        print("Uscita richiesta con 'q'.")
        stop_program = True

cap.release()
cv2.destroyAllWindows()
print("\nRisorse rilasciate. Acquisizione dati per questa classe terminata.")