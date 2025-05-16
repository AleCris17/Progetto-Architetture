import cv2
import time
import signal
import sys
import requests # Per inviare richieste HTTP

# --- Configurazione Essenziale ---
# URL dello stream RTSP (dal tuo server mediamtx)
rtsp_url = "rtsp://localhost:8554/webcam_stream"

# URL dell'endpoint sull'ESP32 che riceverà il frame
# !!! IMPORTANTE: SOSTITUISCI <IP_ESP32> CON L'INDIRIZZO IP REALE DEL TUO ESP32 !!!
# !!! Assicurati anche che l'endpoint (es. '/upload_frame') e la porta (se non è 80) corrispondano a quelli del server sull'ESP32 !!!
esp32_target_url = "http://192.168.217.67:80/upload_frame"
# Esempio con porta diversa: esp32_target_url = "http://<IP_ESP32>:8080/upload_frame"

# Qualità della compressione JPEG (0-100, più alto è meglio ma più grande il file)
jpeg_quality = 85
# --- Fine Configurazione Essenziale ---

# --- Configurazione Opzionale Visualizzazione Locale ---
# Se vuoi visualizzare i frame anche sul PC dove gira questo script
enable_local_display = True
local_display_window_name = "Webcam Stream (PC)"
# Se vuoi ridimensionare la finestra di visualizzazione locale (non influisce sul frame inviato all'ESP32)
# Queste dimensioni sono solo per la finestra locale. Il frame inviato all'ESP32
# avrà le dimensioni originali dello stream (es. 160x120, come impostato in ffmpeg).
local_display_width = 640 # Larghezza per la visualizzazione locale
local_display_height = 480 # Altezza per la visualizzazione locale
# --- Fine Configurazione Opzionale Visualizzazione Locale ---


# --- Gestione Uscita con Ctrl+C ---
stop_program_flag = False

def signal_handler_function(sig, frame_signal):
    print("\nSegnale di interruzione (Ctrl+C) ricevuto. Uscita in corso...")
    global stop_program_flag
    stop_program_flag = True

signal.signal(signal.SIGINT, signal_handler_function)
# --- Fine Gestione Uscita con Ctrl+C ---

def main():
    global stop_program_flag # Necessario se modifichi stop_program_flag in una funzione annidata (non il caso qui, ma buona pratica)

    # Inizializza la cattura video dall'URL RTSP
    video_capture = cv2.VideoCapture(rtsp_url)

    if not video_capture.isOpened():
        print(f"ERRORE: Impossibile connettersi allo stream RTSP all'URL: {rtsp_url}")
        sys.exit(1)

    print(f"Connesso correttamente allo stream RTSP: {rtsp_url}")
    print(f"I frame verranno inviati (come JPEG qualità {jpeg_quality}) a: {esp32_target_url}")
    if enable_local_display:
        print(f"La visualizzazione locale è attiva sulla finestra: '{local_display_window_name}'")
    print("\nAvvio elaborazione stream. Premi Ctrl+C nel terminale per uscire.")
    if enable_local_display:
        print("Puoi anche premere 'q' sulla finestra di visualizzazione per uscire.")


    # --- Loop Principale di Elaborazione Frame ---
    while not stop_program_flag:
        # Leggi un frame dallo stream
        success, frame_data = video_capture.read()

        if not success:
            print("Impossibile leggere il frame dallo stream. Potrebbe essere terminato o c'è un problema di connessione.")
            # stop_program_flag = True # Decidi se uscire o tentare di riconnettersi
            time.sleep(0.5) # Attendi un po' prima di riprovare o uscire
            continue # Salta il resto del loop e prova a leggere il prossimo frame

        # --- Elaborazione del Frame per l'invio ---
        # Il frame_data letto ha le dimensioni definite in ffmpeg (es. 160x120)
        # Codifica il frame in JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        result, encoded_jpeg = cv2.imencode('.jpg', frame_data, encode_param)

        if not result:
            print("ERRORE: Durante la codifica del frame in JPEG.")
            continue

        # --- Invio del frame JPEG all'ESP32 ---
        try:
            response = requests.post(
                esp32_target_url,
                data=encoded_jpeg.tobytes(),
                headers={'Content-Type': 'image/jpeg'},
                timeout=3 # Timeout in secondi (es. 3 secondi)
            )

            # Controlla la risposta dall'ESP32 (opzionale ma utile per debug)
            if response.status_code == 200:
                print(f"Frame inviato con successo. Risposta ESP32: {response.text[:100]}") # Mostra i primi 100 caratteri della risposta
            else:
                print(f"ERRORE invio frame. Status: {response.status_code}, Risposta ESP32: {response.text[:200]}")

        except requests.exceptions.RequestException as e:
            print(f"ERRORE di connessione/richiesta all'ESP32: {e}")
            # Potresti voler attendere un po' prima di riprovare per non sovraccaricare di log
            time.sleep(1)


        # --- Visualizzazione Locale (Opzionale) ---
        if enable_local_display:
            # Ridimensiona per la visualizzazione locale se le dimensioni sono diverse
            display_frame_local = cv2.resize(frame_data, (local_display_width, local_display_height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(local_display_window_name, display_frame_local)

            # Attende un breve periodo e gestisce gli eventi della finestra (ESSENZIALE per cv2.imshow)
            # Permette anche di uscire premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Tasto 'q' premuto sulla finestra, uscita in corso...")
                stop_program_flag = True
        # --- Fine Visualizzazione Locale ---

        # Piccolo ritardo per non sovraccaricare CPU/rete se il framerate RTSP fosse inaspettatamente alto
        # Se il tuo stream RTSP è già a basso framerate (es. 1 frame ogni 3 sec), questo potrebbe non essere necessario
        # o potresti volerlo più corto.
        # time.sleep(0.01) # Pausa molto breve

    # --- Pulizia ---
    print("\nRilascio risorse...")
    video_capture.release()
    if enable_local_display:
        cv2.destroyAllWindows()
    print("Programma terminato.")
    # --- Fine Pulizia ---

if __name__ == '__main__':
    main()
