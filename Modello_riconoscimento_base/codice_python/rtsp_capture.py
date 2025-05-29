import cv2
import time

# ------------------- CONFIGURAZIONE -------------------
# Sostituisci questa stringa con l'URL RTSP della tua videocamera.
# Il formato tipico è: rtsp://utente:password@indirizzo_ip:porta/percorso_specifico_stream
# Esempio (senza autenticazione): rtsp://192.168.1.100:554/stream1
# Esempio (con autenticazione): rtsp://admin:password123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
RTSP_URL = "rtsp://localhost:8554/webcam_stream"

# Opzionale: ridimensionare i frame per la visualizzazione
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480


# ----------------------------------------------------

def main():
    print(f"Tentativo di connessione allo stream RTSP: {RTSP_URL}")

    # Prova ad aprire lo stream RTSP
    # Potrebbe essere necessario specificare cv2.CAP_FFMPEG o cv2.CAP_GSTREAMER
    # se OpenCV ha problemi a determinare il backend corretto.
    # cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Errore: Impossibile aprire lo stream RTSP.")
        print("Verifica l'URL RTSP, la connessione di rete e che la videocamera sia attiva.")
        print("Potrebbe anche essere necessario che FFmpeg o GStreamer siano installati e accessibili da OpenCV.")
        return

    print("Stream RTSP aperto con successo!")
    print("Premi 'q' per uscire.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Leggi un frame dalla videocamera
            ret, frame = cap.read()

            # Se ret è False, significa che non è stato possibile leggere il frame
            if not ret:
                print("Errore: Impossibile leggere il frame dallo stream. Lo stream potrebbe essersi interrotto.")
                # Prova a riaprire lo stream dopo una pausa
                cap.release()
                time.sleep(2)  # Pausa di 2 secondi
                cap = cv2.VideoCapture(RTSP_URL)
                if not cap.isOpened():
                    print("Errore: Impossibile riaprire lo stream. Uscita.")
                    break
                else:
                    print("Stream riaperto con successo.")
                    continue  # Riprova a leggere il frame

            # Puoi elaborare il 'frame' qui (ad esempio, passarlo a TensorFlow)
            # Per ora, lo visualizziamo e basta.

            # Ridimensiona il frame per una visualizzazione più gestibile (opzionale)
            if RESIZE_WIDTH > 0 and RESIZE_HEIGHT > 0:
                display_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            else:
                display_frame = frame

            cv2.imshow('Stream RTSP', display_frame)
            frame_count += 1

            # Interrompi il loop se viene premuto il tasto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Uscita richiesta dall'utente.")
                break

    except KeyboardInterrupt:
        print("Interruzione da tastiera rilevata. Uscita...")

    finally:
        # Rilascia la cattura e chiudi tutte le finestre
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            print(f"FPS medio: {fps:.2f}")

        print("Rilascio della videocamera e chiusura delle finestre.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()