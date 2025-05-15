import cv2
import time
import signal
import sys

# --- Configurazione ---
# URL dello stream RTSP (verificare l'URL se è corretto)
rtsp_url = "rtsp://localhost:8554/webcam_stream"

# Dimensioni fisse per l'immagine di output
# Imposta questi valori a una risoluzione MAGGIORE di quella originale per ingrandire
new_width = 1280
new_height = 960
# --- Fine Configurazione ---

# --- Gestione Uscita con Ctrl+C ---
# Variabile flag per segnalare quando terminare il loop
stop_program = False

# Funzione che viene chiamata quando si riceve il segnale SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("\nSegnale di interruzione (Ctrl+C) ricevuto.")
    global stop_program
    stop_program = True # Imposta il flag a True per uscire dal loop

# Registra la funzione signal_handler per gestire il segnale SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)
# --- Fine Gestione Uscita con Ctrl+C ---


# Inizializza la cattura video dall'URL RTSP
cap = cv2.VideoCapture(rtsp_url)

# Controlla se la connessione allo stream è riuscita
if not cap.isOpened():
    print(f"Errore: Impossibile connettersi allo stream RTSP all'URL: {rtsp_url}")
    # Utilizza sys.exit(1) per una pulita uscita con codice di errore
    sys.exit(1)

print(f"Connesso correttamente allo stream RTSP: {rtsp_url}")
print(f"Dimensione di visualizzazione impostata: {new_width}x{new_height}")
# Aggiorna il messaggio per indicare solo l'uscita con Ctrl+C
print("\nAvvio elaborazione stream. Premi Ctrl+C nel terminale per uscire.")


# --- Loop Principale di Elaborazione Frame ---
# Il loop continua finché il flag stop_program non diventa True
while not stop_program:
    # Leggi un frame dallo stream
    ret, frame = cap.read()

    # Se non si riesce a leggere il frame, imposta il flag e esce dal loop
    if not ret:
        print("Impossibile leggere il frame dallo stream. Fine o problema di connessione.")
        stop_program = True # Imposta il flag per garantire la pulizia finale
        # Non c'è bisogno di break qui, il loop si fermerà alla prossima iterazione
        # grazie alla condizione `while not stop_program:`
        continue # Salta il resto del loop in questa iterazione

    # --- Elaborazione del Frame ---
    # Ridimensiona il frame alle dimensioni fisse impostate
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # --- Fine Elaborazione ---

    # --- Visualizzazione ---
    # Mostra il frame elaborato (quello ridimensionato)
    cv2.imshow("Webcam Stream Elaborato", resized_frame)
    # --- Fine Visualizzazione ---

    # --- Mantiene la finestra aggiornata ---
    # Attende 1ms e gestisce gli eventi della finestra (ESSENZIALE per cv2.imshow)
    cv2.waitKey(1)
    # --- Fine Mantiene la finestra aggiornata ---

    # La parte che controllava il tasto 'q' è stata rimossa.


# --- Pulizia ---
# Queste righe vengono eseguite una volta che il loop `while not stop_program:` termina
# Rilascia la risorsa di cattura video
cap.release()
# Chiude tutte le finestre di OpenCV aperte
cv2.destroyAllWindows()

print("\nRisorse rilasciate. Programma terminato in modo corretto.")
# --- Fine Pulizia ---
