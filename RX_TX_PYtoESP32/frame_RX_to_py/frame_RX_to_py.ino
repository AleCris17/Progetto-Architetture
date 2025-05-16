#include <WiFi.h>
#include <WebServer.h> // Libreria per il server web base

// --- Configurazione Wi-Fi ---
const char* ssid = "Alex il Leone";         // SOSTITUISCI CON IL NOME DELLA TUA RETE WI-FI
const char* password = "NapoliColera"; // SOSTITUISCI CON LA PASSWORD DELLA TUA WI-FI
// --- Fine Configurazione Wi-Fi ---

WebServer server(80); // Crea un'istanza del server sulla porta 80

// --- Funzione chiamata quando arriva un frame ---
void handleFrameUpload() {
  if (server.method() != HTTP_POST) {
    server.send(405, "text/plain", "Method Not Allowed");
    Serial.println("Richiesta non POST ricevuta su /upload_frame");
    return;
  }

  if (!server.hasArg("plain")) { // "plain" è il nome dell'argomento che contiene il corpo della richiesta POST non formattata
    server.send(400, "text/plain", "Bad Request: No data in body");
    Serial.println("Richiesta POST su /upload_frame senza dati nel corpo.");
    return;
  }

  // Il corpo della richiesta POST viene letto come una String.
  // Questo funziona bene per dati testuali e per dati binari che non contengono caratteri nulli prematuri (come i JPEG di solito).
  // Per file binari molto grandi o complessi, ESPAsyncWebServer è una scelta migliore, ma WebServer è più semplice per iniziare.
  String requestBody = server.arg("plain");
  int bodyLength = requestBody.length(); // Ottiene la lunghezza dei dati ricevuti

  Serial.printf("Frame JPEG ricevuto! Dimensione: %d bytes\n", bodyLength);

  // A QUESTO PUNTO, HAI I DATI JPEG IN 'requestBody'
  // Puoi accedere ai byte grezzi con:
  // const char* jpegBytes = requestBody.c_str();
  // size_t jpegSize = bodyLength;

  // --- COSA FARE CON I DATI DEL FRAME? ---
  // Qui puoi aggiungere il codice per:
  // 1. Salvare i dati su una microSD card:
  //    (Richiede FS.h, SD.h, e codice per inizializzare e scrivere sulla SD)
  //    Esempio concettuale:
  //    File imageFile = SD.open("/received_frame.jpg", FILE_WRITE);
  //    if (imageFile) {
  //      imageFile.write((const uint8_t*)jpegBytes, jpegSize);
  //      imageFile.close();
  //      Serial.println("Frame salvato su SD card.");
  //    } else {
  //      Serial.println("Errore apertura file su SD card.");
  //    }

  // 2. Decodificare il JPEG e mostrarlo su un display TFT:
  //    (Richiede librerie di decodifica JPEG come TJpg_Decoder e una libreria per il tuo display TFT)
  //    Questo è più complesso e richiede più risorse.

  // 3. Eseguire altre elaborazioni...

  // Invia una risposta HTTP 200 OK al client Python
  server.send(200, "text/plain", "Frame ricevuto con successo dall'ESP32!");
}

// --- Funzione per la pagina root (opzionale) ---
void handleRoot() {
  String html = "<html><head><title>ESP32 Webcam Receiver</title></head><body>";
  html += "<h1>ESP32 Server Attivo</h1>";
  html += "<p>Pronto a ricevere frame su /upload_frame (via POST).</p>";
  html += "<p>IP Address: " + WiFi.localIP().toString() + "</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

// --- Setup Iniziale ---
void setup() {
  Serial.begin(115200); // Inizializza la comunicazione seriale per il debug
  while (!Serial) {
    ; // Attendi che la porta seriale si connetta (necessario per alcuni ESP32)
  }
  Serial.println("\nESP32 Webcam Frame Receiver");

  // Connessione al Wi-Fi
  Serial.printf("Connessione a %s ", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnesso al Wi-Fi!");
  Serial.print("Indirizzo IP ESP32: ");
  Serial.println(WiFi.localIP()); // Stampa l'IP dell'ESP32 (IMPORTANTE!)

  // Configura gli handler del server web
  server.on("/", HTTP_GET, handleRoot);                 // Handler per la pagina principale
  server.on("/upload_frame", HTTP_POST, handleFrameUpload); // Handler per la ricezione dei frame

  server.begin(); // Avvia il server HTTP
  Serial.println("Server HTTP avviato. In ascolto sulla porta 80.");
  Serial.println("Pronto a ricevere frame su /upload_frame (POST)");
}

// --- Loop Principale ---
void loop() {
  server.handleClient(); // Gestisce le richieste HTTP in arrivo
  delay(10); // Piccolo ritardo per stabilità, puoi sperimentare
}