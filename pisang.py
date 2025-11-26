import cv2
import numpy as np
import random

def klasifikasi_hue(hue_value):
    # Rentang Hue untuk klasifikasi kematangan
    if 40 <= hue_value <= 80:
        return "Segar", "Belum Matang"
    if 25 <= hue_value <= 39:
        return "Matang Optimal", "Siap Makan"
    if 0 <= hue_value <= 24:
        return "Mulai Busuk", "Terlalu Matang"
    return "Tidak Terkelompok", "Unknown"

def simulasi_sensor(kategori):
    """
    Fungsi ini mensimulasikan pembacaan sensor berdasarkan
    fase kematangan pisang secara biologis.
    Nilai di-random sedikit agar terlihat seperti pembacaan sensor real-time.
    """
    co2 = 0.0
    ph = 0.0
    kelembaban = 0.0

    if kategori == "Segar":
        # Fase Pre-klimakterik: Respirasi rendah, pH asam (pati tinggi)
        co2 = random.uniform(200, 350)      # ppm (rendah)
        ph = random.uniform(4.5, 5.2)       # Asam
        kelembaban = random.uniform(85, 90) # Kadar air tinggi/keras

    elif kategori == "Matang Optimal":
        # Puncak Klimakterik: Respirasi tinggi, pH naik (gula terbentuk)
        co2 = random.uniform(450, 600)      # ppm (tinggi/puncak)
        ph = random.uniform(5.3, 6.2)       # Manis/Sedang
        kelembaban = random.uniform(80, 85) # Optimal

    elif kategori == "Mulai Busuk":
        # Fase Senescence: Struktur rusak, fermentasi
        co2 = random.uniform(300, 500)      # ppm (bervariasi/fermentasi)
        ph = random.uniform(6.3, 7.0)       # Mendekati netral/tinggi
        kelembaban = random.uniform(60, 75) # Kehilangan air/lembek

    else:
        return 0, 0, 0

    return round(co2, 1), round(ph, 2), round(kelembaban, 1)

# --- Setup Kamera & Parameter ---
cap = cv2.VideoCapture(0)

# Rentang HSV Kuning & Hijau Pisang (Diperlebar sedikit agar lebih sensitif)
lower_banana = np.array([10, 40, 40])
upper_banana = np.array([80, 255, 255])

MIN_AREA = 5000  
MIN_RATIO = 1.5 

print("Tekan 'ESC' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame agar seperti cermin
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_banana, upper_banana)

    # Filter noise (Erosi & Dilasi)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False 

    if contours:
        # Ambil kontur terbesar
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            ratio = max(w, h) / max(1, min(w, h))

            if ratio >= MIN_RATIO:
                detected = True
                
                # 1. Analisis Warna (Hue)
                banana_region_hsv = hsv[y:y+h, x:x+w]
                H_avg = int(np.mean(banana_region_hsv[:, :, 0]))
                
                status, deskripsi = klasifikasi_hue(H_avg)
                
                # 2. Dapatkan Data Simulasi Sensor
                val_co2, val_ph, val_hum = simulasi_sensor(status)

                # --- VISUALISASI UI ---
                # Gambar kotak di sekitar pisang
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                # Background semi-transparan untuk teks agar mudah dibaca
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y-130), (x+w+50, y), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Tampilkan Status Utama
                color_text = (0, 255, 0) if status == "Matang Optimal" else (255, 255, 255)
                cv2.putText(frame, f"{status} (Hue: {H_avg})", (x+5, y-105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
                
                # Tampilkan Data Sensor (Simulasi)
                font_scale = 0.5
                color_sensor = (100, 255, 255) # Kuning muda
                
                cv2.putText(frame, f"Est. CO2 : {val_co2} ppm", (x+5, y-80),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_sensor, 1)
                
                cv2.putText(frame, f"Est. pH  : {val_ph}", (x+5, y-55),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_sensor, 1)
                
                cv2.putText(frame, f"Kelembaban: {val_hum}%", (x+5, y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_sensor, 1)

    # Indikator jika tidak ada objek
    if not detected:
        cv2.putText(frame, "Mencari Pisang...", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Smart Banana Analyzer + IoT Sensor Sim", frame)
    # cv2.imshow("Mask", mask) # Uncomment jika ingin melihat mask hitam putih

    if cv2.waitKey(1) & 0xFF == 27: # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()