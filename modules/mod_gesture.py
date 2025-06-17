import os
import time
import csv
import threading
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from modules.arduino import ArduinoComm
except ImportError:
    ArduinoComm = None  # Arduino bağlantısı opsiyonel olabilir

POZ_DIR = "./modules/pozlar"
MODEL_PATH = "./modules/gesturemodel/model.pkl"

if not os.path.exists(POZ_DIR):
    os.makedirs(POZ_DIR)
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

current_landmarks = None
arduino = None  # Arduino bağlantısı bu modül içinde tanımlı

# ✅ Gesture -> Servo yüzdeleri eşleşmesi
GESTURE_TO_SERVO = {
    "Lateral Prehension":    [0, 20, 20, 80, 90],
    "Thumb Adduction":       [30, 50, 30, 50, 30],
    "Thumb and Little Finger Opposition": [40, 40, 40, 40, 40],
    "Thumb and Index Finger Opposition":  [20, 70, 30, 30, 30],
    "Thumb and Index Finger Extension":   [10, 10, 90, 90, 90],
    "Thumb and Little Finger Extension":  [0, 30, 30, 30, 100],
    "Index and Middle Finger Extension":  [30, 100, 100, 30, 30],
    "Little Finger Extension":            [50, 50, 50, 50, 100],
    "Index Finger Extension":             [20, 100, 20, 20, 20],
    "Thumb Finger Extension":             [100, 20, 20, 20, 20],
    "Wrist Extension":                    [10, 10, 10, 10, 10],
    "Wrist Flexion":                      [90, 90, 90, 90, 90],
    "Forearm Supination":                [20, 30, 40, 50, 60],
    "Forearm Pronation":                 [60, 50, 40, 30, 20],
    "open_hand_right":                   [0, 0, 0, 0, 0],
    "Hand Close":                        [100, 100, 100, 100, 100]
}

def set_current_landmarks(landmarks):
    global current_landmarks
    current_landmarks = landmarks

def set_current_frame(frame):
    pass

def get_bounding_box(landmarks, margin=20):
    coords = np.array(landmarks)[:, :2]
    x_min = int(np.min(coords[:, 0])) - margin
    y_min = int(np.min(coords[:, 1])) - margin
    x_max = int(np.max(coords[:, 0])) + margin
    y_max = int(np.max(coords[:, 1])) + margin
    return x_min, y_min, x_max, y_max

def augment_and_save_direct(landmarks, label, writer):
    flat = np.array(landmarks).flatten()
    for _ in range(4):
        noisy = flat + np.random.normal(0, 0.5, size=flat.shape)
        writer.writerow(noisy)
        print(f"✅ Varyasyon kaydedildi ({label})")

def collect_samples(label, delay, samples, _unused_detector=None):
    filename = os.path.join(POZ_DIR, f"{label}.csv")
    def capture_loop():
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            count = 0
            while count < samples:
                if current_landmarks is not None:
                    augment_and_save_direct(current_landmarks, label, writer)
                    count += 1
                    print(f"📸 {count}/{samples} örnek alındı.")
                time.sleep(delay)
    threading.Thread(target=capture_loop, daemon=True).start()

def train_model():
    X, y = [], []
    for fname in os.listdir(POZ_DIR):
        if not fname.endswith(".csv"):
            continue
        label = os.path.splitext(fname)[0]
        with open(os.path.join(POZ_DIR, fname)) as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    X.append([float(r) for r in row])
                    y.append(label)

    if not X:
        print("⚠️ Eğitim verisi bulunamadı. Lütfen poz örneklerini kontrol edin.")
        return None

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print("✅ Model başarıyla eğitildi.")
    return model

def start_live_prediction(model, label_widget, send_callback=None):
    def loop():
        global arduino
        if ArduinoComm and arduino is None:
            try:
                arduino = ArduinoComm()
                print("🔌 Arduino bağlı.")
            except:
                print("⚠️ Arduino bağlanamadı.")
                arduino = None

        while True:
            try:
                if current_landmarks is not None:
                    flat = np.array(current_landmarks).flatten().reshape(1, -1)
                    pred = model.predict(flat)[0]

                    # ✅ UI’ye yaz
                    try:
                        if label_widget and label_widget.winfo_exists():
                            label_widget.config(text=f"🤖 Tahmin: {pred}")
                    except Exception as ui_err:
                        print("⚠️ UI güncellenemedi:", ui_err)

                    # ✅ Eğer dışarıdan gönderim fonksiyonu verilmişse çağır
                    if send_callback:
                        send_callback(pred)
                    elif arduino and pred in GESTURE_TO_SERVO:
                        arduino.send_percentages(GESTURE_TO_SERVO[pred])

            except Exception as e:
                print("❌ Tahmin hatası:", e)

            time.sleep(0.3)

    threading.Thread(target=loop, daemon=True).start()


def get_all_poses():
    poses = {}
    for fname in os.listdir(POZ_DIR):
        if fname.endswith(".csv"):
            name = os.path.splitext(fname)[0]
            with open(os.path.join(POZ_DIR, fname)) as f:
                poses[name] = sum(1 for _ in f)
    return poses

def delete_pose(name):
    fname = os.path.join(POZ_DIR, f"{name}.csv")
    if os.path.exists(fname):
        os.remove(fname)

def load_model():
    if os.path.exists(MODEL_PATH):
        print("📥 Model diskte bulundu, yükleniyor...")
        return joblib.load(MODEL_PATH)
    else:
        print("⚠️ Model dosyası bulunamadı.")
        return None
