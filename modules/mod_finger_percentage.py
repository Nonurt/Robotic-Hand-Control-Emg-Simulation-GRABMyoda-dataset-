import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
import math
import json
import os
import matplotlib.pyplot as plt

CALIBRATION_PATH = "calibration_data.json"

class FingerPercentageEstimator:
    def __init__(self, smoothing=0.5):
        self.calibration_data = defaultdict(lambda: defaultdict(dict))  # finger: percent: angle
        self.prev_output = {}  # for low-pass
        self.smoothing = smoothing
        self.finger_points = {
            "Baş": [0, 2, 4],
            "İşaret": [0, 6, 8],
            "Orta": [0, 10, 12],
            "Yüzük": [0, 14, 16],
            "Serçe": [0, 18, 20],
        }

        self.load_calibration()

    def calculate_angle(self, p1, p2, p3):
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    def calibrate(self, landmarks, percent, finger=None):
        targets = [finger] if finger else self.finger_points.keys()
        for name in targets:
            i1, i2, i3 = self.finger_points[name]
            angle = self.calculate_angle(landmarks[i1], landmarks[i2], landmarks[i3])
            self.calibration_data[name][percent] = angle
        print(f"✅ Kalibrasyon: {finger or 'tümü'} için %{percent}")

    def estimate(self, landmarks):
        result = {}
        for name, (i1, i2, i3) in self.finger_points.items():
            p1, p2, p3 = landmarks[i1], landmarks[i2], landmarks[i3]
            angle = self.calculate_angle(p1, p2, p3)

            cal = self.calibration_data[name]
            if len(cal) < 2:
                result[name] = 0
                continue

            percents = sorted(cal.keys())
            angles = [cal[p] for p in percents]

            try:
                f = interp1d(angles, percents, kind='linear', fill_value="extrapolate", bounds_error=False)
                raw = float(f(angle))
                smooth = self._low_pass(name, raw)
                result[name] = np.clip(smooth, 0, 100)
            except Exception:
                result[name] = 0

        return result

    def _low_pass(self, name, new_val):
        old_val = self.prev_output.get(name, new_val)
        val = self.smoothing * old_val + (1 - self.smoothing) * new_val
        self.prev_output[name] = val
        return val

    def save_calibration(self):
        data = {f: dict(v) for f, v in self.calibration_data.items()}
        with open(CALIBRATION_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print("💾 Kalibrasyon kaydedildi.")

    def load_calibration(self):
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH, 'r') as f:
                raw = json.load(f)
            for finger, values in raw.items():
                self.calibration_data[finger] = {float(k): v for k, v in values.items()}
            print("✅ Kalibrasyon yüklendi.")
        else:
            print("⚠️ Kalibrasyon dosyası bulunamadı, yeni kalibrasyon yapılmalı.")

    def plot_calibration_graphs(self):
        os.makedirs("./calibration_graphs", exist_ok=True)

        for finger, cal in self.calibration_data.items():
            if len(cal) < 2:
                print(f"⚠️ {finger} için yeterli kalibrasyon verisi yok, grafik çizilemiyor.")
                continue

            # Kalibrasyon verilerini sırala
            percents = sorted(cal.keys())
            angles = [cal[p] for p in percents]

            plt.figure(figsize=(6, 4))
            plt.scatter(angles, percents, label="Kalibrasyon Noktaları", color='red')

            # Interpolasyon için model belirle
            x = np.array(angles)
            y = np.array(percents)

            degree = 1  # default linear

            if len(cal) == 3:
                degree = 2  # parabol
            elif len(cal) >= 4:
                degree = len(cal) - 1  # polinom derecesi nokta sayısına göre

            # Polinom katsayılarını hesapla
            p = np.polyfit(x, y, deg=degree)
            poly = np.poly1d(p)

            # Daha düzgün çizim için açı aralığı oluştur
            x_line = np.linspace(min(x) - 5, max(x) + 5, 200)
            y_line = poly(x_line)

            plt.plot(x_line, y_line, label=f"{degree}. Derece Polinom Fit", color='blue')

            plt.title(f"{finger} Parmak Kalibrasyon Grafiği")
            plt.xlabel("Açı (derece)")
            plt.ylabel("Yüzde (%)")
            plt.legend()
            plt.grid(True)

            # Dosya adı güvenli yap
            filename = f"./calibration_graphs/{finger}_calibration.png"
            plt.savefig(filename)
            plt.close()
            print(f"📊 {finger} için grafik kaydedildi: {filename}")
