import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from modules.arduino import ArduinoComm

ANGLE_MAP_PATH = "angle_map.json"

class FingerPercentageUI:
    def __init__(self, root, return_callback=None):
        self.root = root
        self.return_callback = return_callback
        for widget in self.root.winfo_children():
            widget.destroy()

        self.selected_finger = tk.StringVar(value="Tümü")
        self._last_landmarks = None
        self.sending = False
        self.arduino = None
        self.angle_map = self.load_angle_map()

        # Yeni: Gönderim eşik ve aralığı
        self.threshold = tk.IntVar(value=7)        # %7 default
        self.send_interval = tk.IntVar(value=50)   # 50 ms default
        self.last_sent_angles = [None] * 5         # Önceki gönderilen açıları tutar

        ttk.Label(root, text="Parmak Yüzdesi", font=("Arial", 12)).pack(pady=5)
        ttk.Label(root, text="Parmak Seç (opsiyonel)").pack()

        fingers = ["Tümü", "Baş", "İşaret", "Orta", "Yüzük", "Serçe"]
        ttk.OptionMenu(root, self.selected_finger, "Tümü", *fingers).pack(pady=2)

        ttk.Button(root, text="Açık El (%0)", command=lambda: self.calibrate(0)).pack(pady=2)
        ttk.Button(root, text="Kapalı El (%100)", command=lambda: self.calibrate(100)).pack(pady=2)

        optional = ttk.LabelFrame(root, text="Opsiyonel Kalibrasyonlar")
        optional.pack(pady=5)
        for val in [25, 50, 75]:
            ttk.Button(optional, text=f"%{val}", command=lambda v=val: self.calibrate(v)).pack(pady=1)

        frame = ttk.Frame(optional)
        frame.pack(pady=2)
        self.custom_percent = tk.DoubleVar(value=50)
        ttk.Scale(frame, from_=0, to=100, variable=self.custom_percent, orient="horizontal", length=150).pack(side="left")
        self.percent_label = ttk.Label(frame, text="50%")
        self.percent_label.pack(side="left", padx=5)
        ttk.Button(optional, text="Uygula", command=self.apply_custom).pack()
        self.custom_percent.trace_add("write", self.update_percent_label)

        self.labels = {}
        self.current_values = [0] * 5
        for i, name in enumerate(["Baş", "İşaret", "Orta", "Yüzük", "Serçe"]):
            var = tk.StringVar(value="0%")
            ttk.Label(root, text=name).pack()
            lbl = ttk.Label(root, textvariable=var)
            lbl.pack()
            self.labels[name] = var

        # Yeni: Gönderim eşiği ayarı
        threshold_frame = ttk.Frame(root)
        threshold_frame.pack(pady=5, fill="x", padx=10)
        ttk.Label(threshold_frame, text="Gönderim Eşiği (%):").pack(side="left")
        ttk.Scale(threshold_frame, from_=0, to=20, variable=self.threshold, orient="horizontal", length=150).pack(side="left", padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.threshold.get()}%")
        self.threshold_label.pack(side="left")
        self.threshold.trace_add("write", self.update_threshold_label)

        # Yeni: Gönderim aralığı ayarı
        interval_frame = ttk.Frame(root)
        interval_frame.pack(pady=5, fill="x", padx=10)
        ttk.Label(interval_frame, text="Gönderim Hızı (ms):").pack(side="left")
        ttk.Scale(interval_frame, from_=10, to=200, variable=self.send_interval, orient="horizontal", length=150).pack(side="left", padx=5)
        self.interval_label = ttk.Label(interval_frame, text=f"{self.send_interval.get()} ms")
        self.interval_label.pack(side="left")
        self.send_interval.trace_add("write", self.update_interval_label)

        self.toggle_button = ttk.Button(root, text="🔌 Arduino’ya Gönderimi Başlat", command=self.toggle_sending)
        self.toggle_button.pack(pady=5)

        ttk.Label(root, text="Açı Ayarları (Servo)", font=("Arial", 11)).pack(pady=5)
        self.angle_entries = {}
        angle_frame = ttk.LabelFrame(root, text="Min (Açık) / Max (Kapalı)")
        angle_frame.pack(pady=5)
        for finger in ["Baş", "İşaret", "Orta", "Yüzük", "Serçe"]:
            row = ttk.Frame(angle_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=finger, width=7).pack(side="left")
            min_var = tk.IntVar(value=self.angle_map[finger][0])
            max_var = tk.IntVar(value=self.angle_map[finger][1])
            tk.Entry(row, textvariable=min_var, width=5).pack(side="left", padx=2)
            tk.Entry(row, textvariable=max_var, width=5).pack(side="left", padx=2)
            self.angle_entries[finger] = (min_var, max_var)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="💾 Kaydet", command=self.save_angle_map).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📂 Yükle", command=self.reload_angle_map).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🔄 Sıfırla", command=self.reset_angle_map).pack(side="left", padx=5)

        if self.return_callback:
            ttk.Button(root, text="◀ Geri", command=self.exit_and_save).pack(pady=5)

        self.root.after(self.send_interval.get(), self.send_loop)

    def update_percent_label(self, *_):
        self.percent_label.config(text=f"{self.custom_percent.get():.0f}%")

    def update_threshold_label(self, *_):
        self.threshold_label.config(text=f"{self.threshold.get()}%")

    def update_interval_label(self, *_):
        self.interval_label.config(text=f"{self.send_interval.get()} ms")

    def calibrate(self, percent):
        if self._last_landmarks and hasattr(self, 'estimator'):
            finger = None if self.selected_finger.get() == "Tümü" else self.selected_finger.get()
            self.estimator.calibrate(self._last_landmarks, percent, finger)

    def apply_custom(self):
        self.calibrate(self.custom_percent.get())

    def update_from_landmarks(self, landmarks):
        self._last_landmarks = landmarks
        if not hasattr(self, 'estimator'):
            from modules.mod_finger_percentage import FingerPercentageEstimator
            self.estimator = FingerPercentageEstimator()
        result = self.estimator.estimate(landmarks)
        for i, (name, val) in enumerate(result.items()):
            if name in self.labels:
                self.labels[name].set(f"{val:.0f}%")
            self.current_values[i] = int(val)

    def toggle_sending(self):
        self.sending = not self.sending
        if self.sending:
            if self.arduino is None:
                self.arduino = ArduinoComm()
            self.toggle_button.config(text="⏹️ Gönderimi Durdur")
        else:
            if self.arduino:
                # Tüm parmakları açık konuma gönder
                messages = []
                for i in range(5):
                    open_angle = 0
                    messages.append(f"{i}:{open_angle}")
                    self.last_sent_angles[i] = open_angle
                message_str = ",".join(messages)
                self.arduino.send_raw(message_str)

                self.arduino.close()
                self.arduino = None
            self.toggle_button.config(text="🔌 Arduino’ya Gönderimi Başlat")

    def send_loop(self):
        if self.sending and self.arduino:
            messages = []
            for i, finger in enumerate(["Baş", "İşaret", "Orta", "Yüzük", "Serçe"]):
                percent = self.current_values[i]
                min_angle, max_angle = self.angle_map[finger]
                angle = int(self.map_percent_to_angle_reverse(percent, min_angle, max_angle))

                last_angle = self.last_sent_angles[i]
                if last_angle is None or abs(angle - last_angle) >= self.threshold.get():
                    messages.append(f"{i}:{angle}")
                    self.last_sent_angles[i] = angle

            if messages:
                message_str = ",".join(messages)
                self.arduino.send_raw(message_str)

        self.root.after(self.send_interval.get(), self.send_loop)

    def map_percent_to_angle_reverse(self, percent, open_angle, closed_angle):
        # %0 -> open_angle (180), %100 -> closed_angle (60) ters dönüşüm
        return open_angle - (open_angle - closed_angle) * (percent / 100)

    def save_angle_map(self):
        for finger, (min_var, max_var) in self.angle_entries.items():
            self.angle_map[finger] = (min_var.get(), max_var.get())
        with open(ANGLE_MAP_PATH, "w") as f:
            json.dump(self.angle_map, f, indent=2)
        messagebox.showinfo("Kaydedildi", "Açı ayarları kaydedildi.")

    def load_angle_map(self):
        if os.path.exists(ANGLE_MAP_PATH):
            with open(ANGLE_MAP_PATH, "r") as f:
                return json.load(f)
        return {
            "Baş": [180, 60],
            "İşaret": [180, 60],
            "Orta": [180, 60],
            "Yüzük": [180, 60],
            "Serçe": [180, 60],
        }

    def reload_angle_map(self):
        self.angle_map = self.load_angle_map()
        for finger, (min_var, max_var) in self.angle_entries.items():
            min_angle, max_angle = self.angle_map[finger]
            min_var.set(min_angle)
            max_var.set(max_angle)
        messagebox.showinfo("Yüklendi", "Açı ayarları yüklendi.")

    def reset_angle_map(self):
        self.angle_map = {
            "Baş": [0, 60],
            "İşaret": [0, 60],
            "Orta": [0, 60],
            "Yüzük": [0, 60],
            "Serçe": [0, 60],
        }
        for finger, (min_var, max_var) in self.angle_entries.items():
            min_var.set(180)
            max_var.set(60)
        messagebox.showinfo("Sıfırlandı", "Tüm açı ayarları sıfırlandı.")

    def exit_and_save(self):
        if hasattr(self, 'estimator'):
            self.estimator.save_calibration()
            self.estimator.plot_calibration_graphs()
        self.save_angle_map()
        if self.arduino:
            self.arduino.close()
        if self.return_callback:
            self.return_callback()