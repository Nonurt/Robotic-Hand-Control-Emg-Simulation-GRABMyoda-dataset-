import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import time
import numpy as np
import threading
from modules import mod_gesture
from sklearn.metrics import accuracy_score
from modules.arduino import ArduinoComm  # Arduino entegrasyonu

class GestureUI:
    def __init__(self, parent, return_callback):
        self.parent = parent
        self.return_callback = return_callback
        self.delay = tk.DoubleVar(value=0.2)
        self.sample_count = tk.IntVar(value=20)
        self.new_label = tk.StringVar()

        self.frame = tk.Frame(parent)
        self.frame.pack(fill="both", expand=True)

        ttk.Label(self.frame, text="Yeni Poz İsmi:").pack(pady=(10, 0))
        self.entry = ttk.Entry(self.frame, textvariable=self.new_label)
        self.entry.pack()

        ttk.Label(self.frame, text="Örnekleme Gecikmesi (sn):").pack()
        ttk.Entry(self.frame, textvariable=self.delay).pack()

        ttk.Label(self.frame, text="Toplam Örnek Sayısı:").pack()
        ttk.Entry(self.frame, textvariable=self.sample_count).pack()

        ttk.Button(self.frame, text="📸 Örnek Al", command=self.collect_samples).pack(pady=5)
        ttk.Button(self.frame, text="🧠 Modeli Yeniden Eğit", command=self.train_model).pack(pady=5)
        ttk.Button(self.frame, text="📊 Modeli Test Et", command=self.test_model).pack(pady=5)

        ttk.Separator(self.frame).pack(pady=10, fill="x")
        ttk.Label(self.frame, text="Mevcut Pozlar:").pack(pady=(10, 0))

        list_frame = tk.Frame(self.frame)
        list_frame.pack(pady=5)

        self.pose_listbox = tk.Listbox(list_frame, height=10, width=25)
        self.pose_listbox.pack(side="left", fill="y")

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        scrollbar.config(command=self.pose_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.pose_listbox.config(yscrollcommand=scrollbar.set)

        self.pose_listbox.bind("<Double-Button-1>", self.copy_selected_pose)

        self.status_label = ttk.Label(self.frame, text="")
        self.status_label.pack(pady=5)

        ttk.Button(self.frame, text="🗑️ Seçili Poztan Sil", command=self.delete_selected_pose).pack(pady=(5, 0))
        ttk.Button(self.frame, text="⏪ Geri Dön", command=self.exit_and_save).pack(pady=(10, 0))

        self.model = None
        self.pred_label = ttk.Label(self.frame, text="🤖 Tahmin: -", font=("Arial", 12, "bold"))
        self.pred_label.pack(pady=5)

        self.update_pose_list()

        # Arduino gönderim kontrolü
        self.arduino = None
        self.arduino_sending = False
        self.toggle_btn = ttk.Button(self.frame, text="🔌 Arduino Gönderimini Başlat", command=self.toggle_arduino)
        self.toggle_btn.pack(pady=5)

    def collect_samples(self):
        label = self.new_label.get().strip()
        delay = self.delay.get()
        count = self.sample_count.get()

        if not label:
            messagebox.showerror("Hata", "Lütfen bir poz etiketi girin.")
            return

        self.status_label.config(text=f"📋 '{label}' ismi kopyalandı.")
        self.after_clear_status()

        def sample_thread():
            mod_gesture.collect_samples(label, delay, count)

            def delayed_update():
                total_delay = delay * count + 1
                time.sleep(total_delay)
                self.update_pose_list()

            threading.Thread(target=delayed_update, daemon=True).start()

        threading.Thread(target=sample_thread, daemon=True).start()

    def after_clear_status(self):
        def clear():
            time.sleep(2)
            self.status_label.config(text="")
        threading.Thread(target=clear, daemon=True).start()

    def train_model(self):
        self.model = mod_gesture.train_model()
        if self.model:
            messagebox.showinfo("Model Eğitildi", "✅ Gesture modeli başarıyla eğitildi.")
            mod_gesture.start_live_prediction(self.model, self.pred_label, self.send_to_arduino_if_enabled)

    def test_model(self):
        if not self.model:
            messagebox.showwarning("Uyarı", "Model eğitilmedi. Lütfen önce eğitin.")
            return

        X, y_true = [], []
        for fname in os.listdir(mod_gesture.POZ_DIR):
            if fname.endswith(".csv"):
                label = os.path.splitext(fname)[0]
                with open(os.path.join(mod_gesture.POZ_DIR, fname)) as f:
                    for row in csv.reader(f):
                        if row:
                            X.append([float(r) for r in row])
                            y_true.append(label)

        if not X:
            print("⚠️ Test için veri bulunamadı.")
            return

        y_pred = self.model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"📊 Test doğruluğu: {acc * 100:.2f}%")
        messagebox.showinfo("Model Testi", f"📊 Test doğruluğu: {acc * 100:.2f}%")

    def update_from_landmarks(self, landmarks):
        mod_gesture.set_current_landmarks(landmarks)

    def update_pose_list(self):
        self.pose_listbox.delete(0, tk.END)
        poses = mod_gesture.get_all_poses()
        for name, count in poses.items():
            self.pose_listbox.insert(tk.END, f"{name} ({count} örnek)")

    def copy_selected_pose(self, event):
        selection = self.pose_listbox.curselection()
        if selection:
            pose_name = self.pose_listbox.get(selection[0]).split(" ")[0]
            self.new_label.set(pose_name)
            print(f"📋 '{pose_name}' ismi kopyalandı.")

    def delete_selected_pose(self):
        selection = self.pose_listbox.curselection()
        if selection:
            name = self.pose_listbox.get(selection[0]).split(" ")[0]
            confirm = messagebox.askyesno("Emin misiniz?", f"{name} pozunu silmek istediğinize emin misiniz?")
            if confirm:
                mod_gesture.delete_pose(name)
                self.update_pose_list()

    def toggle_arduino(self):
        self.arduino_sending = not self.arduino_sending
        if self.arduino_sending:
            self.toggle_btn.config(text="⏹️ Gönderimi Durdur")
            if self.arduino is None:
                self.arduino = ArduinoComm()
        else:
            self.toggle_btn.config(text="🔌 Arduino Gönderimini Başlat")
            if self.arduino:
                self.arduino.close()
                self.arduino = None

    def send_to_arduino_if_enabled(self, gesture_name):
        if self.arduino_sending and self.arduino:
            self.arduino.send_gesture(gesture_name)

    def exit_and_save(self):
        if self.model:
            print("💾 Gesture modeli kaydedildi.")
        if self.arduino:
            self.arduino.close()
        self.frame.destroy()
        self.return_callback()
