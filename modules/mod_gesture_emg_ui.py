import tkinter as tk
from tkinter import ttk
from modules import mod_gesture, mod_gesture_emg
import numpy as np
import time
import threading
import json
import socket
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class EMGGestureUI:
    def __init__(self, parent, return_callback):
        self.parent = parent
        self.return_callback = return_callback
        self.frame = tk.Frame(parent)
        self.frame.pack(fill="both", expand=True)

        ttk.Label(self.frame, text="EMG Gesture Detection Modu", font=("Arial", 14, "bold")).pack(pady=10)

        self.current_pred = "-"
        self.gesture_label = ttk.Label(self.frame, text="Gesture Tahmini: -", font=("Arial", 12))
        self.gesture_label.pack(pady=10)

        self.count_label = ttk.Label(self.frame, text="Örnek sayısı: -", font=("Arial", 10))
        self.count_label.pack(pady=(0, 5))

        fig = plt.figure(figsize=(5, 2.5), dpi=100)
        self.ax1 = fig.add_subplot(211)
        self.ax2 = fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.canvas.get_tk_widget().pack()

        ip_port_frame = ttk.Frame(self.frame)
        ip_port_frame.pack(pady=5)
        ttk.Label(ip_port_frame, text="IP:").grid(row=0, column=0)
        self.ip_entry = ttk.Entry(ip_port_frame, width=15)
        self.ip_entry.insert(0, "127.0.0.1")
        self.ip_entry.grid(row=0, column=1, padx=5)
        ttk.Label(ip_port_frame, text="Port:").grid(row=0, column=2)
        self.port_entry = ttk.Entry(ip_port_frame, width=6)
        self.port_entry.insert(0, "5000")
        self.port_entry.grid(row=0, column=3)

        self.send_socket = False
        self.socket_client = None
        self.toggle_btn = ttk.Button(self.frame, text="Canlı Socket Gönderimini Başlat", command=self.toggle_socket)
        self.toggle_btn.pack(pady=5)

        manual_frame = ttk.Frame(self.frame)
        manual_frame.pack(pady=10)
        ttk.Label(manual_frame, text="Manuel Gesture Index (0-15):").grid(row=0, column=0)
        self.manual_index_entry = ttk.Entry(manual_frame, width=5)
        self.manual_index_entry.grid(row=0, column=1, padx=5)
        ttk.Button(manual_frame, text="Manuel Gönder", command=self.send_manual_emg).grid(row=0, column=2)

        ttk.Button(self.frame, text="Geri Dön", command=self.exit_and_save).pack(pady=10)

        self.model = mod_gesture.load_model()
        self.running = True

        threading.Thread(target=self.predict_loop, daemon=True).start()
        threading.Thread(target=self.emg_update_loop, daemon=True).start()

    def update_from_landmarks(self, landmarks):
        mod_gesture.set_current_landmarks(landmarks)

    def toggle_socket(self):
        self.send_socket = not self.send_socket
        if self.send_socket:
            try:
                ip = self.ip_entry.get()
                port = int(self.port_entry.get())
                self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_client.connect((ip, port))
                self.toggle_btn.config(text="Canlı Socket Gönderimini Durdur")
            except Exception as e:
                print("Socket bağlantı hatası:", e)
                self.send_socket = False
                self.socket_client = None
                self.toggle_btn.config(text="Canlı Socket Gönderimini Başlat")
        else:
            if self.socket_client:
                self.socket_client.close()
            self.socket_client = None
            self.toggle_btn.config(text="Canlı Socket Gönderimini Başlat")

    def predict_loop(self):
        while self.running:
            try:
                if self.model and mod_gesture.current_landmarks:
                    flat = np.array(mod_gesture.current_landmarks).flatten().reshape(1, -1)
                    pred = self.model.predict(flat)[0]
                    self.current_pred = pred
                    self.parent.after(0, lambda: self.gesture_label.config(text=f"Gesture Tahmini: {pred}"))
                time.sleep(0.5)
            except Exception as e:
                print("Tahmin hatası:", e)
                break

    def emg_update_loop(self):
        while self.running:
            gesture = self.current_pred
            forearm, wrist, err = mod_gesture_emg.load_random_emg(gesture)
            if forearm is not None:
                self.parent.after(0, lambda: self.plot_signals(forearm[:1024], wrist[:1024]))
                self.parent.after(0, lambda: self.count_label.config(text=f"Örnek sayısı: {len(forearm)}"))

                if self.send_socket and self.socket_client:
                    try:
                        payload = json.dumps({
                            "forearm": forearm[:512, :].tolist(),
                            "wrist": wrist[:512, :].tolist()
                        }) + "\n"
                        self.socket_client.sendall(payload.encode('utf-8'))
                    except Exception as e:
                        print("Gönderim hatası:", e)
                        self.parent.after(0, self.toggle_socket)
            else:
                self.parent.after(0, lambda: self.plot_message(err))

            time.sleep(1)

    def send_manual_emg(self):
        try:
            index = int(self.manual_index_entry.get())
            if not (0 <= index <= 15):
                raise ValueError("Index out of range")

            forearm, wrist, err = mod_gesture_emg.load_random_emg_by_index(index)
            if forearm is not None:
                self.parent.after(0, lambda: self.plot_signals(forearm[:1024], wrist[:1024]))
                if self.send_socket and self.socket_client:
                    payload = json.dumps({
                        "forearm": forearm[:512, :].tolist(),
                        "wrist": wrist[:512, :].tolist()
                    }) + "\n"
                    self.socket_client.sendall(payload.encode('utf-8'))
            else:
                self.parent.after(0, lambda: self.plot_message(err))
        except Exception as e:
            print("Manuel gönderim hatası:", e)
            self.parent.after(0, lambda: self.plot_message("Manuel gönderim hatası"))

    def plot_signals(self, forearm, wrist):
        self.ax1.clear()
        self.ax1.plot(forearm, linewidth=1)
        self.ax1.set_title("Forearm EMG", fontsize=10)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.ax2.clear()
        self.ax2.plot(wrist, linewidth=1)
        self.ax2.set_title("Wrist EMG", fontsize=10)
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        self.canvas.draw()

    def plot_message(self, msg):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("Forearm EMG", fontsize=10)
        self.ax1.text(0.5, 0.5, msg, fontsize=9, ha='center', va='center')
        self.ax2.set_title("Wrist EMG", fontsize=10)
        self.ax2.text(0.5, 0.5, msg, fontsize=9, ha='center', va='center')
        self.canvas.draw()

    def exit_and_save(self):
        self.running = False
        if self.socket_client:
            self.socket_client.close()
        time.sleep(0.1)
        self.frame.destroy()
        self.return_callback()
