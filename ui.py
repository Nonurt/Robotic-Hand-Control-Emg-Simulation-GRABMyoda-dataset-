import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
from threading import Thread

from video import VideoProcessor
from modules.mod_finger_percentage_ui import FingerPercentageUI
from modules.mod_gesture_ui import GestureUI
from modules.mod_gesture_emg_ui import EMGGestureUI  # Yeni EMG UI

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("El Modu Seçici")
        self.root.geometry("820x520")

        self.video = VideoProcessor()
        self.current_mode = None

        self.mode_var = tk.StringVar(value="Finger Percentage")
        self.left_panel = tk.Frame(root, width=300, height=500)
        self.left_panel.pack_propagate(False)
        self.left_panel.pack(side="left", fill="y")

        self.video_label = ttk.Label(root)
        self.video_label.place(x=400, y=20, width=380, height=380)

        self.build_main_ui()

        self.running = True
        self.video_thread = Thread(target=self.update_frame, daemon=True)
        self.video_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def build_main_ui(self):
        for widget in self.left_panel.winfo_children():
            widget.destroy()
        self.video.draw_triangles = False

        ttk.Label(self.left_panel, text="Mod Seç:", font=("Arial", 14)).pack(pady=10)
        modes = ["Finger Percentage", "Gesture Classification", "EMG Gesture Detection"]
        for mode in modes:
            ttk.Radiobutton(self.left_panel, text=mode, variable=self.mode_var, value=mode).pack(anchor="w", padx=20)
        ttk.Button(self.left_panel, text="Başlat", command=self.run_selected_mode).pack(pady=10)

        # Görüntü Ayarları
        ttk.Separator(self.left_panel).pack(pady=5, fill="x")
        ttk.Label(self.left_panel, text="Görüntü İyileştirme", font=("Arial", 11)).pack(pady=(10, 0))

        self.gamma_var = tk.BooleanVar(value=self.video.auto_gamma)
        self.hist_var = tk.BooleanVar(value=self.video.equalize_hist)

        ttk.Checkbutton(self.left_panel, text="Otomatik Gamma", variable=self.gamma_var,
                        command=self.toggle_gamma).pack(anchor="w", padx=20)
        ttk.Checkbutton(self.left_panel, text="Histogram Eşitle", variable=self.hist_var,
                        command=self.toggle_hist_eq).pack(anchor="w", padx=20)

    def reload_main_ui(self):
        self.current_mode = None
        self.build_main_ui()

    def run_selected_mode(self):
        selected = self.mode_var.get()
        if selected == "Finger Percentage":
            self.video.draw_triangles = True
            self.video.show_bbox = False
            self.current_mode = FingerPercentageUI(self.left_panel, return_callback=self.reload_main_ui)

        elif selected == "Gesture Classification":
            self.video.draw_triangles = False
            self.video.show_bbox = True
            self.current_mode = GestureUI(self.left_panel, return_callback=self.reload_main_ui)

        elif selected == "EMG Gesture Detection":
            self.video.draw_triangles = False
            self.video.show_bbox = False
            self.current_mode = EMGGestureUI(self.left_panel, return_callback=self.reload_main_ui)

    def toggle_gamma(self):
        self.video.auto_gamma = self.gamma_var.get()

    def toggle_hist_eq(self):
        self.video.equalize_hist = self.hist_var.get()

    def update_frame(self):
        while self.running:
            frame, landmarks = self.video.get_frame()
            if frame is not None:
                if self.current_mode and hasattr(self.current_mode, 'update_from_landmarks') and landmarks:
                    self.current_mode.update_from_landmarks(landmarks)

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            time.sleep(0.03)

    def on_closing(self):
        self.running = False
        time.sleep(0.1)
        if self.current_mode and hasattr(self.current_mode, "exit_and_save"):
            self.current_mode.exit_and_save()
        self.video.release()
        self.root.destroy()

# Ana fonksiyon
def start_ui():
    root = tk.Tk()
    app = App(root)
    root.mainloop()
