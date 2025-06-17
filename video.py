import cv2
import numpy as np
from utils.mediapipe import HandDetector
from modules.mod_finger_percentage import FingerPercentageEstimator
from modules import mod_gesture  # ✳️ El kutusu için

class VideoProcessor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.gamma = 1.0
        self.auto_gamma = False
        self.equalize_hist = False
        self.hand_detector = HandDetector()
        self.draw_triangles = False
        self.show_bbox = False  # ✅ Yeni: sadece kutu çizimi kontrolü
        self.estimator = FingerPercentageEstimator()

    def adjust_gamma(self, image, gamma_value):
        inv_gamma = 1.0 / gamma_value
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def auto_gamma_correction(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)

        if mean_intensity < 100:
            gamma = 1.8
        elif mean_intensity > 170:
            gamma = 0.6
        else:
            gamma = 1.2

        return self.adjust_gamma(image, gamma)

    def auto_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.resize(frame, (380, 380))

        if self.auto_gamma:
            frame = self.auto_gamma_correction(frame)
        else:
            frame = self.adjust_gamma(frame, self.gamma)

        if self.equalize_hist:
            frame = self.auto_contrast(frame)

        frame, landmarks = self.hand_detector.process_with_landmarks(frame)

        if landmarks:
            mod_gesture.set_current_landmarks(landmarks)
            mod_gesture.set_current_frame(frame.copy())

            if self.draw_triangles:
                for i1, i2, i3 in self.estimator.finger_points.values():
                    pts = np.array([landmarks[i1], landmarks[i2], landmarks[i3]], np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

            if self.show_bbox:
                x1, y1, x2, y2 = mod_gesture.get_bounding_box(landmarks)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks

    def release(self):
        self.cap.release()
