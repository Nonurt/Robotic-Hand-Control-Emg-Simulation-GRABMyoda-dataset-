import os
import random
import numpy as np
from scipy.io import loadmat

# Sabitler
DATA_PATH = "E:/emg_data/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.1.0/Output BM"
FORARM_CHANNELS = 16  # Dokümantasyona göre
WRIST_CHANNELS = 12  # Dokümantasyona göre
MAX_ATTEMPTS = 20  # Maksimum deneme sayısı

GESTURE_TO_INDEX = {
    'Lateral Prehension': 0,
    'Thumb Adduction': 1,
    'Thumb and Little Finger Opposition': 2,
    'Thumb and Index Finger Opposition': 3,
    'Thumb and Index Finger Extension': 4,
    'Thumb and Little Finger Extension': 5,
    'Index and Middle Finger Extension': 6,
    'Little Finger Extension': 7,
    'Index Finger Extension': 8,
    'Thumb Finger Extension': 9,
    'Wrist Extension': 10,
    'Wrist Flexion': 11,
    'Forearm Supination': 12,
    'Forearm Pronation': 13,
    'open_hand_right': 14,
    'Hand Close': 15
}

def validate_data(forearm, wrist):
    """Veri yapısını kontrol et"""
    if forearm is None or wrist is None:
        return False, "Boş veri"

    if forearm.ndim != 2 or wrist.ndim != 2:
        return False, "2 boyutlu olmalı"

    if forearm.shape[1] != FORARM_CHANNELS:
        return False, f"Forearm {FORARM_CHANNELS} kanal olmalı"

    if wrist.shape[1] != WRIST_CHANNELS:
        return False, f"Wrist {WRIST_CHANNELS} kanal olmalı"

    if forearm.shape[0] < 100 or wrist.shape[0] < 100:
        return False, "Yeterli örnek yok"

    return True, "Geçerli"


def load_random_emg(gesture_name, min_session=1, max_session=3, max_subject=43, max_trial=6):
    """Rastgele EMG verisi yükle"""
    if gesture_name not in GESTURE_TO_INDEX:
        return None, None, "Tanımsız hareket"

    gesture_idx = GESTURE_TO_INDEX[gesture_name]

    for attempt in range(MAX_ATTEMPTS):
        try:
            # Rastgele seçim
            session = random.randint(min_session, max_session)
            subject = random.randint(1, max_subject)
            trial = random.randint(0, max_trial)

            # Dosya yolu oluştur (farklı formatları deneyelim)
            path_variants = [
                f"{DATA_PATH}/Session{session}_converted/session{session}_participant{subject}.mat",
                f"{DATA_PATH}/Session{session}/session{session}_participant{subject}.mat",
                f"{DATA_PATH}/session{session}_participant{subject}.mat"
            ]

            for path in path_variants:
                if os.path.exists(path):
                    data = loadmat(path)
                    forearm = data["DATA_FOREARM"][trial, gesture_idx]
                    wrist = data["DATA_WRIST"][trial, gesture_idx]

                    # Validasyon
                    is_valid, msg = validate_data(forearm, wrist)
                    if is_valid:
                        return forearm, wrist, None
                    else:
                        print(f"Geçersiz veri: {msg}")
                        break

        except Exception as e:
            print(f"Deneme {attempt + 1} hata: {str(e)}")
            continue

    return None, None, f"{MAX_ATTEMPTS} denemede uygun veri bulunamadı"


def load_random_emg_by_index(index, **kwargs):
    """Index'e göre veri yükleme"""
    gesture_name = next((k for k, v in GESTURE_TO_INDEX.items() if v == index), None)
    if gesture_name:
        return load_random_emg(gesture_name, **kwargs)
    return None, None, "Geçersiz index"