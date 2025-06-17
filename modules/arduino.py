import serial
import serial.tools.list_ports
import threading
import time

class ArduinoComm:
    def __init__(self, baudrate=9600, reconnect_interval=2):
        self.baudrate = baudrate
        self.reconnect_interval = reconnect_interval
        self.serial_conn = None
        self.running = True
        self.port = "COM8"  # Burada COM9 sabit port olarak girildi
        self.connect_thread = threading.Thread(target=self.try_connect_loop, daemon=True)
        self.connect_thread.start()

    def try_connect_loop(self):
        # Port zaten belirli: COM9
        while self.running and self.serial_conn is None:
            try:
                ser = serial.Serial(self.port, self.baudrate, timeout=1)
                time.sleep(2)  # Arduino resetlenmesi için bekle
                ser.write(b'HELLO\n')
                ser.flush()
                print(f"🔌 Arduino bulundu: {self.port}")
                self.serial_conn = ser
                return
            except Exception as e:
                print(f"Bağlanılamadı {self.port}: {e}")
                print("🔄 Bağlantı tekrar deneniyor...")
                time.sleep(self.reconnect_interval)

    def send_raw(self, message: str):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write((message + "\n").encode())

    def send_percentages(self, angle_list):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                message = ",".join(f"{i}:{angle}" for i, angle in enumerate(angle_list))
                self.send_raw(message)
            except Exception as e:
                print("❌ Arduino yazma hatası:", e)
                self.serial_conn = None

    def close(self):
        self.running = False
        if self.serial_conn:
            try:
                self.serial_conn.close()
                print("🛑 Arduino bağlantısı kapatıldı.")
            except:
                pass
            self.serial_conn = None
