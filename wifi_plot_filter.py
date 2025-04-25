import sys
import time
import re
import csv
import socket
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import serial
import serial.tools.list_ports
from scipy.signal import butter, filtfilt

# -------------------- Filter Function --------------------
def apply_bandpass_filter(sig, lowcut, highcut, fs=500, order=5):
    """
    Apply a Butterworth filter to the 1D signal.
    If lowcut <= 0, a low-pass filter with cutoff=highcut is used.
    """
    if len(sig) < (order * 3):
        return sig
    nyq = 0.5 * fs
    if lowcut <= 0:
        norm_cut = highcut / nyq
        b, a = butter(order, norm_cut, btype='low', analog=False)
        return filtfilt(b, a, sig)
    else:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)
        return filtfilt(b, a, sig)

# -------------------- SerialThread: Reads EEG data from serial port --------------------

class TCPThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(str)

    def __init__(self, esp_ip="172.20.10.3", port=8080):  # 🔹 替换为你的 ESP32 IP
        super().__init__()
        self.esp_ip = esp_ip
        self.port = port
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)  # 设置超时，防止无响应卡死
        try:
            self.sock.connect((self.esp_ip, self.port))
            print("✅ connected to ESP32")
        except Exception as e:
            print(f"❌  ESP32: {e}")
            self.running = False

        # 初始化数据缓冲区
        self.data_pool = [deque(maxlen=750) for _ in range(9)]


    def run(self):
        buffer = ""  # 用于拼接不完整的 TCP 数据
        while self.running:
            try:
                chunk = self.sock.recv(1024).decode(errors='ignore')  # 忽略解码错误
                if not chunk:
                    continue
                buffer += chunk

                # 按换行分割多条数据
                lines = buffer.split('\n')
                buffer = lines[-1]  # 最后一行可能是不完整的，留给下次处理

                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    self.data_received.emit(line)

                    # 匹配 9 个浮点数，用 , 分割
                    match = re.match(r'^([\d\.\-eE]+,){8}[\d\.\-eE]+$', line)
                    if match:
                        try:
                            values = [float(x) for x in line.split(',')]
                            for i in range(9):
                                self.data_pool[i].append(values[i])
                        except ValueError:
                            print(f"⚠️ 转换失败: {line}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ TCP 接收错误: {e}")
                break

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except:
            pass

# -------------------- InstructionWidget: White background with red fixation cross and instruction text --------------------
class InstructionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.instruction = ""  # e.g., "Rest", "Left", "Right"
        self.setStyleSheet("background-color: white;")
    
    def setInstruction(self, text):
        self.instruction = text
        self.update()
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Fill background white
        painter.fillRect(self.rect(), QtCore.Qt.white)
        # Draw red fixation cross at center
        center = self.rect().center()
        cross_size = 40
        pen = QtGui.QPen(QtCore.Qt.red, 4)  # Red cross
        painter.setPen(pen)
        painter.drawLine(center.x() - cross_size, center.y(), center.x() + cross_size, center.y())
        painter.drawLine(center.x(), center.y() - cross_size, center.x(), center.y() + cross_size)
        # Draw instruction text if provided (centered in black)
        if self.instruction:
            font = QtGui.QFont("Arial", 48)
            painter.setFont(font)
            painter.setPen(QtCore.Qt.black)
            metrics = QtGui.QFontMetrics(font)
            text_width = metrics.horizontalAdvance(self.instruction)
            text_height = metrics.height()
            x = center.x() - text_width / 2
            y = center.y() + text_height / 4
            painter.drawText(x, y, self.instruction)
        painter.end()

# -------------------- DataAcquisitionWindow: Main window controlling acquisition sequence and CSV saving --------------------
class DataAcquisitionWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Data Acquisition")
        # Set window dimensions to 2400 x 1600 (double original)
        self.setGeometry(100, 100, 2400, 1600)
        
        # Create central widget and main vertical layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # ---------------- User Control Panel ----------------
        control_layout = QtWidgets.QHBoxLayout()
        # Number of rounds input
        control_layout.addWidget(QtWidgets.QLabel("Rounds:"))
        self.rounds_input = QtWidgets.QLineEdit("1")
        control_layout.addWidget(self.rounds_input)
        # Rest time (in seconds)
        control_layout.addWidget(QtWidgets.QLabel("Rest (sec):"))
        self.rest_input = QtWidgets.QLineEdit("1")
        control_layout.addWidget(self.rest_input)
        # Left phase time (in seconds)
        control_layout.addWidget(QtWidgets.QLabel("Left (sec):"))
        self.left_input = QtWidgets.QLineEdit("3")
        control_layout.addWidget(self.left_input)
        # Right phase time (in seconds)
        control_layout.addWidget(QtWidgets.QLabel("Right (sec):"))
        self.right_input = QtWidgets.QLineEdit("3")
        control_layout.addWidget(self.right_input)
        main_layout.addLayout(control_layout)
        
        # ---------------- InstructionWidget for fixation cross and instructions ----------------
        self.instruction_widget = InstructionWidget()
        self.instruction_widget.setFixedSize(800, 600)
        main_layout.addWidget(self.instruction_widget, alignment=QtCore.Qt.AlignCenter)
        
        # ---------------- Start Button ----------------
        self.start_button = QtWidgets.QPushButton("Start Data Acquisition")
        main_layout.addWidget(self.start_button, alignment=QtCore.Qt.AlignCenter)
        self.start_button.clicked.connect(self.startAcquisition)
        
        # Data acquisition variables
        self.recording = False
        self.current_class = None  # "Left" or "Right"
        # Each row: [Timestamp, EEG_1, ..., EEG_8, Class]
        self.acquired_data = []
        self.rounds_remaining = 0
        
        # Duration variables (in milliseconds)
        self.rest_time = 1000
        self.left_time = 3000
        self.right_time = 3000
        
        # Acquisition start timestamp (for relative time)
        self.acquisition_start_time = None
        
        # Filter parameters for offline filtering (in Hz)
        self.lowcut = 0.5
        self.highcut = 30.0
        
        # Initialize SerialThread to read EEG data from COM port
        # 🔹 使用 TCP 替代串口
        self.serial_thread = TCPThread(esp_ip="172.20.10.3", port=8080)  # 改为你的 ESP32 IP
        self.serial_thread.data_received.connect(self.handle_serial_data)
        self.serial_thread.start()


    def detect_com_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'Bluetooth' not in p.description and 'Virtual' not in p.description:
                return p.device
        raise Exception("No available COM ports.")

    def startAcquisition(self):
        try:
            self.rounds_remaining = int(self.rounds_input.text())
        except ValueError:
            self.rounds_remaining = 1
        try:
            self.rest_time = int(float(self.rest_input.text()) * 1000)
        except ValueError:
            self.rest_time = 1000
        try:
            self.left_time = int(float(self.left_input.text()) * 1000)
        except ValueError:
            self.left_time = 3000
        try:
            self.right_time = int(float(self.right_input.text()) * 1000)
        except ValueError:
            self.right_time = 3000

        self.acquired_data = []
        self.start_button.setDisabled(True)
        # Set acquisition start time (relative time = 0)
        self.acquisition_start_time = time.time()
        self.runRound()

    def runRound(self):
        if self.rounds_remaining <= 0:
            self.instruction_widget.setInstruction("Collection finish")
            self.saveCSV()
            self.start_button.setEnabled(True)
            return
        # Phase 1: "Rest" for the specified rest time (no recording)
        self.instruction_widget.setInstruction("Rest")
        QtCore.QTimer.singleShot(self.rest_time, self.showCrossBeforeLeft)
    
    def showCrossBeforeLeft(self):
        # Show cross only (no text) for 1 second
        self.instruction_widget.setInstruction("")
        QtCore.QTimer.singleShot(1000, self.phaseLeft)
    
    def phaseLeft(self):
        # Phase 2: "Left" for the specified left time; start recording after 300ms
        self.instruction_widget.setInstruction("Left")
        self.current_class = "Left"
        QtCore.QTimer.singleShot(300, self.startRecording)
        QtCore.QTimer.singleShot(self.left_time, self.stopRecordingAndRestAfterLeft)
    
    def stopRecordingAndRestAfterLeft(self):
        self.stopRecording()
        # Phase 3: "Rest" for the specified rest time (no recording)
        self.instruction_widget.setInstruction("Rest")
        QtCore.QTimer.singleShot(self.rest_time, self.showCrossBeforeRight)
    
    def showCrossBeforeRight(self):
        # Show cross only (no text) for 1 second
        self.instruction_widget.setInstruction("")
        QtCore.QTimer.singleShot(1000, self.phaseRight)
    
    def phaseRight(self):
        # Phase 4: "Right" for the specified right time; start recording after 300ms
        self.instruction_widget.setInstruction("Right")
        self.current_class = "Right"
        QtCore.QTimer.singleShot(300, self.startRecording)
        QtCore.QTimer.singleShot(self.right_time, self.stopRecordingAndFinishRound)
    
    def stopRecordingAndFinishRound(self):
        self.stopRecording()
        self.rounds_remaining -= 1
        QtCore.QTimer.singleShot(1000, self.runRound)
    
    def startRecording(self):
        self.recording = True
    
    def stopRecording(self):
        self.recording = False
    
    def handle_serial_data(self, raw_data):
        # Record EEG data only during designated acquisition periods
        if self.recording:
            match = re.match(r'Channel:([\d\.\-]+,){8}[\d\.\-]+', raw_data)
            if match:
                values = [float(x) for x in raw_data.split('Channel:')[1].split(',')]
                # Use channels 2-9 (8 EEG channels)
                eeg_values = values[1:9]
                # Compute timestamp (in seconds) relative to acquisition start
                timestamp = time.time() - self.acquisition_start_time
                # Prepend timestamp to the row, append current class label
                row = [timestamp] + eeg_values + [self.current_class]
                self.acquired_data.append(row)
    
    def saveCSV(self):
        # Save raw captured data to a CSV file (optional)
        raw_filename = "EEG_data_raw.csv"
        header = ["Timestamp", "EEG_1", "EEG_2", "EEG_3", "EEG_4", "EEG_5", "EEG_6", "EEG_7", "EEG_8", "Class"]
        with open(raw_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.acquired_data)
        print(f"Raw data saved to {raw_filename}")

        # Now apply filter to the EEG channels and save filtered data.
        data_array = np.array(self.acquired_data)
        if data_array.shape[0] == 0:
            print("No data captured.")
            return
        # Separate columns: first is timestamp, next 8 are EEG channels, last is class.
        timestamps = data_array[:, 0]
        eeg_data = data_array[:, 1:9].astype(float)
        classes = data_array[:, 9]
        
        # Apply filtering channel by channel.
        filtered_eeg = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            filtered_eeg[:, ch] = apply_bandpass_filter(eeg_data[:, ch], self.lowcut, self.highcut, fs=500, order=5)
        
        # Combine filtered EEG data with timestamp and class label.
        filtered_data = np.column_stack((timestamps, filtered_eeg, classes))
        filtered_filename = "EEG_data_filtered.csv"
        with open(filtered_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(filtered_data)
        print(f"Filtered data saved to {filtered_filename}")
    
    def closeEvent(self, event):
        self.serial_thread.stop()
        self.serial_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DataAcquisitionWindow()
    window.show()
    sys.exit(app.exec_())
