import sys
import time
import re
import numpy as np
import math

import socket

from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import serial
import serial.tools.list_ports

from pyqtgraph import LegendItem
from qfluentwidgets import LineEdit, PushButton, TextEdit, CheckBox
from scipy.signal import butter, filtfilt

# ~~~ Existing Config ~~~
data_len = 750

class TCPThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(str)

    def __init__(self, esp_ip="172.20.10.3", port=8080):  # 为ESP32的IP和端口
        super().__init__()
        self.esp_ip = esp_ip
        self.port = port
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)  # 设置超时
        try:
            self.sock.connect((self.esp_ip, self.port))
            print("\u2705 connected to ESP32")
        except Exception as e:
            print(f"❌ 无法连接 ESP32: {e}")
            self.running = False

        # 初始化数据缓冲区
        self.data_pool = [deque(maxlen=data_len) for _ in range(9)]

    def run(self):
        buffer = ""
        while self.running:
            try:
                chunk = self.sock.recv(1024).decode(errors='ignore')
                if not chunk:
                    continue
                buffer += chunk
                lines = buffer.split('\n')
                buffer = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    self.data_received.emit(line)
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

    def send_data(self, data):  # 保留和 SerialThread 一致的接口
        try:
            self.sock.sendall((data + '\n').encode('utf-8'))
        except:
            pass

    def get_latest_data(self, size):  # 和之前一样，返回最新的行
        available = len(self.data_pool[0])
        if available == 0:
            return None
        size = min(size, available)
        return np.array([list(self.data_pool[i])[-size:] for i in range(9)])

# ~~~ New: Head Plot Widget ~~~
class HeadPlotWidget(QtWidgets.QWidget):
    """
    A head-drawing widget that places electrodes based on normalized (nx, ny),
    with (0.5, 0.5) as the center of the circle. 
    The circle and electrodes scale properly on window resize.
    
    Color thresholds:
      amplitude > 0.45           => Red
      0.3 < amplitude <= 0.45    => Orange
      amplitude <= 0.3           => Green
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Define (nx, ny) for each electrode in [0..1], 
        # with (0.5, 0.5) = circle center. 
        # e.g., (0.5, 0.2) => near top-center, (0.5, 0.8) => near bottom-center
        self.electrode_positions = [
            (0.39, 0.17),  # #1
            (0.61, 0.17),  # #2
            (0.34, 0.50),  # #3
            (0.66, 0.50),  # #4
            (0.21, 0.70),  # #5
            (0.79, 0.70),  # #6
            (0.38, 0.85),  # #7
            (0.62, 0.85),  # #8
        ]
        self.amplitudes = [0.0]*8  # amplitude array for each electrode

    def update_amplitudes(self, amps):
        """
        amps: list of 8 amplitude values (floats).
        """
        self.amplitudes = amps
        self.update()  # request a repaint

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        # 1) Draw the head circle
        center_x = w // 2
        center_y = h // 2
        head_radius = int(min(w, h) * 0.45)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPoint(center_x, center_y), head_radius, head_radius)

        # 2) For each electrode, map normalized coords -> circle coords
        electrode_radius = 20
        for i, (nx, ny) in enumerate(self.electrode_positions):
            # Translate from (0..1) with (0.5,0.5) as center => widget coords
            cx = center_x + int((nx - 0.5) * 2 * head_radius)
            cy = center_y + int((ny - 0.5) * 2 * head_radius)

            # Pick color based on amplitude threshold
            amp = abs(self.amplitudes[i])
            if amp > 0.45:
                color = QtGui.QColor("red")
            elif amp > 0.3:
                color = QtGui.QColor("orange")
            else:
                color = QtGui.QColor("green")

            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 2))
            painter.drawEllipse(QtCore.QPoint(cx, cy), electrode_radius, electrode_radius)

            # Label each electrode with channel number
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
            text = f"{i+1}"
            painter.drawText(cx - 5, cy + 5, text)

        # (Optional) draw "R" in the center
        painter.setPen(QtGui.QPen(QtCore.Qt.darkGray, 2))
        painter.drawText(center_x - 5, center_y + 5, "R")

        painter.end()





class ADCPlotter(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("8-Channel EEG Data Viewer + FFT + Head Plot")
        self.setGeometry(100, 100, 1200, 800)

        com_port = self.detect_com_port()
        if not com_port:
            raise Exception("No available COM ports.")
        print(f"Connected to: {com_port}")

        # ~~~ Main Layout ~~~
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # Left Vertical Layout (Time-Domain Plot + FFT Plot)
        left_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(left_layout, stretch=3)

        # ~~~ Time-Domain Plot ~~~
        self.plot_widget = pg.PlotWidget()
        left_layout.addWidget(self.plot_widget)
        self.legend = LegendItem(offset=(30, -30))
        self.legend.setParentItem(self.plot_widget.graphicsItem())

        # We have 8 channels (ignoring original CH1)
        self.colors = ['g','b','c','m','y','w','grey','orange']
        self.plot_data = []
        for i in range(8):
            plot_item = self.plot_widget.plot(pen=pg.mkPen(color=self.colors[i], width=2))
            self.plot_data.append(plot_item)
            self.legend.addItem(plot_item, f"CH{i+1}")

        # ~~~ FFT Plot ~~~
        self.fft_plot_widget = pg.PlotWidget()
        left_layout.addWidget(self.fft_plot_widget)
        self.fft_plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot_widget.setLabel('left', 'Amplitude')
        self.fft_plot_data = []
        for i in range(8):
            fft_item = self.fft_plot_widget.plot(pen=pg.mkPen(color=self.colors[i], width=1))
            self.fft_plot_data.append(fft_item)

        # ~~~ Right Vertical Layout (Head Plot + Controls + Text Display) ~~~
        right_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(right_layout, stretch=2)

        # Head plot
        self.head_plot = HeadPlotWidget()
        self.head_plot.setMinimumSize(300,300)
        right_layout.addWidget(self.head_plot)

        # Raw data text box
        self.text_box = TextEdit()
        self.text_box.setReadOnly(True)
        right_layout.addWidget(self.text_box)

        # Control buttons
        control_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(control_layout)

        self.send_text = LineEdit()
        self.send_button = PushButton("Send")
        self.send_button.clicked.connect(self.send_data)
        self.adc_button = PushButton("Normal")
        self.adc_button.clicked.connect(self.mod_switch)
        control_layout.addWidget(self.adc_button)
        control_layout.addWidget(self.send_text)
        control_layout.addWidget(self.send_button)

        # Checkboxes for 8 channels
        self.checkbox_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(self.checkbox_layout)
        self.checkboxes = []
        for i in range(8):
            cb = CheckBox(f"CH{i+1}")
            cb.setChecked(True)
            self.checkboxes.append(cb)
            self.checkbox_layout.addWidget(cb)

        # ~~~ 使用 WiFi 方式连接 ESP32 ~~~
        self.serial_thread = TCPThread(esp_ip="172.20.10.3", port=8080)  # ✅ 替换为你的 ESP32 IP 地址
        self.serial_thread.data_received.connect(self.handle_serial_data)
        self.serial_thread.start()

        # ~~~ Timer ~~~
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # update every 100 ms

        self.beg = time.time()

    def detect_com_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if 'Bluetooth' not in p.description and 'Virtual' not in p.description:
                return p.device
        return None

    def send_data(self):
        data = self.send_text.text()
        self.serial_thread.send_data(data)

    def mod_switch(self):
        if self.adc_button.text() == "Normal":
            self.serial_thread.send_data("3")
            self.adc_button.setText("Test")
        else:
            self.serial_thread.send_data("1")
            self.adc_button.setText("Normal")

    def handle_serial_data(self, raw_data):
        # Append last 10 lines to text box
        current_text = self.text_box.toPlainText()
        lines = current_text.split('\n')
        lines.append(raw_data)
        if len(lines) > 10:
            lines = lines[-10:]
        self.text_box.setPlainText('\n'.join(lines))

    def closeEvent(self, event):
        self.serial_thread.stop()
        self.serial_thread.wait()
        event.accept()

    # Optional low-pass filter
    def apply_lowpass_filter(self, sig, cutoff=50, fs=500, order=5):
        if len(sig) < 19:
            return sig
        nyq = 0.5 * fs
        norm_cut = cutoff / nyq
        b, a = butter(order, norm_cut, btype='low', analog=False)
        return filtfilt(b, a, sig)

    def update_plot(self):
        latest_data = self.serial_thread.get_latest_data(data_len)
        if latest_data is None:
            return

        # We skip original CH1 => reindexed_data has shape (8, data_len)
        reindexed_data = latest_data[1:9]

        # ~~~ 1) Update Time-Domain Plot ~~~
        for i in range(8):
            if self.checkboxes[i].isChecked():
                wave = reindexed_data[i]
                filtered = self.apply_lowpass_filter(wave)
                self.plot_data[i].setData(filtered)
            else:
                self.plot_data[i].setData([])

        # Adjust Y-range
        selected_data = []
        for i in range(8):
            if self.checkboxes[i].isChecked():
                selected_data.append(reindexed_data[i])
        if selected_data:
            global_min = np.min([d.min() for d in selected_data])
            global_max = np.max([d.max() for d in selected_data])
            self.plot_widget.setYRange(global_min, global_max)
        else:
            self.plot_widget.setYRange(-4.5, 4.5)

        # ~~~ 2) Update FFT Plot for each channel ~~~
        # We'll use the last 256 samples for a quick FFT
        fft_size = 256
        fs = 500.0  # sampling freq (adjust if different)
        for i in range(8):
            if self.checkboxes[i].isChecked():
                wave = reindexed_data[i]
                wave = wave[-fft_size:] if len(wave) > fft_size else wave
                if len(wave) > 2:
                    # Simple FFT
                    freqs = np.fft.rfftfreq(len(wave), 1.0/fs)
                    fft_vals = np.fft.rfft(wave)
                    amps = np.abs(fft_vals) / len(wave)
                    self.fft_plot_data[i].setData(freqs, amps)
                else:
                    self.fft_plot_data[i].setData([], [])
            else:
                self.fft_plot_data[i].setData([], [])

        # ~~~ 3) Update Head Plot Colors ~~~
        # We use the absolute amplitude of the most recent sample
        # You can also use mean amplitude or other metric
        head_amplitudes = []
        for i in range(8):
            wave = reindexed_data[i]
            amp = 0.0
            if len(wave) > 0:
                amp = abs(wave[-1])
            head_amplitudes.append(amp)

        # Update the custom HeadPlotWidget
        self.head_plot.update_amplitudes(head_amplitudes)

        # ~~~ (Optional) do something every second ~~~
        if time.time() - self.beg > 1:
            self.beg = time.time()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = ADCPlotter()
    main.show()
    sys.exit(app.exec_())