# BCI ESP32 TCP Communication & EEG Visualization

This repository contains the source code for TCP communication between ESP32 and PC, as well as Python-based real-time EEG data visualization.

## Contents
- `ESP32_TCP.ino` : ESP32 code for TCP server and ADS1299 SPI communication.
- `wifi_plot_only.py` : Python script for real-time EEG plotting.
- `wifi_plot_filter.py` : Python script with filtering and visualization features.

## How to Use
1. Upload `ESP32_TCP.ino` to your ESP32 device.
2. Run the Python scripts on your PC to visualize incoming EEG data via Wi-Fi.

## Requirements
- Python 3.9.13
- Libraries: `pyqt5`, `pyqtgraph`, `socket`, `numpy`