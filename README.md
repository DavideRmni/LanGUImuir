# LanGUImuir v2.3 🌊

<img width="697" height="480" alt="Screenshot 2026-01-07 173205" src="https://github.com/user-attachments/assets/d5ce167f-86a9-48b2-bb56-ba4d6afd5e56" />

<img width="697" height="480" alt="Screenshot 2026-01-07 173224" src="https://github.com/user-attachments/assets/18a5d5f7-142f-49b0-a612-1538bdd471fd" />

**LanGUImuir** is a comprehensive Python Control Interface for **Langmuir-Blodgett Troughs**. It provides real-time control over barriers (stepper motors), reads sensor data (Wilhelmy plate balance), visualizes isotherms, and logs experimental data.

> **🆕 New in v2.3:** Enhanced support for **Calibrated Surface Tension**. The software now distinguishes between raw sensor weight (command `p`) and firmware-calibrated surface tension (command `P`), plotting them simultaneously on dual axes.

## ✨ Key Features

* **Dual Measurement Modes:**
    * 🔵 **Raw Weight:** Logs raw sensor data (mg).
    * 🟢 **Calibrated Surface Tension:** Logs firmware-calculated tension (mN/m).
* **Real-Time Visualization:** Live plotting of Isotherms (Pressure vs Area) and Stability (Pressure/Weight vs Time).
* **Hardware Control:** Manual and automated control of stepper motors (Up/Down/Left/Right) via Serial/Arduino.
* **Scripting Engine:** Automate complex experiment protocols using simple text files.
* **Data Logging:** Automatic CSV generation with timestamps and extended metadata.
* **Debug Console:** Built-in terminal for monitoring raw serial TX/RX traffic.

## 🛠️ Installation & Requirements

### Prerequisites
* Python 3.7+
* Windows OS (Required for `winsound` audio notifications; code modification needed for Linux/Mac).

### Dependencies
Install the required libraries using pip:

pip install matplotlib pyserial numpy

(Note: tkinter is usually included with Python. If missing, install python-tk).

🚀 Usage

1. Connect Hardware: Plug in your Arduino/Langmuir controller via USB.

2. Run the Software:

Bash

python LanGUImuir_latest.py
3. Connection:

* Select the correct COM Port from the dropdown in the top-left corner.
* Click Connect.

4. Manual Control:

Use the directional buttons to move barriers.

Measure (p): Takes a raw weight reading.

Measure Cal (P): Takes a calibrated surface tension reading (New in v2.3).

📄 Scripting Automation
LanGUImuir supports automation via .txt scripts. Load them via File > Load Script.

Syntax: [Command][Value]

p	Read Raw Weight (N times)	p10 (Read raw weight 10 times)
P	Read Calibrated Tension (N times)	P5 (Read surface tension 5 times)
w	Move Barrier UP (steps)	w100
s	Move Barrier DOWN (steps)	s100
a	Move Barrier LEFT (steps)	a50
d	Move Barrier RIGHT (steps)	d50

Note: The software waits for a "Done" response from the hardware before proceeding to the next line in the script.

📊 Output Data (CSV)
Files are automatically saved in the /log folder. Format: langmuir_data_YYYYMMDD_HHMMSS.csv

🖥️ Interface Overview
(It is recommended to add a screenshot of the running application here)

1. Left Panel: Connection settings, logging status, manual controls, and script execution.

2. Right Panel: Top Plot: Time vs. Weight (Blue) AND Time vs. Calibrated Tension (Green).

3. Bottom Plot: Isotherm (Area vs. Pressure).

Debug Console (Ctrl+Shift+D): View raw serial data stream.

⚠️ Compatibility Note
This software uses import winsound for audio alerts, which is specific to Windows.

Linux/Mac Users: Please comment out lines referring to winsound or AudioNotifier to prevent crashes.

🤝 Contributing
Contributions are welcome! Please ensure any Pull Request maintains backward compatibility with the existing serial protocol.
