#!/usr/bin/env python3
"""
LanGUImuir - Langmuir Balance Control Interface v2.3
Enhanced with calibrated surface tension support (Command 'P')

Features:
- Support for both 'p' (raw weight) and 'P' (calibrated surface tension)
- Extended CSV logging with SURFACE_TENSION_CALIBRATED column
- Enhanced plots with calibrated data visualization
- All previous v2.2 features maintained
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
import datetime
import re
import os
import csv
import json
import math
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any
import winsound  # For Windows audio notification


# Constants
AVOGADRO_NUMBER = 6.02214076E23
DEFAULT_BAUDRATE = 9600
BUFFER_SIZE = 1000
UPDATE_INTERVAL = 100  # ms
MOVING_AVERAGE_WINDOW = 10

# Physical dimensions (mm)
TROUGH_WIDTH = 110.0
TROUGH_LENGTH_MAX = 240.0
MM_STEP = 0.01057480271

# Script execution constants
DONE_TIMEOUT = 10.0  # seconds (reduced from 30)
COMMAND_DELAY = 1  # seconds (reduced from 0.5)
MESSAGE_BUFFER_SIZE = 1000  # Keep all messages for debug visibility

# Regex patterns for Arduino responses
WEIGHT_PATTERN = r"Average of (\d+) weighings: ([+-]?\d+\.?\d*)"
POSITION_PATTERN = r"Actual position, X axis: (\d+), Y Axis: (\d+)"
DONE_PATTERN = r"Done"

# Get script directory for absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")
LOG_DIR = os.path.join(SCRIPT_DIR, "log")
GRAPHICS_DIR = os.path.join(SCRIPT_DIR, "graphics")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")


class MessageBuffer:
    """Thread-safe buffer for storing and searching messages."""
    
    def __init__(self, max_size: int = MESSAGE_BUFFER_SIZE):
        self.messages = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.done_event = threading.Event()
        self.last_command_index = 0
        self.last_command_type = None  # Track if last command was 'p' or 'P'
        
    def add_message(self, message: str, message_type: str = "rx"):
        """Add a message to the buffer."""
        with self.lock:
            timestamp = datetime.datetime.now()
            msg_entry = {
                "timestamp": timestamp,
                "message": message,
                "type": message_type,
                "index": len(self.messages)
            }
            self.messages.append(msg_entry)
            
            # Check for "Done" pattern
            if message_type == "rx" and re.search(DONE_PATTERN, message, re.IGNORECASE):
                self.done_event.set()
    
    def mark_command_sent(self, command_type: Optional[str] = None):
        """Mark the current position as when a command was sent."""
        with self.lock:
            self.last_command_index = len(self.messages)
            self.last_command_type = command_type
            self.done_event.clear()
    
    def get_last_command_type(self) -> Optional[str]:
        """Get the type of the last command sent ('p' or 'P')."""
        with self.lock:
            return self.last_command_type
    
    def wait_for_done(self, timeout: float = DONE_TIMEOUT) -> bool:
        """Wait for "Done" response with intelligent fallback."""
        # First, try event-based waiting
        if self.done_event.wait(timeout):
            return True
        
        # Fallback: check if any messages were received since last command
        with self.lock:
            messages_since_command = [
                msg for msg in list(self.messages)[self.last_command_index:]
                if msg["type"] == "rx"
            ]
            
            # If we received messages but no "Done", still consider it successful
            # This handles cases where the device responds but doesn't send "Done"
            if messages_since_command:
                return True
            
        return False
    
    def get_recent_messages(self, count: int = 50) -> List[Dict]:
        """Get the most recent messages."""
        with self.lock:
            return list(self.messages)[-count:] if count > 0 else list(self.messages)
    
    def get_all_messages(self) -> List[Dict]:
        """Get all messages for debug display."""
        with self.lock:
            return list(self.messages)
    
    def search_messages_since_command(self, pattern: str) -> List[Dict]:
        """Search for pattern in messages received since last command."""
        with self.lock:
            messages_since_command = [
                msg for msg in list(self.messages)[self.last_command_index:]
                if msg["type"] == "rx"
            ]
            return [msg for msg in messages_since_command if re.search(pattern, msg["message"], re.IGNORECASE)]


class LoggingManager:
    """Handles all logging operations with automatic CSV generation."""
    
    def __init__(self):
        self.current_log_file = None
        self.current_full_log_file = None
        self.csv_writer = None
        self.full_log_writer = None
        self.csv_file_handle = None
        self.full_log_file_handle = None
        self.measurement_counter = 0
        self.logging_enabled = False
        self.full_logging_enabled = False
        
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Start automatic logging at startup
        self.start_new_log_session()
    
    def start_new_log_session(self):
        """Start a new logging session with automatic file creation."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Close existing files
        self.close_current_logs()
        
        # Create new CSV file for measurements
        self.current_log_file = os.path.join(LOG_DIR, f"langmuir_data_{timestamp}.csv")
        self.csv_file_handle = open(self.current_log_file, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file_handle, delimiter=';')
        
        # Write CSV header - MODIFIED to include SURFACE_TENSION_CALIBRATED
        self.csv_writer.writerow([
            "COUNT", "TIMESTAMP", "NUM_WEIGHINGS", "WEIGHT", "SURFACE_TENSION_CALIBRATED",
            "MOVING_AVERAGE", "POSITION_X", "POSITION_Y", "AREA_AVAILABLE", "AREA_PER_LIPID"
        ])
        self.csv_file_handle.flush()
        
        # Create full log file for TX/RX if enabled
        if self.full_logging_enabled:
            self.current_full_log_file = os.path.join(LOG_DIR, f"full_log_{timestamp}.txt")
            self.full_log_file_handle = open(self.current_full_log_file, 'w', encoding='utf-8')
            self.full_log_file_handle.write(f"LanGUImuir Full Log Session Started: {datetime.datetime.now()}\n")
            self.full_log_file_handle.write("=" * 60 + "\n\n")
            self.full_log_file_handle.flush()
        
        self.measurement_counter = 0
        self.logging_enabled = True
        
        print(f"New log session started: {self.current_log_file}")
    
    def close_current_logs(self):
        """Close current log files."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None
        
        if self.full_log_file_handle:
            self.full_log_file_handle.close()
            self.full_log_file_handle = None
    
    def log_measurement(self, num_weighings: int, weight: float, surface_tension_calibrated: Optional[float],
                       moving_avg: float, position_x: int, position_y: int, 
                       area_available: float, area_per_lipid: float):
        """Log a measurement to CSV - MODIFIED to include surface_tension_calibrated."""
        if not self.logging_enabled or not self.csv_writer:
            return
        
        self.measurement_counter += 1
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Format surface tension calibrated value (empty if None)
        st_cal_str = f"{surface_tension_calibrated:.4f}" if surface_tension_calibrated is not None else ""
        
        row = [
            self.measurement_counter,
            timestamp,
            num_weighings,
            weight,
            st_cal_str,  # NEW COLUMN
            f"{moving_avg:.3f}",
            position_x,
            position_y,
            f"{area_available:.2e}",
            f"{area_per_lipid:.2f}" if area_per_lipid > 0 else ""
        ]
        
        self.csv_writer.writerow(row)
        self.csv_file_handle.flush()
    
    def log_serial_message(self, message_type: str, message: str):
        """Log a serial message to the full log file."""
        if not self.full_logging_enabled or not self.full_log_file_handle:
            return
        
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message_type.upper()}: {message}\n"
        
        self.full_log_file_handle.write(log_entry)
        self.full_log_file_handle.flush()
    
    def set_full_logging(self, enabled: bool):
        """Enable or disable full TX/RX logging."""
        self.full_logging_enabled = enabled
        
        if enabled and not self.full_log_file_handle:
            # Create full log file if not exists
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_full_log_file = os.path.join(LOG_DIR, f"full_log_{timestamp}.txt")
            self.full_log_file_handle = open(self.current_full_log_file, 'w', encoding='utf-8')
            self.full_log_file_handle.write(f"LanGUImuir Full Log Started: {datetime.datetime.now()}\n")
            self.full_log_file_handle.write("=" * 60 + "\n\n")
            self.full_log_file_handle.flush()
        elif not enabled and self.full_log_file_handle:
            self.full_log_file_handle.close()
            self.full_log_file_handle = None
    
    def get_current_log_info(self) -> Dict[str, str]:
        """Get information about current log files."""
        info = {
            "csv_file": os.path.basename(self.current_log_file) if self.current_log_file else "None",
            "full_log_file": os.path.basename(self.current_full_log_file) if self.current_full_log_file else "None",
            "measurement_count": str(self.measurement_counter),
            "logging_enabled": str(self.logging_enabled),
            "full_logging_enabled": str(self.full_logging_enabled)
        }
        return info


class SerialManager:
    """Handles all serial communication with the Langmuir balance."""
    
    def __init__(self, data_queue: queue.Queue, logging_manager: LoggingManager):
        self.serial_connection: Optional[serial.Serial] = None
        self.data_queue = data_queue
        self.logging_manager = logging_manager
        self.is_connected = False
        self.read_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.message_buffer = MessageBuffer()
        
    def get_available_ports(self) -> List[str]:
        """Get list of available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [f"{port.device} - {port.description}" for port in ports]
    
    def connect(self, port: str, baudrate: int = DEFAULT_BAUDRATE) -> bool:
        """Connect to the specified serial port."""
        try:
            # Extract port name from the formatted string
            port_name = port.split(" - ")[0]
            
            self.serial_connection = serial.Serial(port_name, baudrate, timeout=1)
            self.is_connected = True
            
            # Reset message buffer
            self.message_buffer = MessageBuffer()
            
            # Start reading thread
            self.should_stop.clear()
            self.read_thread = threading.Thread(target=self._read_serial_data, daemon=True)
            self.read_thread.start()
            
            status_msg = f"Connected to {port}"
            self.message_buffer.add_message(status_msg, "status")
            self.logging_manager.log_serial_message("status", status_msg)
            self.data_queue.put({"type": "status", "message": status_msg})
            return True
            
        except serial.SerialException as e:
            error_msg = f"Connection failed: {str(e)}"
            self.data_queue.put({"type": "error", "message": error_msg})
            return False
    
    def disconnect(self):
        """Disconnect from the serial port."""
        self.should_stop.set()
        self.is_connected = False
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
            
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            status_msg = "Disconnected"
            self.message_buffer.add_message(status_msg, "status")
            self.logging_manager.log_serial_message("status", status_msg)
            self.data_queue.put({"type": "status", "message": status_msg})
    
    def send_command(self, command: str, command_type: Optional[str] = None) -> bool:
        """Send a command to the device - MODIFIED to track command type."""
        if not self.is_connected or not self.serial_connection:
            return False
            
        try:
            # Mark command sent in buffer before sending - track type
            self.message_buffer.mark_command_sent(command_type)
            self.message_buffer.add_message(command, "tx")
            self.logging_manager.log_serial_message("tx", command)
            
            self.serial_connection.write(f"{command}\n".encode())
            self.data_queue.put({"type": "tx", "message": command})
            return True
        except serial.SerialException as e:
            error_msg = f"Send failed: {str(e)}"
            self.message_buffer.add_message(error_msg, "error")
            self.logging_manager.log_serial_message("error", error_msg)
            self.data_queue.put({"type": "error", "message": error_msg})
            return False
    
    def wait_for_command_completion(self, timeout: float = DONE_TIMEOUT) -> bool:
        """Wait for command completion using the message buffer."""
        return self.message_buffer.wait_for_done(timeout)
    
    def _read_serial_data(self):
        """Read data from serial port in a separate thread."""
        while not self.should_stop.is_set() and self.is_connected:
            try:
                if self.serial_connection and self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        # Add to message buffer first
                        self.message_buffer.add_message(line, "rx")
                        # Log to file
                        self.logging_manager.log_serial_message("rx", line)
                        # Then send to queue for GUI processing
                        self.data_queue.put({"type": "rx", "message": line})
            except serial.SerialException as e:
                error_msg = f"Read error: {str(e)}"
                self.message_buffer.add_message(error_msg, "error")
                self.logging_manager.log_serial_message("error", error_msg)
                self.data_queue.put({"type": "error", "message": error_msg})
                break
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.message_buffer.add_message(error_msg, "error")
                self.logging_manager.log_serial_message("error", error_msg)
                self.data_queue.put({"type": "error", "message": error_msg})
                break
                
            time.sleep(0.03)  # Small delay to prevent excessive CPU usage


class DataProcessor:
    """Processes and stores measurement data - ENHANCED for calibrated data."""
    
    def __init__(self, logging_manager: LoggingManager, serial_manager: 'SerialManager'):
        self.logging_manager = logging_manager
        self.serial_manager = serial_manager
        self.reset_data()
        
    def reset_data(self):
        """Reset all data buffers."""
        self.timestamps = deque(maxlen=BUFFER_SIZE)
        self.weights = deque(maxlen=BUFFER_SIZE)  # From command 'p'
        self.surface_tensions_calibrated = deque(maxlen=BUFFER_SIZE)  # From command 'P' - NEW
        self.positions_x = deque(maxlen=BUFFER_SIZE)
        self.positions_y = deque(maxlen=BUFFER_SIZE)
        self.weight_moving_avg = deque(maxlen=BUFFER_SIZE)
        self.tension_area_values = deque(maxlen=BUFFER_SIZE)
        
        self.current_weight = 0.0
        self.current_surface_tension_calibrated = None  # NEW
        self.current_position_x = 0
        self.current_position_y = 0
        self.current_area_available = 0.0
        self.current_area_per_lipid = 0.0
        self.current_num_molecules = 0.0
        
    def set_num_molecules(self, num_molecules: float):
        """Set the number of molecules for area per lipid calculation."""
        self.current_num_molecules = num_molecules
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process incoming serial message and extract data - ENHANCED."""
        result = {"processed": False}
        
        # Check for weight measurement
        weight_match = re.search(WEIGHT_PATTERN, message)
        if weight_match:
            num_samples = int(weight_match.group(1))
            weight_value = float(weight_match.group(2))
            
            # Determine if this is from 'p' or 'P' command
            last_command = self.serial_manager.message_buffer.get_last_command_type()
            
            timestamp = datetime.datetime.now()
            self.timestamps.append(timestamp)
            
            # Store in appropriate buffer based on command type
            if last_command == 'P':
                # This is calibrated surface tension from command 'P'
                self.surface_tensions_calibrated.append(weight_value)
                self.current_surface_tension_calibrated = weight_value
                # Don't update weights buffer
            else:
                # This is raw weight from command 'p' (or unknown)
                self.weights.append(weight_value)
                self.current_weight = weight_value
                # Clear calibrated value if switching back to 'p'
                if last_command == 'p':
                    self.current_surface_tension_calibrated = None
            
            # Calculate moving average (only for raw weights)
            if len(self.weights) >= MOVING_AVERAGE_WINDOW:
                avg = sum(list(self.weights)[-MOVING_AVERAGE_WINDOW:]) / MOVING_AVERAGE_WINDOW
            elif len(self.weights) > 0:
                avg = sum(self.weights) / len(self.weights)
            else:
                avg = 0.0
            self.weight_moving_avg.append(avg)
            
            # Calculate area per lipid
            self.current_area_per_lipid = self.calculate_area_per_lipid(self.current_num_molecules)
            
            # Log to CSV
            self.logging_manager.log_measurement(
                num_samples, weight_value, self.current_surface_tension_calibrated,
                avg, self.current_position_x, self.current_position_y,
                self.current_area_available, self.current_area_per_lipid
            )
            
            result.update({
                "processed": True,
                "type": "weight",
                "command_type": last_command,
                "samples": num_samples,
                "value": weight_value,
                "moving_avg": avg,
                "timestamp": timestamp
            })
        
        # Check for position data
        position_match = re.search(POSITION_PATTERN, message)
        if position_match:
            x_pos = int(position_match.group(1))
            y_pos = int(position_match.group(2))
            
            self.positions_x.append(x_pos)
            self.positions_y.append(y_pos)
            self.current_position_x = x_pos
            self.current_position_y = y_pos
            
            # Calculate available area
            self.current_area_available = self._calculate_available_area(x_pos)
            self.current_area_per_lipid = self.calculate_area_per_lipid(self.current_num_molecules)
            
            result.update({
                "processed": True,
                "type": "position",
                "x": x_pos,
                "y": y_pos,
                "area_available": self.current_area_available
            })
        
        return result
    
    def _calculate_available_area(self, x_position: int) -> float:
        """Calculate available area based on X position."""
        # Convert steps to mm and calculate available length
        current_length = TROUGH_LENGTH_MAX - (MM_STEP * x_position)
        area_mm2 = TROUGH_WIDTH * current_length
        # Convert to Ångström²
        area_angstrom2 = area_mm2 * 1e14  # 1 mm² = 1e14 Ų
        return area_angstrom2
    
    def calculate_area_per_lipid(self, num_molecules: float) -> float:
        """Calculate area per lipid molecule."""
        if num_molecules > 0 and self.current_area_available > 0:
            return self.current_area_available / num_molecules
        return 0.0
    
    def get_plot_data(self, max_points: int = 600) -> Dict[str, List]:
        """Get data for plotting with consistent array lengths - ENHANCED."""
        if not self.timestamps:
            return {
                "time_steps": [], "weights": [], "surface_tensions_calibrated": [],
                "positions_x": [], "weight_avg": [], "tension_area": []
            }
        
        # Get last max_points
        end_idx = len(self.timestamps)
        start_idx = max(0, end_idx - max_points)
        
        # Get the actual data arrays with length checking
        timestamps_slice = list(self.timestamps)[start_idx:]
        weights_slice = list(self.weights)[start_idx:] if self.weights else []
        st_cal_slice = list(self.surface_tensions_calibrated)[start_idx:] if self.surface_tensions_calibrated else []
        positions_x_slice = list(self.positions_x)[start_idx:] if self.positions_x else []
        weight_avg_slice = list(self.weight_moving_avg)[start_idx:] if self.weight_moving_avg else []
        
        # Find the minimum length to ensure consistency
        actual_length = len(timestamps_slice)
        
        if actual_length == 0:
            return {
                "time_steps": [], "weights": [], "surface_tensions_calibrated": [],
                "positions_x": [], "weight_avg": [], "tension_area": []
            }
        
        # Create consistent time steps
        time_steps = list(range(start_idx, start_idx + actual_length))
        
        # Pad arrays to match timestamps length
        def pad_array(arr, length, default=0):
            if len(arr) < length:
                return list(arr) + [default] * (length - len(arr))
            return arr[:length]
        
        weights = pad_array(weights_slice, actual_length, 0)
        surface_tensions_cal = pad_array(st_cal_slice, actual_length, None)
        positions_x = pad_array(positions_x_slice, actual_length, 0)
        weight_avg = pad_array(weight_avg_slice, actual_length, 0)
        
        # Calculate tension/area values if we have both weight and position data
        tension_area = []
        for i in range(actual_length):
            weight = weights[i] if i < len(weights) else 0
            pos_x = positions_x[i] if i < len(positions_x) else 0
            
            if pos_x > 0:
                area = self._calculate_available_area(pos_x)
                if area > 0:
                    tension_area.append(weight / area * 1e12)  # Normalize units
                else:
                    tension_area.append(0)
            else:
                tension_area.append(0)
        
        return {
            "time_steps": time_steps,
            "weights": weights,
            "surface_tensions_calibrated": surface_tensions_cal,  # NEW
            "positions_x": positions_x,
            "weight_avg": weight_avg,
            "tension_area": tension_area
        }


class AudioNotifier:
    """Handles audio notifications."""
    
    @staticmethod
    def play_beep():
        """Play a simple beep sound."""
        try:
            winsound.Beep(1000, 300)  # 1000 Hz, 300 ms
        except Exception as e:
            print(f"Audio notification failed: {str(e)}")


class ScriptExecutor:
    """Handles script file execution with enhanced done detection."""
    
    def __init__(self, serial_manager: SerialManager, data_queue: queue.Queue):
        self.serial_manager = serial_manager
        self.data_queue = data_queue
        self.is_executing = False
        self.should_cancel = threading.Event()
        self.execution_thread: Optional[threading.Thread] = None
        
    def execute_script_file(self, script_path: str, delay_hours: int = 0, delay_minutes: int = 0):
        """Execute commands from a script file."""
        if self.is_executing:
            self.data_queue.put({"type": "error", "message": "Script already executing"})
            return
        
        self.is_executing = True
        self.should_cancel.clear()
        self.execution_thread = threading.Thread(
            target=self._execute_script_thread,
            args=(script_path, delay_hours, delay_minutes),
            daemon=True
        )
        self.execution_thread.start()
    
    def cancel_execution(self):
        """Cancel the current script execution."""
        self.should_cancel.set()
    
    def _execute_script_thread(self, script_path: str, delay_hours: int, delay_minutes: int):
        """Execute script in a separate thread."""
        try:
            # Initial delay
            total_delay_seconds = (delay_hours * 3600) + (delay_minutes * 60)
            if total_delay_seconds > 0:
                self.data_queue.put({
                    "type": "status",
                    "message": f"Waiting {delay_hours}h {delay_minutes}m before starting..."
                })
                
                # Check for cancellation during delay
                for _ in range(total_delay_seconds):
                    if self.should_cancel.is_set():
                        self.data_queue.put({"type": "status", "message": "Execution cancelled during delay"})
                        return
                    time.sleep(1)
            
            # Read script file
            with open(script_path, 'r') as f:
                commands = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            
            self.data_queue.put({
                "type": "status",
                "message": f"Starting script execution: {len(commands)} commands"
            })
            
            # Execute commands
            for i, command in enumerate(commands):
                if self.should_cancel.is_set():
                    self.data_queue.put({"type": "status", "message": "Execution cancelled"})
                    break
                
                # Determine command type for tracking
                command_type = None
                if command.startswith('p'):
                    if len(command) > 1 and command[1].isdigit():
                        command_type = 'p'
                    elif command == 'P' or (len(command) > 1 and command[0] == 'P'):
                        command_type = 'P'
                
                self.data_queue.put({
                    "type": "status",
                    "message": f"Executing command {i+1}/{len(commands)}: {command}"
                })
                
                # Send command with type tracking
                if self.serial_manager.send_command(command, command_type):
                    # Wait for completion
                    if not self.serial_manager.wait_for_command_completion(DONE_TIMEOUT):
                        self.data_queue.put({
                            "type": "error",
                            "message": f"Command timeout (continuing): {command}"
                        })
                    
                    # Delay between commands
                    time.sleep(COMMAND_DELAY)
                else:
                    self.data_queue.put({
                        "type": "error",
                        "message": f"Failed to send command (continuing): {command}"
                    })
            
            self.data_queue.put({"type": "status", "message": "Script execution completed"})
            AudioNotifier.play_beep()
            
        except Exception as e:
            self.data_queue.put({"type": "error", "message": f"Script execution failed: {str(e)}"})
        finally:
            self.is_executing = False


class MainWindow:
    """Main application window - ENHANCED with calibrated data visualization."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LanGUImuir v2.3 - Enhanced with Calibrated Surface Tension")
        self.root.geometry("1400x900")
        
        # Initialize managers
        self.data_queue = queue.Queue()
        self.logging_manager = LoggingManager()
        self.serial_manager = SerialManager(self.data_queue, self.logging_manager)
        self.data_processor = DataProcessor(self.logging_manager, self.serial_manager)
        self.script_executor = ScriptExecutor(self.serial_manager, self.data_queue)
        
        # Debug mode
        self.debug_mode = False
        
        # Setup UI
        self.setup_ui()
        self.bind_keyboard_shortcuts()
        
        # Start update loop
        self.update_gui()
        self.update_plot()
        self.update_logging_display()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frames
        self.main_frame = tk.Frame(self.root, bg='white')
        self.main_frame.pack(fill='both', expand=True)
        
        # Left panel (controls)
        self.left_frame = tk.Frame(self.main_frame, bg='lightgray', width=400)
        self.left_frame.pack(side='left', fill='y')
        self.left_frame.pack_propagate(False)
        self.left_frame.grid_rowconfigure(4, weight=1)
        
        # Right panel (plots)
        self.right_frame = tk.Frame(self.main_frame, bg='white')
        self.right_frame.pack(side='right', fill='both', expand=True)
        
        # Setup panels
        self.setup_left_panel()
        self.setup_debug_mode()
        self.setup_right_panel()
        self.setup_menu()
        self.setup_plots()
        
        # Show normal mode by default
        self.show_normal_mode()
    
    def setup_menu(self):
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Script", command=self.load_script)
        file_menu.add_command(label="Create New Log Session", command=self.create_new_log)
        file_menu.add_separator()
        file_menu.add_command(label="Export Debug Log", command=self.export_debug_log)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Debug Mode", command=self.toggle_debug_mode)
        view_menu.add_command(label="Reset Data Buffers", command=self.reset_data_buffers)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="Enhanced Features", command=self.show_enhanced_features)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_left_panel(self):
        """Setup left control panel."""
        # Connection Frame
        self.settings_frame = tk.LabelFrame(self.left_frame, text="Connection Settings", bg='lightgray')
        self.settings_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        tk.Label(self.settings_frame, text="Serial Port:", bg='lightgray').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.port_combo = ttk.Combobox(self.settings_frame, width=30)
        self.port_combo.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        refresh_button = tk.Button(self.settings_frame, text="Refresh Ports", command=self.refresh_ports)
        refresh_button.grid(row=1, column=0, padx=5, pady=2)
        
        self.connect_button = tk.Button(self.settings_frame, text="Connect", command=self.connect_device, bg='lightgreen')
        self.connect_button.grid(row=1, column=1, padx=5, pady=2)
        
        # Script Selection Frame
        script_frame = tk.Frame(self.settings_frame, bg='lightgray')
        script_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        tk.Label(script_frame, text="Script file:", bg='lightgray').grid(row=0, column=0, sticky='w')
        self.script_label = tk.Label(script_frame, text="No script loaded", bg='white', relief='sunken')
        self.script_label.grid(row=0, column=1, sticky='ew', padx=5)
        script_frame.grid_columnconfigure(1, weight=1)
        
        tk.Button(script_frame, text="Load Script", command=self.load_script).grid(row=0, column=2, padx=5)
        
        # Logging Frame
        self.logging_frame = tk.LabelFrame(self.left_frame, text="Logging Configuration", bg='lightgray')
        self.logging_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        self.full_logging_var = tk.BooleanVar()
        self.full_logging_checkbox = tk.Checkbutton(self.logging_frame, text="Enable full TX/RX logging", 
                                                    variable=self.full_logging_var, bg='lightgray',
                                                    command=self.toggle_full_logging)
        self.full_logging_checkbox.grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        tk.Label(self.logging_frame, text="Current CSV file:", bg='lightgray').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.current_csv_label = tk.Label(self.logging_frame, text="", bg='white', relief='sunken')
        self.current_csv_label.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        tk.Label(self.logging_frame, text="Measurements logged:", bg='lightgray').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.measurements_count_label = tk.Label(self.logging_frame, text="0", bg='white', relief='sunken')
        self.measurements_count_label.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        
        # New log button
        self.new_log_button = tk.Button(self.logging_frame, text="Create New Log", command=self.create_new_log, bg='lightblue')
        self.new_log_button.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Lipid Amount Calculation Frame
        self.lipid_frame = tk.LabelFrame(self.left_frame, text="Lipid Amount Calculation", bg='lightgray')
        self.lipid_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        tk.Label(self.lipid_frame, text="Number of molecules deposited:", bg='lightgray').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.molecules_entry = tk.Entry(self.lipid_frame, width=15)
        self.molecules_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        self.molecules_entry.bind('<KeyRelease>', self.update_lipid_calculations)
        
        tk.Label(self.lipid_frame, text="Area available (Ų):", bg='lightgray').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.area_available_label = tk.Label(self.lipid_frame, text="0.0", bg='white', relief='sunken', width=20)
        self.area_available_label.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        tk.Label(self.lipid_frame, text="Area available per lipid (Ų):", bg='lightgray').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.area_per_lipid_label = tk.Label(self.lipid_frame, text="0.0", bg='white', relief='sunken', width=20)
        self.area_per_lipid_label.grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # Execution Frame
        self.execution_frame = tk.LabelFrame(self.left_frame, text="Script Execution", bg='lightgray')
        self.execution_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        
        # Timer
        timer_frame = tk.Frame(self.execution_frame, bg='lightgray')
        timer_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        tk.Label(timer_frame, text="Set timer:", bg='lightgray').grid(row=0, column=0, padx=5)
        self.hours_entry = tk.Entry(timer_frame, width=3)
        self.hours_entry.grid(row=0, column=1, padx=2)
        self.hours_entry.insert(0, "0")
        tk.Label(timer_frame, text=":", bg='lightgray').grid(row=0, column=2)
        self.minutes_entry = tk.Entry(timer_frame, width=3)
        self.minutes_entry.grid(row=0, column=3, padx=2)
        self.minutes_entry.insert(0, "0")
        
        # Buttons
        button_frame = tk.Frame(self.execution_frame, bg='lightgray')
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.execute_button = tk.Button(button_frame, text="Execute", command=self.execute_script)
        self.execute_button.grid(row=0, column=0, padx=5)
        
        self.cancel_button = tk.Button(button_frame, text="Cancel", command=self.cancel_execution, state='disabled')
        self.cancel_button.grid(row=0, column=1, padx=5)
        
        # Manual Control Frame
        self.manual_frame = tk.LabelFrame(self.left_frame, text="Manual Control", bg='lightgray')
        self.manual_frame.grid(row=5, column=0, sticky='ew', padx=5, pady=5)
        
        # Manual control buttons and inputs
        control_frame1 = tk.Frame(self.manual_frame, bg='lightgray')
        control_frame1.grid(row=0, column=0, pady=5)
        
        tk.Button(control_frame1, text="Measure (p)", command=lambda: self.send_manual_command('measure')).grid(row=0, column=0, padx=2)
        self.measure_entry = tk.Entry(control_frame1, width=5)
        self.measure_entry.grid(row=0, column=1, padx=2)
        self.measure_entry.insert(0, "3")
        
        # NEW: Button for calibrated measurement
        tk.Button(control_frame1, text="Measure Cal (P)", command=lambda: self.send_manual_command('measure_calibrated'), bg='lightcyan').grid(row=0, column=2, padx=2)
        
        control_frame2 = tk.Frame(self.manual_frame, bg='lightgray')
        control_frame2.grid(row=1, column=0, pady=5)
        
        tk.Button(control_frame2, text="Up", command=lambda: self.send_manual_command('up')).grid(row=0, column=0, padx=2)
        tk.Button(control_frame2, text="Down", command=lambda: self.send_manual_command('down')).grid(row=0, column=1, padx=2)
        tk.Button(control_frame2, text="Left", command=lambda: self.send_manual_command('left')).grid(row=0, column=2, padx=2)
        tk.Button(control_frame2, text="Right", command=lambda: self.send_manual_command('right')).grid(row=0, column=3, padx=2)
        
        self.movement_entry = tk.Entry(control_frame2, width=5)
        self.movement_entry.grid(row=0, column=4, padx=5)
        self.movement_entry.insert(0, "50")
    
    def setup_debug_mode(self):
        """Setup debug mode interface with enhanced message display."""
        self.debug_frame = tk.Frame(self.left_frame, bg='lightgray')
        
        # Debug logging controls (compact)
        debug_logging_frame = tk.LabelFrame(self.debug_frame, text="Logging Controls", bg='lightgray')
        debug_logging_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        debug_logging_frame.grid_columnconfigure(2, weight=1)
        
        # Compact logging controls
        self.debug_full_logging_var = tk.BooleanVar()
        self.debug_full_logging_checkbox = tk.Checkbutton(debug_logging_frame, text="Full Log", 
                                                         variable=self.debug_full_logging_var, bg='lightgray',
                                                         command=self.toggle_full_logging_debug)
        self.debug_full_logging_checkbox.grid(row=0, column=0, padx=5, pady=2)
        
        self.debug_new_log_button = tk.Button(debug_logging_frame, text="New Log", command=self.create_new_log, bg='lightblue')
        self.debug_new_log_button.grid(row=0, column=1, padx=5, pady=2)
        
        self.debug_log_info_label = tk.Label(debug_logging_frame, text="", bg='lightgray', font=('Arial', 8))
        self.debug_log_info_label.grid(row=0, column=2, sticky='ew', padx=5, pady=2)
        
        # Serial console
        console_frame = tk.LabelFrame(self.debug_frame, text="Enhanced Serial Console - All Messages", bg='lightgray')
        console_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        console_frame.grid_rowconfigure(0, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)
        
        # Console text area with scrollbar
        console_text_frame = tk.Frame(console_frame)
        console_text_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        console_text_frame.grid_rowconfigure(0, weight=1)
        console_text_frame.grid_columnconfigure(0, weight=1)
        
        self.console_text = tk.Text(console_text_frame, bg='black', fg='green', font=('Courier', 9))
        self.console_text.grid(row=0, column=0, sticky='nsew')
        
        # Configure text tags for different message types
        self.console_text.tag_configure("tx", foreground="cyan")
        self.console_text.tag_configure("rx", foreground="white")
        self.console_text.tag_configure("error", foreground="red")
        self.console_text.tag_configure("status", foreground="yellow")
        self.console_text.tag_configure("debug", foreground="lightgreen")
        self.console_text.tag_configure("done", foreground="lime", background="darkgreen")
        
        console_scrollbar = tk.Scrollbar(console_text_frame, command=self.console_text.yview)
        console_scrollbar.grid(row=0, column=1, sticky='ns')
        self.console_text.config(yscrollcommand=console_scrollbar.set)
        
        # Console controls
        console_controls = tk.Frame(console_frame, bg='lightgray')
        console_controls.grid(row=1, column=0, sticky='ew', padx=5, pady=2)
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(console_controls, text="Auto-scroll", variable=self.auto_scroll_var, bg='lightgray').grid(row=0, column=0, padx=5)
        
        tk.Button(console_controls, text="Clear Console", command=self.clear_console).grid(row=0, column=1, padx=5)
        tk.Button(console_controls, text="Refresh Buffer", command=self.refresh_console_buffer).grid(row=0, column=2, padx=5)
        
        # Command input
        command_frame = tk.Frame(console_frame, bg='lightgray')
        command_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        command_frame.grid_columnconfigure(1, weight=1)
        
        tk.Label(command_frame, text="Command:", bg='lightgray').grid(row=0, column=0, padx=5)
        self.command_entry = tk.Entry(command_frame, bg='black', fg='white')
        self.command_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.command_entry.bind('<Return>', self.send_debug_command)
        
        tk.Button(command_frame, text="Send", command=self.send_debug_command).grid(row=0, column=2, padx=5)
        tk.Button(command_frame, text="Execute Script", command=self.execute_script).grid(row=0, column=3, padx=5)
        tk.Button(command_frame, text="Disconnect", command=self.disconnect_device).grid(row=0, column=4, padx=5)
        
        # Debug info frame
        debug_info_frame = tk.LabelFrame(self.debug_frame, text="Debug Information", bg='lightgray')
        debug_info_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        # Message statistics
        stats_frame = tk.Frame(debug_info_frame, bg='lightgray')
        stats_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        tk.Label(stats_frame, text="Total Messages:", bg='lightgray').grid(row=0, column=0, padx=5)
        self.total_messages_label = tk.Label(stats_frame, text="0", bg='white', relief='sunken', width=8)
        self.total_messages_label.grid(row=0, column=1, padx=5)
        
        tk.Label(stats_frame, text="Done Events:", bg='lightgray').grid(row=0, column=2, padx=5)
        self.done_events_label = tk.Label(stats_frame, text="0", bg='white', relief='sunken', width=8)
        self.done_events_label.grid(row=0, column=3, padx=5)
        
        tk.Label(stats_frame, text="CSV Rows:", bg='lightgray').grid(row=1, column=0, padx=5)
        self.csv_rows_label = tk.Label(stats_frame, text="0", bg='white', relief='sunken', width=8)
        self.csv_rows_label.grid(row=1, column=1, padx=5)
        
        tk.Label(stats_frame, text="Log File Size:", bg='lightgray').grid(row=1, column=2, padx=5)
        self.log_size_label = tk.Label(stats_frame, text="0 KB", bg='white', relief='sunken', width=15)
        self.log_size_label.grid(row=1, column=3, padx=5)
        
        self.debug_frame.grid_rowconfigure(1, weight=1)
        self.debug_frame.grid_columnconfigure(0, weight=1)
    
    def setup_right_panel(self):
        """Setup right panel with plots."""
        # Title
        title_frame = tk.Frame(self.right_frame, bg='white')
        title_frame.grid(row=0, column=0, pady=10)
        
        title_label = tk.Label(title_frame, text="LanGUImuir Interface v2.3", 
                              font=('Arial', 16, 'bold'), bg='white')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Enhanced with Calibrated Surface Tension Support", 
                                font=('Arial', 10), bg='white', fg='gray')
        subtitle_label.pack()
        
        # Plots frame
        self.plots_frame = tk.Frame(self.right_frame, bg='white')
        self.plots_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
    
    def setup_plots(self):
        """Setup matplotlib plots - ENHANCED with calibrated data line."""
        # Create figure with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.patch.set_facecolor('white')
        
        # Upper plot: Weight/Tension and Position vs Time
        self.ax1.set_xlabel('Time steps')
        self.ax1.set_ylabel('Tension / Surface Tension', color='blue')
        self.ax1.tick_params(axis='y', labelcolor='blue')
        
        self.ax1_twin = self.ax1.twinx()
        self.ax1_twin.set_ylabel('Position', color='red')
        self.ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # Initialize empty plots - ENHANCED with calibrated line
        self.tension_line, = self.ax1.plot([], [], 'b-', label='Raw Weight (p)', linewidth=1.5)
        self.tension_calibrated_line, = self.ax1.plot([], [], 'g-', label='Surface Tension Cal (P)', linewidth=2)
        self.position_line, = self.ax1_twin.plot([], [], 'r-', label='Position', alpha=0.7)
        
        # Add legend
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax1_twin.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Lower plot: Tension/Area vs Time
        self.ax2.set_xlabel('Time steps')
        self.ax2.set_ylabel('Tension/Area')
        self.tension_area_line, = self.ax2.plot([], [], 'b-')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plots_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def bind_keyboard_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind('<Control-Shift-D>', lambda e: self.toggle_debug_mode())
        self.root.bind('<Control-Shift-d>', lambda e: self.toggle_debug_mode())
        self.root.bind('<Control-r>', lambda e: self.refresh_console_buffer())
        self.root.bind('<Control-l>', lambda e: self.clear_console())
        self.root.bind('<Control-n>', lambda e: self.create_new_log())
    
    def show_normal_mode(self):
        """Show normal mode interface."""
        self.debug_frame.grid_remove()
        self.settings_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        self.logging_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        self.lipid_frame.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        self.execution_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        self.manual_frame.grid(row=5, column=0, sticky='ew', padx=5, pady=5)
        self.debug_mode = False
        
    def show_debug_mode(self):
        """Show debug mode interface."""
        self.settings_frame.grid_remove()
        self.logging_frame.grid_remove()
        self.lipid_frame.grid_remove()
        self.execution_frame.grid_remove()
        self.manual_frame.grid_remove()
        self.debug_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.debug_mode = True
        self.refresh_console_buffer()
        self.sync_logging_checkboxes()
    
    def toggle_debug_mode(self):
        """Toggle between normal and debug mode."""
        if self.debug_mode:
            self.show_normal_mode()
        else:
            self.show_debug_mode()
    
    def sync_logging_checkboxes(self):
        """Synchronize logging checkboxes between modes."""
        if hasattr(self, 'debug_full_logging_var'):
            self.debug_full_logging_var.set(self.full_logging_var.get())
    
    def toggle_full_logging(self):
        """Toggle full logging from normal mode."""
        enabled = self.full_logging_var.get()
        self.logging_manager.set_full_logging(enabled)
        if hasattr(self, 'debug_full_logging_var'):
            self.debug_full_logging_var.set(enabled)
    
    def toggle_full_logging_debug(self):
        """Toggle full logging from debug mode."""
        enabled = self.debug_full_logging_var.get()
        self.logging_manager.set_full_logging(enabled)
        self.full_logging_var.set(enabled)
    
    def create_new_log(self):
        """Create a new log session."""
        result = messagebox.askyesno("New Log Session", 
                                   "This will start a new log session. Current session data will be saved. Continue?")
        if result:
            self.logging_manager.start_new_log_session()
            messagebox.showinfo("New Log", "New log session started successfully!")
    
    def update_logging_display(self):
        """Update logging display elements."""
        log_info = self.logging_manager.get_current_log_info()
        
        # Update normal mode display
        if hasattr(self, 'current_csv_label'):
            self.current_csv_label.config(text=log_info["csv_file"])
        if hasattr(self, 'measurements_count_label'):
            self.measurements_count_label.config(text=log_info["measurement_count"])
        
        # Update debug mode display
        if hasattr(self, 'debug_log_info_label'):
            info_text = f"CSV: {log_info['measurement_count']} rows"
            if log_info["full_logging_enabled"] == "True":
                info_text += f" | Full: {log_info['full_log_file']}"
            self.debug_log_info_label.config(text=info_text)
        
        # Update debug statistics
        if self.debug_mode and hasattr(self, 'csv_rows_label'):
            self.csv_rows_label.config(text=log_info["measurement_count"])
            
            # Calculate log file size
            if self.logging_manager.current_log_file and os.path.exists(self.logging_manager.current_log_file):
                size_bytes = os.path.getsize(self.logging_manager.current_log_file)
                size_kb = size_bytes / 1024
                self.log_size_label.config(text=f"{size_kb:.1f} KB")
        
        # Schedule next update
        self.root.after(2000, self.update_logging_display)
    
    def clear_console(self):
        """Clear the debug console."""
        if hasattr(self, 'console_text'):
            self.console_text.delete(1.0, tk.END)
    
    def refresh_console_buffer(self):
        """Refresh console with all messages from buffer."""
        if not self.debug_mode or not hasattr(self, 'console_text'):
            return
            
        if self.serial_manager.is_connected:
            self.clear_console()
            all_messages = self.serial_manager.message_buffer.get_all_messages()
            
            for msg in all_messages:
                timestamp = msg["timestamp"].strftime("%H:%M:%S.%f")[:-3]
                message_text = msg["message"]
                msg_type = msg["type"]
                
                # Determine display format and tag
                if msg_type == "tx":
                    display_text = f"[{timestamp}] >> {message_text}\n"
                    tag = "tx"
                elif msg_type == "rx":
                    display_text = f"[{timestamp}] << {message_text}\n"
                    if re.search(DONE_PATTERN, message_text, re.IGNORECASE):
                        tag = "done"
                    else:
                        tag = "rx"
                elif msg_type == "error":
                    display_text = f"[{timestamp}] ERROR: {message_text}\n"
                    tag = "error"
                elif msg_type == "status":
                    display_text = f"[{timestamp}] STATUS: {message_text}\n"
                    tag = "status"
                else:
                    display_text = f"[{timestamp}] {message_text}\n"
                    tag = "debug"
                
                self.console_text.insert(tk.END, display_text, tag)
            
            if self.auto_scroll_var.get():
                self.console_text.see(tk.END)
            
            # Update statistics
            all_msgs = self.serial_manager.message_buffer.get_all_messages()
            done_count = sum(1 for msg in all_msgs if re.search(DONE_PATTERN, msg["message"], re.IGNORECASE))
            self.total_messages_label.config(text=str(len(all_msgs)))
            self.done_events_label.config(text=str(done_count))
    
    def refresh_ports(self):
        """Refresh available serial ports."""
        ports = self.serial_manager.get_available_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)
    
    def connect_device(self):
        """Connect to the selected serial port."""
        port = self.port_combo.get()
        if not port:
            messagebox.showwarning("Warning", "Please select a serial port")
            return
        
        if self.serial_manager.connect(port):
            self.connect_button.config(text="Disconnect", command=self.disconnect_device, bg='lightcoral')
            messagebox.showinfo("Success", "Connected successfully")
        else:
            messagebox.showerror("Error", "Failed to connect")
    
    def disconnect_device(self):
        """Disconnect from the serial port."""
        self.serial_manager.disconnect()
        self.connect_button.config(text="Connect", command=self.connect_device, bg='lightgreen')
        messagebox.showinfo("Info", "Disconnected")
    
    def load_script(self):
        """Load a script file."""
        file_path = filedialog.askopenfilename(
            title="Select Script File",
            initialdir=SCRIPTS_DIR,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_script_path = file_path
            self.script_label.config(text=os.path.basename(file_path))
    
    def execute_script(self):
        """Execute the loaded script."""
        if not hasattr(self, 'current_script_path'):
            messagebox.showwarning("Warning", "No script loaded")
            return
        
        if not self.serial_manager.is_connected:
            messagebox.showwarning("Warning", "Not connected to device")
            return
        
        try:
            hours = int(self.hours_entry.get())
            minutes = int(self.minutes_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid timer values")
            return
        
        self.script_executor.execute_script_file(self.current_script_path, hours, minutes)
        self.execute_button.config(state='disabled')
        self.cancel_button.config(state='normal')
    
    def cancel_execution(self):
        """Cancel script execution."""
        self.script_executor.cancel_execution()
        self.execute_button.config(state='normal')
        self.cancel_button.config(state='disabled')
    
    def send_manual_command(self, command_type: str):
        """Send a manual command - ENHANCED with calibrated measurement."""
        if not self.serial_manager.is_connected:
            messagebox.showwarning("Warning", "Not connected to device")
            return
        
        command_map = {
            'measure': lambda: f"p{self.measure_entry.get()}",
            'measure_calibrated': lambda: f"P{self.measure_entry.get()}",  # NEW
            'up': lambda: f"w{self.movement_entry.get()}",
            'down': lambda: f"s{self.movement_entry.get()}",
            'left': lambda: f"a{self.movement_entry.get()}",
            'right': lambda: f"d{self.movement_entry.get()}"
        }
        
        if command_type in command_map:
            command = command_map[command_type]()
            # Determine tracking type
            track_type = 'P' if command_type == 'measure_calibrated' else ('p' if command_type == 'measure' else None)
            self.serial_manager.send_command(command, track_type)
    
    def send_debug_command(self, event=None):
        """Send a command from debug console - ENHANCED."""
        if not self.serial_manager.is_connected:
            return
        
        command = self.command_entry.get().strip()
        if command:
            # Determine command type for tracking
            track_type = None
            if command.startswith('p') and len(command) > 1 and command[1].isdigit():
                track_type = 'p'
            elif command.startswith('P'):
                track_type = 'P'
            
            self.serial_manager.send_command(command, track_type)
            self.command_entry.delete(0, tk.END)
    
    def update_lipid_calculations(self, event=None):
        """Update lipid calculations."""
        try:
            num_molecules = float(self.molecules_entry.get())
            self.data_processor.set_num_molecules(num_molecules)
            
            area_available = self.data_processor.current_area_available
            area_per_lipid = self.data_processor.calculate_area_per_lipid(num_molecules)
            
            self.area_available_label.config(text=f"{area_available:.2e}")
            self.area_per_lipid_label.config(text=f"{area_per_lipid:.2f}")
        except ValueError:
            pass
    
    def update_gui(self):
        """Update GUI with new data from queue."""
        try:
            while True:
                data = self.data_queue.get_nowait()
                
                if data["type"] == "rx":
                    # Process received data
                    result = self.data_processor.process_message(data["message"])
                    
                    # Update debug console if in debug mode
                    if self.debug_mode and hasattr(self, 'console_text'):
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        display_text = f"[{timestamp}] << {data['message']}\n"
                        
                        if re.search(DONE_PATTERN, data["message"], re.IGNORECASE):
                            tag = "done"
                        else:
                            tag = "rx"
                        
                        self.console_text.insert(tk.END, display_text, tag)
                        if self.auto_scroll_var.get():
                            self.console_text.see(tk.END)
                
                elif data["type"] == "tx":
                    if self.debug_mode and hasattr(self, 'console_text'):
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        display_text = f"[{timestamp}] >> {data['message']}\n"
                        self.console_text.insert(tk.END, display_text, "tx")
                        if self.auto_scroll_var.get():
                            self.console_text.see(tk.END)
                
                elif data["type"] == "status":
                    if self.debug_mode and hasattr(self, 'console_text'):
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        display_text = f"[{timestamp}] STATUS: {data['message']}\n"
                        self.console_text.insert(tk.END, display_text, "status")
                        if self.auto_scroll_var.get():
                            self.console_text.see(tk.END)
                    
                    # Check for execution completion
                    if "completed" in data["message"].lower():
                        self.execute_button.config(state='normal')
                        self.cancel_button.config(state='disabled')
                
                elif data["type"] == "error":
                    if self.debug_mode and hasattr(self, 'console_text'):
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        display_text = f"[{timestamp}] ERROR: {data['message']}\n"
                        self.console_text.insert(tk.END, display_text, "error")
                        if self.auto_scroll_var.get():
                            self.console_text.see(tk.END)
                
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(UPDATE_INTERVAL, self.update_gui)
    
    def update_plot(self):
        """Update matplotlib plots - ENHANCED with calibrated data."""
        try:
            plot_data = self.data_processor.get_plot_data()
            
            if plot_data["time_steps"]:
                # Update tension/weight line (blue - raw from 'p')
                self.tension_line.set_data(plot_data["time_steps"], plot_data["weights"])
                
                # Update calibrated surface tension line (green - from 'P') - NEW
                # Filter out None values for plotting
                st_cal = plot_data["surface_tensions_calibrated"]
                if st_cal and any(x is not None for x in st_cal):
                    # Create filtered lists with only non-None values
                    filtered_steps = [plot_data["time_steps"][i] for i in range(len(st_cal)) if st_cal[i] is not None]
                    filtered_st_cal = [v for v in st_cal if v is not None]
                    self.tension_calibrated_line.set_data(filtered_steps, filtered_st_cal)
                else:
                    self.tension_calibrated_line.set_data([], [])
                
                # Update position line (red)
                self.position_line.set_data(plot_data["time_steps"], plot_data["positions_x"])
                
                # Update tension/area line
                self.tension_area_line.set_data(plot_data["time_steps"], plot_data["tension_area"])
                
                # Rescale axes
                self.ax1.relim()
                self.ax1.autoscale_view()
                self.ax1_twin.relim()
                self.ax1_twin.autoscale_view()
                
                self.ax2.relim()
                self.ax2.autoscale_view()
                
                # Redraw
                self.canvas.draw_idle()
        
        except Exception as e:
            if self.debug_mode:
                print(f"Plot update error: {str(e)}")
        
        # Schedule next update
        self.root.after(500, self.update_plot)
    
    def reset_data_buffers(self):
        """Reset all data buffers."""
        result = messagebox.askyesno("Confirm Reset", 
                                   "This will clear all collected data and message buffers. Continue?")
        if result:
            self.data_processor.reset_data()
            if self.serial_manager.is_connected:
                self.serial_manager.message_buffer = MessageBuffer()
            if self.debug_mode:
                self.clear_console()
            messagebox.showinfo("Reset", "All data buffers have been reset")
    
    def export_debug_log(self):
        """Export debug log to a file."""
        if not self.serial_manager.is_connected:
            messagebox.showwarning("Warning", "Not connected - no debug log available")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Debug Log",
            initialdir=LOG_DIR
        )
        
        if file_path:
            try:
                all_messages = self.serial_manager.message_buffer.get_all_messages()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"LanGUImuir Enhanced Debug Log v2.3\n")
                    f.write(f"Exported: {timestamp}\n")
                    f.write(f"Total Messages: {len(all_messages)}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for msg in all_messages:
                        msg_timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        f.write(f"[{msg_timestamp}] {msg['type'].upper()}: {msg['message']}\n")
                
                messagebox.showinfo("Export Complete", f"Debug log exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export debug log:\n{str(e)}")
    
    def show_shortcuts(self):
        """Show keyboard shortcuts help."""
        shortcuts_text = """⌨️ Keyboard Shortcuts:

Ctrl+Shift+D    Toggle Debug Mode
Ctrl+R          Refresh Console Buffer  
Ctrl+L          Clear Console
Ctrl+N          Create New Log Session
Enter           Send Command (in debug console)

🎯 Logging Features:
• Automatic CSV logging with SURFACE_TENSION_CALIBRATED column
• Distinguishes between command 'p' (raw) and 'P' (calibrated)
• Real-time file size monitoring
• European date format (DD/MM/YYYY)

📊 CSV Structure (v2.3):
COUNT;TIMESTAMP;NUM_WEIGHINGS;WEIGHT;SURFACE_TENSION_CALIBRATED;MOVING_AVERAGE;POSITION_X;POSITION_Y;AREA_AVAILABLE;AREA_PER_LIPID

🔧 New Features v2.3:
• Command 'P' support for calibrated surface tension
• Green line in plot for calibrated data
• Enhanced command tracking ('p' vs 'P')
• Separate data buffers for raw and calibrated measurements

📈 Plot Legend:
• Blue line: Raw weight (command 'p')
• Green line: Calibrated surface tension (command 'P')
• Red line: Position (right Y-axis)"""
        
        messagebox.showinfo("Keyboard Shortcuts & Features", shortcuts_text)
    
    def show_enhanced_features(self):
        """Show information about enhanced features."""
        features_text = """🚀 Enhanced Features v2.3 - Calibrated Surface Tension Support

🆕 NEW in v2.3:
• Command 'P' support for calibrated surface tension measurements
• SURFACE_TENSION_CALIBRATED column in CSV logs
• Green plot line for visualizing calibrated data
• Smart command tracking to distinguish 'p' from 'P'
• Separate data buffers for raw and calibrated measurements
• "Measure Cal (P)" button in manual controls

📄 Enhanced CSV Logging:
• New column: SURFACE_TENSION_CALIBRATED (mN/m)
• Values populated when using command 'P'
• Empty when using command 'p' (raw weight)
• All previous columns maintained

📊 Enhanced Plotting:
• Blue line: Raw weight from command 'p'
• Green line: Calibrated surface tension from command 'P'
• Both lines plotted simultaneously
• Automatic legend for easy identification
• Independent scaling for optimal visibility

🔧 Technical Implementation:
• MessageBuffer tracks last command type ('p' or 'P')
• DataProcessor maintains separate deques for raw and calibrated
• SerialManager passes command type to tracking system
• LoggingManager handles NULL values for unused columns

⚙️ Backward Compatibility:
• All v2.2 features fully maintained
• Existing scripts continue to work
• Old CSV structure extended, not replaced
• No breaking changes to existing workflows

🎛️ Usage Tips:
• Use 'p' command for raw sensor readings
• Use 'P' command after Arduino calibration (command 'T')
• Both measurements logged with timestamp correlation
• Compare raw vs calibrated in real-time on plot"""
        
        messagebox.showinfo("Enhanced Features v2.3", features_text)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """LanGUImuir Interface v2.3 - Calibrated Surface Tension

A comprehensive Langmuir balance control software with support
for both raw weight measurements and calibrated surface tension.

🆕 Version 2.3 Highlights:
• Support for Arduino command 'P' (calibrated surface tension)
• SURFACE_TENSION_CALIBRATED column in CSV logs
• Enhanced plotting with dual measurement visualization
• Smart command tracking for 'p' vs 'P' distinction

🔧 Core Features:
• Dual measurement modes: raw (p) and calibrated (P)
• Real-time CSV logging with extended columns
• Enhanced plots with color-coded measurement types
• Event-based "Done" detection with intelligent fallback
• Complete debug mode with message tracking

📊 Calibration Workflow:
1. Use Arduino command 'T' to perform calibration
2. Input reference weights and Wilhelmy plate perimeter
3. Use command 'P' for calibrated measurements
4. Data automatically logged with surface tension values

⚙️ Technical Details:
• Separate data buffers for raw and calibrated measurements
• Command type tracking throughout execution pipeline
• Thread-safe operations with immediate data flushing
• Backward compatible with existing scripts

📁 File Structure:
• /scripts/ - Command script files
• /log/ - CSV data and communication logs (enhanced format)
• /graphics/ - Exported plots and visualizations
• config.json - Application settings

Created with Python, Tkinter, matplotlib, and advanced threading.
Enhanced for precision surface chemistry measurements.

For support, use debug mode for detailed system information."""
        
        messagebox.showinfo("About LanGUImuir v2.3", about_text)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()
        
        # Cleanup on exit
        if self.serial_manager.is_connected:
            self.serial_manager.disconnect()
        self.logging_manager.close_current_logs()


def main():
    """Main entry point."""
    try:
        # Check for required directories
        required_dirs = [SCRIPTS_DIR, LOG_DIR, GRAPHICS_DIR]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        
        print(f"LanGUImuir v2.3 starting...")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Log directory: {LOG_DIR}")
        print(f"Scripts directory: {SCRIPTS_DIR}")
        print(f"Graphics directory: {GRAPHICS_DIR}")
        
        app = MainWindow()
        app.run()
        
    except Exception as e:
        error_msg = f"Application failed to start: {str(e)}"
        print(error_msg)
        try:
            messagebox.showerror("Startup Error", error_msg)
        except:
            print("Failed to show error dialog")


if __name__ == "__main__":
    main()
