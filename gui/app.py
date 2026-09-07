"""
gui/app.py
==========
Modern dark-mode CustomTkinter interface for CSRT Object Tracker
and YOLO Autonomous Object Detection/Tracking with Pan-Tilt visual servoing.
"""

from collections import deque
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from camera import open_video_capture
from gui.icons import get_icon
from servo_controller import PanTiltServoing
from tracker import CSRTTrackerEngine, TrackingResult, compute_spatial_reliability_map
from yolo_tracker import YOLOTrackerEngine


class CSRTTrackerApp:
    def __init__(
        self,
        root: ctk.CTk,
        video_source=0,
        servo_controller: PanTiltServoing = None,
        default_mode: str = "YOLO Auto",
        yolo_weights: str = None,
        yolo_conf: float = 0.80,
    ):
        self.root = root
        self.root.title("Autonomous Tracker & Pan/Tilt Visual Servoing")
        self.root.geometry("1180x780")
        self.root.minsize(980, 640)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Subsystems
        self.mode = default_mode  # "YOLO Auto" or "CSRT Manual"
        self.csrt_tracker = CSRTTrackerEngine(max_recovery_frames=15, max_trajectory=40)
        self.yolo_tracker = None
        self.yolo_weights = yolo_weights
        self.yolo_conf = yolo_conf

        # Try initializing YOLO tracker
        try:
            self.yolo_tracker = YOLOTrackerEngine(weights_path=self.yolo_weights, conf=self.yolo_conf)
        except Exception as e:
            print(f"Warning: Could not initialize YOLO tracker: {e}. Defaulting to CSRT Manual.")
            self.mode = "CSRT Manual"

        self.servo = servo_controller or PanTiltServoing()

        # Stream & App State
        self.cap = None
        self.source_desc = "Initializing..."
        self.is_paused = False
        self.is_running = True
        self.drag_start = None
        self.drag_current = None
        self.last_frame = None
        self.last_clean_frame = None
        self.canvas_img_id = None
        self.tk_image = None

        # Telemetry & Performance throttling
        self.fps_tracker = deque(maxlen=20)
        self.prev_time = time.monotonic()
        self.last_telemetry_time = 0.0

        # UI Toggles
        self.var_mode = tk.StringVar(value=self.mode)
        self.var_spatial_map = tk.BooleanVar(value=False)
        self.var_pip_map = tk.BooleanVar(value=True)
        self.var_trajectory = tk.BooleanVar(value=True)
        self.var_boresight = tk.BooleanVar(value=True)
        self.var_mirror = tk.BooleanVar(value=False)
        self.var_servo_active = tk.BooleanVar(value=True)
        self.var_inv_pan = tk.BooleanVar(value=(self.servo.pan_sign < 0))
        self.var_inv_tilt = tk.BooleanVar(value=(self.servo.tilt_sign > 0))

        self._load_icons()
        self._build_ui()
        self._bind_events()

        # Threaded Tracking Subsystem for High-FPS Servo Servoing
        self._track_lock = threading.Lock()
        self._worker_frame = None
        self._worker_result = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)
        self._worker_busy = False
        self._tracker_thread = threading.Thread(target=self._tracker_worker, daemon=True)
        self._tracker_thread.start()

        # Open Video Stream
        self._open_source(video_source)

        # Start Processing Loop
        self.root.after(10, self._process_frame)

    def _tracker_worker(self):
        while self.is_running:
            frame_to_process = None
            with self._track_lock:
                if self._worker_frame is not None and not self.is_paused:
                    frame_to_process = self._worker_frame
                    self._worker_frame = None
                    self._worker_busy = True

            if frame_to_process is not None:
                try:
                    if self.mode == "YOLO Auto" and self.yolo_tracker is not None:
                        res = self.yolo_tracker.update(frame_to_process)
                    else:
                        res = self.csrt_tracker.update(frame_to_process)
                except Exception:
                    res = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)

                with self._track_lock:
                    self._worker_result = res
                    self._worker_busy = False
            else:
                time.sleep(0.005)

    def _load_icons(self):
        self.icon_target = get_icon("target", (16, 16), "#ffffff")
        self.icon_reset = get_icon("reset", (15, 15), "#ffffff")
        self.icon_pause = get_icon("pause", (14, 14), "#ffffff")
        self.icon_play = get_icon("play", (14, 14), "#ffffff")
        self.icon_camera = get_icon("camera", (16, 16), "#cbd0df")
        self.icon_folder = get_icon("folder", (16, 16), "#cbd0df")
        self.icon_servo = get_icon("servo", (16, 16), "#cbd0df")

    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=0, minsize=320)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ----------------- SIDEBAR -----------------
        self.sidebar = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#18191c")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Brand Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(12, 4))

        title_lbl = ctk.CTkLabel(
            brand_frame,
            text="AI VISION TRACKER",
            font=ctk.CTkFont(family="Inter, -apple-system, sans-serif", size=18, weight="bold"),
            text_color="#ffffff",
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            brand_frame,
            text="YOLO Drone Detection & CSRT Tracking",
            font=ctk.CTkFont(size=11),
            text_color="#8c92a4",
        )
        sub_lbl.pack(anchor="w")

        # Mode Selector (YOLO Auto vs CSRT Manual)
        mode_frame = ctk.CTkFrame(self.sidebar, fg_color="#21232a", corner_radius=8)
        mode_frame.pack(fill="x", padx=16, pady=(4, 8))

        ctk.CTkLabel(
            mode_frame,
            text="TRACKING ENGINE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9aa0b4",
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.seg_mode = ctk.CTkSegmentedButton(
            mode_frame,
            values=["YOLO Auto", "CSRT Manual"],
            variable=self.var_mode,
            command=self._on_mode_change,
            selected_color="#2563eb",
            selected_hover_color="#1d4ed8",
            unselected_color="#14151a",
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.seg_mode.pack(fill="x", padx=8, pady=(0, 8))

        # Live Status Pill
        self.status_pill = ctk.CTkLabel(
            self.sidebar,
            text="IDLE READY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e0e0e0",
            fg_color="#2b2d35",
            corner_radius=8,
            height=28,
        )
        self.status_pill.pack(fill="x", padx=16, pady=(0, 8))

        # Confidence Bar Card
        score_card = ctk.CTkFrame(self.sidebar, fg_color="#21232a", corner_radius=10)
        score_card.pack(fill="x", padx=16, pady=(0, 8))

        score_header = ctk.CTkFrame(score_card, fg_color="transparent")
        score_header.pack(fill="x", padx=10, pady=(6, 2))
        ctk.CTkLabel(
            score_header,
            text="Detection / Track Score",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9aa0b4",
        ).pack(side="left")

        self.score_val_lbl = ctk.CTkLabel(
            score_header,
            text="0%",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00d285",
        )
        self.score_val_lbl.pack(side="right")

        self.score_bar = ctk.CTkProgressBar(
            score_card,
            height=7,
            corner_radius=4,
            progress_color="#00d285",
            fg_color="#14151a",
        )
        self.score_bar.pack(fill="x", padx=10, pady=(0, 8))
        self.score_bar.set(0.0)

        # Servo Control Card
        servo_card = ctk.CTkFrame(self.sidebar, fg_color="#21232a", corner_radius=10)
        servo_card.pack(fill="x", padx=16, pady=(0, 8))

        servo_header = ctk.CTkFrame(servo_card, fg_color="transparent")
        servo_header.pack(fill="x", padx=10, pady=(6, 3))
        ctk.CTkLabel(
            servo_header,
            text="PAN-TILT VISUAL SERVOING",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9aa0b4",
        ).pack(side="left")

        hw_port = getattr(self.servo, 'active_port', None)
        hw_text = f"ESP32: {hw_port}" if self.servo.is_hardware else "ESP32: MOCK (SIMULATED)"
        hw_bg = "#065f46" if self.servo.is_hardware else "#374151"
        hw_fg = "#6ee7b7" if self.servo.is_hardware else "#cbd0df"

        self.servo_hw_badge = ctk.CTkLabel(
            servo_card,
            text=hw_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=hw_bg,
            text_color=hw_fg,
            corner_radius=6,
            height=18,
        )
        self.servo_hw_badge.pack(fill="x", padx=10, pady=(0, 4))

        self.lbl_servo_angles = ctk.CTkLabel(
            servo_card,
            text="",
            height=0,
        )

        # Real-Time PWM Visualizer
        pwm_frame = ctk.CTkFrame(servo_card, fg_color="#181a20", corner_radius=8)
        pwm_frame.pack(fill="x", padx=10, pady=(2, 6))

        # Pan PWM Row
        pan_info_row = ctk.CTkFrame(pwm_frame, fg_color="transparent")
        pan_info_row.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(
            pan_info_row,
            text="PAN PWM",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9aa0b4",
        ).pack(side="left")
        self.lbl_pan_pwm = ctk.CTkLabel(
            pan_info_row,
            text="1500 µs (0.0°)",
            font=ctk.CTkFont(size=10, family="Courier, monospace", weight="bold"),
            text_color="#38bdf8",
        )
        self.lbl_pan_pwm.pack(side="right")

        self.bar_pan_pwm = ctk.CTkProgressBar(
            pwm_frame,
            progress_color="#38bdf8",
            fg_color="#2b2d35",
            height=6,
            corner_radius=3,
        )
        self.bar_pan_pwm.set(0.5)
        self.bar_pan_pwm.pack(fill="x", padx=8, pady=(2, 4))

        # Tilt PWM Row
        tilt_info_row = ctk.CTkFrame(pwm_frame, fg_color="transparent")
        tilt_info_row.pack(fill="x", padx=8, pady=(0, 0))
        ctk.CTkLabel(
            tilt_info_row,
            text="TILT PWM",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9aa0b4",
        ).pack(side="left")
        self.lbl_tilt_pwm = ctk.CTkLabel(
            tilt_info_row,
            text="1500 µs (0.0°)",
            font=ctk.CTkFont(size=10, family="Courier, monospace", weight="bold"),
            text_color="#00d285",
        )
        self.lbl_tilt_pwm.pack(side="right")

        self.bar_tilt_pwm = ctk.CTkProgressBar(
            pwm_frame,
            progress_color="#00d285",
            fg_color="#2b2d35",
            height=6,
            corner_radius=3,
        )
        self.bar_tilt_pwm.set(0.5)
        self.bar_tilt_pwm.pack(fill="x", padx=8, pady=(2, 4))

        # Serial TX Stream badge
        tx_row = ctk.CTkFrame(pwm_frame, fg_color="transparent")
        tx_row.pack(fill="x", padx=8, pady=(1, 4))
        ctk.CTkLabel(
            tx_row,
            text="ESP32 TX:",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#64748b",
        ).pack(side="left")
        self.lbl_serial_tx = ctk.CTkLabel(
            tx_row,
            text="P:1500 T:1500",
            font=ctk.CTkFont(size=10, family="Courier, monospace"),
            text_color="#a5f3fc",
        )
        self.lbl_serial_tx.pack(side="right")

        servo_switch_row = ctk.CTkFrame(servo_card, fg_color="transparent")
        servo_switch_row.pack(fill="x", padx=10, pady=(2, 2))

        sw_servo = ctk.CTkSwitch(
            servo_switch_row,
            text="Servo Output",
            variable=self.var_servo_active,
            command=self._on_servo_toggle,
            progress_color="#0284c7",
            font=ctk.CTkFont(size=11),
        )
        sw_servo.pack(side="left")

        inv_row = ctk.CTkFrame(servo_card, fg_color="transparent")
        inv_row.pack(fill="x", padx=10, pady=(0, 4))

        cb_inv_pan = ctk.CTkCheckBox(
            inv_row,
            text="Inv Pan",
            variable=self.var_inv_pan,
            command=self._on_inv_pan_toggle,
            checkbox_width=16,
            checkbox_height=16,
            font=ctk.CTkFont(size=10),
        )
        cb_inv_pan.pack(side="left", padx=(0, 8))

        cb_inv_tilt = ctk.CTkCheckBox(
            inv_row,
            text="Inv Tilt",
            variable=self.var_inv_tilt,
            command=self._on_inv_tilt_toggle,
            checkbox_width=16,
            checkbox_height=16,
            font=ctk.CTkFont(size=10),
        )
        cb_inv_tilt.pack(side="left")

        btn_servo_row = ctk.CTkFrame(servo_card, fg_color="transparent")
        btn_servo_row.pack(fill="x", padx=10, pady=(0, 6))
        btn_servo_row.grid_columnconfigure(0, weight=1)
        btn_servo_row.grid_columnconfigure(1, weight=1)

        btn_recenter = ctk.CTkButton(
            btn_servo_row,
            text="Center Servos",
            command=self._on_recenter_servos,
            fg_color="#1e293b",
            hover_color="#334155",
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
        )
        btn_recenter.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        btn_reconnect = ctk.CTkButton(
            btn_servo_row,
            text="Reconnect ESP32",
            command=self._on_reconnect_esp32,
            fg_color="#1e293b",
            hover_color="#334155",
            height=24,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
        )
        btn_reconnect.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # Telemetry Card
        self.telemetry_card = ctk.CTkFrame(self.sidebar, fg_color="#21232a", corner_radius=10)
        self.telemetry_card.pack(fill="x", padx=16, pady=(0, 8))

        self.lbl_fps = ctk.CTkLabel(
            self.telemetry_card,
            text="FPS: --",
            font=ctk.CTkFont(size=11, family="Courier, monospace"),
            text_color="#cbd0df",
        )
        self.lbl_fps.pack(anchor="w", padx=10, pady=(5, 1))

        self.lbl_bbox = ctk.CTkLabel(
            self.telemetry_card,
            text="Target: None",
            font=ctk.CTkFont(size=11, family="Courier, monospace"),
            text_color="#cbd0df",
        )
        self.lbl_bbox.pack(anchor="w", padx=10, pady=1)

        self.lbl_source = ctk.CTkLabel(
            self.telemetry_card,
            text="Source: Initializing",
            font=ctk.CTkFont(size=11),
            text_color="#7b8196",
        )
        self.lbl_source.pack(anchor="w", padx=10, pady=(1, 5))

        # Action Buttons
        self.btn_select = ctk.CTkButton(
            self.sidebar,
            text="Select Target (Drag on Video)",
            image=self.icon_target,
            compound="left",
            command=self._prompt_select,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=30,
            corner_radius=8,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.btn_select.pack(fill="x", padx=16, pady=2)

        btn_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=2)
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self.btn_reset = ctk.CTkButton(
            btn_row,
            text="Reset",
            image=self.icon_reset,
            compound="left",
            command=self._reset_tracker,
            fg_color="#374151",
            hover_color="#4b5563",
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11),
        )
        self.btn_reset.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self.btn_pause = ctk.CTkButton(
            btn_row,
            text="Pause",
            image=self.icon_pause,
            compound="left",
            command=self._toggle_pause,
            fg_color="#374151",
            hover_color="#4b5563",
            height=28,
            corner_radius=8,
            font=ctk.CTkFont(size=11),
        )
        self.btn_pause.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # Visual Toggles
        toggles_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        toggles_frame.pack(fill="x", padx=16, pady=(4, 2))

        self.sw_spatial = ctk.CTkSwitch(
            toggles_frame,
            text="Spatial Map (CSRT)",
            variable=self.var_spatial_map,
            progress_color="#00d285",
            font=ctk.CTkFont(size=11),
        )
        self.sw_spatial.pack(anchor="w", pady=1)

        self.sw_pip = ctk.CTkSwitch(
            toggles_frame,
            text="PiP Heatmap",
            variable=self.var_pip_map,
            progress_color="#00d285",
            font=ctk.CTkFont(size=11),
        )
        self.sw_pip.pack(anchor="w", pady=1)

        sw3 = ctk.CTkSwitch(
            toggles_frame,
            text="Trajectory Trail",
            variable=self.var_trajectory,
            progress_color="#2563eb",
            font=ctk.CTkFont(size=11),
        )
        sw3.pack(anchor="w", pady=1)

        sw4 = ctk.CTkSwitch(
            toggles_frame,
            text="Optical Boresight",
            variable=self.var_boresight,
            progress_color="#38bdf8",
            font=ctk.CTkFont(size=11),
        )
        sw4.pack(anchor="w", pady=1)

        sw_mirror = ctk.CTkSwitch(
            toggles_frame,
            text="Mirror View",
            variable=self.var_mirror,
            progress_color="#0284c7",
            font=ctk.CTkFont(size=11),
        )
        sw_mirror.pack(anchor="w", pady=1)

        # Source Switchers
        src_label = ctk.CTkLabel(
            self.sidebar,
            text="VIDEO SOURCE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#9aa0b4",
        )
        src_label.pack(anchor="w", padx=16, pady=(6, 2))

        src_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        src_row.pack(fill="x", padx=16, pady=2)
        src_row.grid_columnconfigure(0, weight=2)
        src_row.grid_columnconfigure(1, weight=1)

        self.cam_options = ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]
        self.opt_camera = ctk.CTkOptionMenu(
            src_row,
            values=self.cam_options,
            command=self._on_camera_selected,
            fg_color="#1f2937",
            button_color="#374151",
            button_hover_color="#4b5563",
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
        )
        self.opt_camera.set("Camera 0")
        self.opt_camera.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        btn_file = ctk.CTkButton(
            src_row,
            text="File",
            image=self.icon_folder,
            compound="left",
            command=self._open_file_dialog,
            fg_color="#1f2937",
            hover_color="#374151",
            height=26,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
        )
        btn_file.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # Instructions Footer
        shortcuts_box = ctk.CTkLabel(
            self.sidebar,
            text="[T] Switch Engine | [SPACE] Pause | [C] Reset | [Q] Quit",
            font=ctk.CTkFont(size=9),
            text_color="#555a6d",
            wraplength=270,
            justify="center",
        )
        shortcuts_box.pack(side="bottom", pady=6)

        # Update initial UI state based on mode
        self._update_mode_ui()

        # ----------------- MAIN VIDEO CANVAS -----------------
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#0d0e12")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(
            self.main_frame,
            bg="#0d0e12",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

    def _bind_events(self):
        self.video_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.video_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.root.bind("<space>", lambda e: self._on_space_key())
        self.root.bind("<c>", lambda e: self._reset_tracker())
        self.root.bind("<C>", lambda e: self._reset_tracker())
        self.root.bind("<m>", lambda e: self.var_spatial_map.set(not self.var_spatial_map.get()))
        self.root.bind("<M>", lambda e: self.var_spatial_map.set(not self.var_spatial_map.get()))
        self.root.bind("<t>", lambda e: self._toggle_mode())
        self.root.bind("<T>", lambda e: self._toggle_mode())
        self.root.bind("<q>", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_mode(self):
        new_mode = "CSRT Manual" if self.mode == "YOLO Auto" else "YOLO Auto"
        self.var_mode.set(new_mode)
        self._on_mode_change(new_mode)

    def _on_mode_change(self, mode_val):
        self.mode = mode_val
        self._reset_tracker()
        self._update_mode_ui()

    def _update_mode_ui(self):
        if self.mode == "YOLO Auto":
            self.btn_select.configure(
                text="Autonomous Detection Active",
                state="disabled",
                fg_color="#1e293b",
            )
            self._set_status("YOLO SCANNING FOR DRONE...", "#1e3a8a", "#93c5fd")
        else:
            self.btn_select.configure(
                text="Select Target (Drag on Video)",
                state="normal",
                fg_color="#2563eb",
            )
            self._set_status("IDLE READY (DRAG TO SELECT)", "#2b2d35", "#e0e0e0")

    def _on_servo_toggle(self):
        self.servo.enabled = self.var_servo_active.get()

    def _on_inv_pan_toggle(self):
        self.servo.pan_sign = -1 if self.var_inv_pan.get() else 1

    def _on_inv_tilt_toggle(self):
        self.servo.tilt_sign = 1 if self.var_inv_tilt.get() else -1

    def _update_servo_telemetry(self, pan_ang: float, tilt_ang: float):
        pan_us = self.servo.pan_us
        tilt_us = self.servo.tilt_us
        pan_ratio = max(0.0, min(1.0, (pan_us - 500) / 2000.0))
        tilt_ratio = max(0.0, min(1.0, (tilt_us - 500) / 2000.0))

        self.bar_pan_pwm.set(pan_ratio)
        self.bar_tilt_pwm.set(tilt_ratio)
        self.lbl_pan_pwm.configure(text=f"{pan_us} µs ({pan_ang:+5.1f}°)")
        self.lbl_tilt_pwm.configure(text=f"{tilt_us} µs ({tilt_ang:+5.1f}°)")
        self.lbl_serial_tx.configure(text=f"P:{pan_us} T:{tilt_us}")

    def _on_recenter_servos(self):
        self.servo.center_servos()
        self._update_servo_telemetry(self.servo.pan_angle, self.servo.tilt_angle)

    def _on_camera_selected(self, choice: str):
        try:
            cam_idx = int(choice.split()[-1])
        except Exception:
            cam_idx = 0
        self._open_source(cam_idx)

    def _on_reconnect_esp32(self):
        if self.servo.reconnect():
            self.servo_hw_badge.configure(
                text=f"ESP32: {self.servo.active_port}",
                fg_color="#065f46",
                text_color="#6ee7b7",
            )
        else:
            self.servo_hw_badge.configure(
                text="ESP32: NOT DETECTED (MOCK)",
                fg_color="#7f1d1d",
                text_color="#fca5a5",
            )

    # -------------------------------------------------------------------------
    # Stream Management
    # -------------------------------------------------------------------------
    def _open_source(self, source):
        if self.cap is not None:
            self.cap.release()

        self._reset_tracker()
        self.canvas_img_id = None

        cap = open_video_capture(source)

        if not cap.isOpened():
            err_msg = f"Cannot access video source '{source}'."
            print(f"Warning: {err_msg}")
            self._set_status("NO FEED - SELECT VIDEO FILE", "#7f1d1d", "#fca5a5")
            self.lbl_source.configure(text="Source: Unavailable")
            self.cap = None
            return

        self.cap = cap
        self.source_desc = getattr(cap, "source_desc", str(source))
        self.lbl_source.configure(text=f"Source: {self.source_desc}")
        self._update_mode_ui()

    def _init_webcam(self):
        self._open_source(0)

    def _open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All Files", "*.*")],
        )
        if path:
            self._open_source(path)

    # -------------------------------------------------------------------------
    # State & Control Callbacks
    # -------------------------------------------------------------------------
    def _set_status(self, text, bg_color, text_color):
        self.status_pill.configure(text=text, fg_color=bg_color, text_color=text_color)

    def _reset_tracker(self):
        with self._track_lock:
            self._worker_frame = None
            self._worker_result = TrackingResult(success=False, bbox=None, confidence=0.0, center=None)
            if self.mode == "YOLO Auto" and self.yolo_tracker is not None:
                self.yolo_tracker.reset()
            else:
                self.csrt_tracker.reset()

        self.score_bar.set(0.0)
        self.score_val_lbl.configure(text="0%", text_color="#9aa0b4")
        self.lbl_bbox.configure(text="Target: None")
        self._update_mode_ui()

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.configure(
            text="Resume" if self.is_paused else "Pause",
            image=self.icon_play if self.is_paused else self.icon_pause,
        )

    def _on_space_key(self):
        if self.mode == "CSRT Manual" and not self.csrt_tracker.is_tracking:
            self._prompt_select()
        else:
            self._toggle_pause()

    def _prompt_select(self):
        if self.mode == "CSRT Manual":
            self._set_status("CLICK & DRAG ON OBJECT", "#b45309", "#fef3c7")

    # -------------------------------------------------------------------------
    # Mouse Canvas Interaction (CSRT Drag Selection)
    # -------------------------------------------------------------------------
    def _canvas_to_frame_coords(self, cx, cy):
        if self.last_clean_frame is None:
            return None, None
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        fh, fw = self.last_clean_frame.shape[:2]

        if cw <= 0 or ch <= 0 or fw <= 0 or fh <= 0:
            return None, None

        scale = min(cw / fw, ch / fh)
        dw = int(fw * scale)
        dh = int(fh * scale)
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2

        fx = int((cx - ox) / scale)
        fy = int((cy - oy) / scale)
        return np.clip(fx, 0, fw - 1), np.clip(fy, 0, fh - 1)

    def _on_canvas_press(self, event):
        if self.mode != "CSRT Manual":
            return
        fx, fy = self._canvas_to_frame_coords(event.x, event.y)
        if fx is not None and fy is not None:
            self.drag_start = (fx, fy)
            self.drag_current = (fx, fy)
            self._set_status("DRAGGING TARGET BOX...", "#b45309", "#fef3c7")

    def _on_canvas_drag(self, event):
        if self.mode != "CSRT Manual":
            return
        if self.drag_start is not None:
            fx, fy = self._canvas_to_frame_coords(event.x, event.y)
            if fx is not None and fy is not None:
                self.drag_current = (fx, fy)

    def _on_canvas_release(self, event):
        if self.mode != "CSRT Manual":
            return
        if self.drag_start is None or self.last_clean_frame is None:
            self.drag_start = None
            self.drag_current = None
            return

        x1, y1 = self.drag_start
        x2, y2 = self._canvas_to_frame_coords(event.x, event.y)
        self.drag_start = None
        self.drag_current = None

        if x2 is None or y2 is None:
            return

        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        if rw >= 12 and rh >= 12:
            bbox = (int(rx), int(ry), int(rw), int(rh))
            with self._track_lock:
                success = self.csrt_tracker.init(self.last_clean_frame, bbox)
                if success:
                    self._worker_result = TrackingResult(
                        success=True,
                        bbox=bbox,
                        confidence=1.0,
                        center=(rx + rw / 2.0, ry + rh / 2.0),
                        status="CSRT TRACKING ACTIVE",
                    )
                    self._set_status("CSRT TRACKING ACTIVE", "#065f46", "#6ee7b7")
                else:
                    self._set_status("INIT FAILED - TRY AGAIN", "#7f1d1d", "#fca5a5")
        else:
            self._set_status("BOX TOO SMALL - TRY AGAIN", "#7f1d1d", "#fca5a5")

    # -------------------------------------------------------------------------
    # Main Processing Loop
    # -------------------------------------------------------------------------
    def _process_frame(self):
        if not self.is_running:
            return

        t_now = time.monotonic()
        dt = max(1e-3, t_now - self.prev_time)
        self.prev_time = t_now

        if dt > 0:
            self.fps_tracker.append(1.0 / dt)

        # Update FPS label at 4 Hz
        if (t_now - self.last_telemetry_time) > 0.25:
            self.last_telemetry_time = t_now
            if self.fps_tracker:
                mean_fps = sum(self.fps_tracker) / len(self.fps_tracker)
                self.lbl_fps.configure(text=f"FPS: {mean_fps:.1f}")

        if not self.is_paused and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.var_mirror.get():
                    frame = cv2.flip(frame, 1)

                self.last_clean_frame = frame.copy()
                fh, fw = frame.shape[:2]

                # Dispatch frame to tracker worker if available
                with self._track_lock:
                    if not self._worker_busy:
                        self._worker_frame = frame.copy()
                    result = self._worker_result
                    if self.mode == "YOLO Auto" and self.yolo_tracker is not None:
                        trajectory = self.yolo_tracker.trajectory
                    else:
                        trajectory = self.csrt_tracker.trajectory

                # Feed Detection to Pan-Tilt Servoing Subsystem at full video rate (30-60 FPS)
                pan_ang, tilt_ang = self.servo.update(
                    center=result.center if result.success else None,
                    frame_w=fw,
                    frame_h=fh,
                    dt=dt,
                )
                self._update_servo_telemetry(pan_ang, tilt_ang)

                # Update Status and Telemetry
                if result.success:
                    x, y, w, h = result.bbox
                    self.score_bar.set(result.confidence)
                    self.score_val_lbl.configure(
                        text=f"{int(result.confidence * 100)}%",
                        text_color="#00d285" if result.confidence > 0.4 else "#f59e0b",
                    )
                    self.lbl_bbox.configure(text=f"[{x}, {y}, {w}, {h}]")

                    if self.mode == "YOLO Auto":
                        self._set_status(result.status, "#065f46", "#6ee7b7")
                        self._draw_yolo_overlay(frame, x, y, w, h, result.confidence, trajectory)
                    else:
                        if result.is_recovered:
                            self._set_status("RECOVERED TARGET", "#0284c7", "#e0f2fe")
                            self._draw_csrt_overlay(frame, x, y, w, h, color=(0, 240, 255))
                        else:
                            self._set_status("CSRT TRACKING ACTIVE", "#065f46", "#6ee7b7")
                            self._draw_csrt_overlay(frame, x, y, w, h, color=(0, 230, 118))
                elif result.is_recovering:
                    self._set_status("RECOVERING FROM MOTION...", "#b45309", "#fef3c7")
                    if result.bbox is not None:
                        bx, by, bw, bh = result.bbox
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)
                else:
                    if self.mode == "YOLO Auto":
                        self.score_bar.set(0.0)
                        self.score_val_lbl.configure(text="0%", text_color="#9aa0b4")
                        self._set_status("YOLO SCANNING FOR DRONE...", "#1e3a8a", "#93c5fd")
                    elif self.csrt_tracker.is_tracking:
                        self.score_bar.set(0.0)
                        self.score_val_lbl.configure(text="0%", text_color="#ef4444")
                        self._set_status("TARGET LOST - RE-SELECT", "#7f1d1d", "#fca5a5")

                # Draw optical boresight center crosshair
                if self.var_boresight.get():
                    self._draw_boresight(frame, fw, fh)

                self.last_frame = frame
            else:
                if not self.source_desc.startswith("Webcam"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Render to canvas
        if self.last_frame is not None:
            self._render_to_canvas(self.last_frame)
        elif self.cap is None or not self.cap.isOpened():
            self._render_placeholder()

        if self.is_running:
            self.root.after(16, self._process_frame)

    def _draw_boresight(self, frame, fw, fh):
        cx, cy = fw // 2, fh // 2
        color = (120, 120, 140)
        cv2.line(frame, (cx - 15, cy), (cx - 4, cy), color, 1)
        cv2.line(frame, (cx + 4, cy), (cx + 15, cy), color, 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy - 4), color, 1)
        cv2.line(frame, (cx, cy + 4), (cx, cy + 15), color, 1)
        cv2.circle(frame, (cx, cy), 2, (180, 180, 200), -1)

    def _draw_yolo_overlay(self, frame, x, y, w, h, conf, trajectory):
        """Aesthetic YOLO Drone detection overlay."""
        color = (255, 180, 0)  # Vibrant cyan/blue in BGR
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Subtle target crosshair
        cx, cy = x + w // 2, y + h // 2
        cv2.line(frame, (cx - 6, cy), (cx + 6, cy), color, 1)
        cv2.line(frame, (cx, cy - 6), (cx, cy + 6), color, 1)

        # Label tag badge
        label_text = f"BLUE DRONE [{int(conf * 100)}%]"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(18, y - 6)
        cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + 2), (20, 24, 32), -1)
        cv2.rectangle(frame, (x, tag_y - th - 4), (x + tw + 8, tag_y + 2), color, 1)
        cv2.putText(frame, label_text, (x + 4, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Trajectory trail
        if self.var_trajectory.get() and len(trajectory) > 1:
            pts = list(trajectory)
            for i in range(1, len(pts)):
                alpha = i / float(len(pts))
                thick_t = max(1, int(2.5 * alpha))
                color_t = (int(255 * alpha), int(180 * alpha), int(0 * alpha))
                cv2.line(frame, pts[i - 1], pts[i], color_t, thick_t)
            cv2.circle(frame, pts[-1], 3, (255, 220, 0), -1)

    def _draw_csrt_overlay(self, frame, x, y, w, h, color=(0, 230, 118)):
        reliability_heatmap = None
        if self.var_spatial_map.get() or self.var_pip_map.get():
            reliability_heatmap, _ = compute_spatial_reliability_map(frame, (x, y, w, h))

        if self.var_spatial_map.get() and reliability_heatmap is not None:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            roi_h, roi_w = y2 - y1, x2 - x1
            if roi_w > 0 and roi_h > 0:
                rh_crop = cv2.resize(reliability_heatmap, (roi_w, roi_h))
                sub_roi = frame[y1:y2, x1:x2]
                blended = cv2.addWeighted(rh_crop, 0.45, sub_roi, 0.55, 0)
                frame[y1:y2, x1:x2] = blended

        if self.var_pip_map.get() and reliability_heatmap is not None:
            pip_size = 110
            pip_img = cv2.resize(reliability_heatmap, (pip_size, pip_size))
            fh, fw = frame.shape[:2]
            px1 = fw - pip_size - 16
            py1 = 16
            px2 = px1 + pip_size
            py2 = py1 + pip_size

            if px1 > 0 and py1 > 0:
                cv2.rectangle(frame, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (28, 30, 36), -1)
                cv2.rectangle(frame, (px1 - 1, py1 - 1), (px2 + 1, py2 + 1), (0, 210, 133), 1)
                frame[py1:py2, px1:px2] = pip_img
                cv2.putText(
                    frame,
                    "SPATIAL MAP",
                    (px1 + 4, py1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (0, 210, 133),
                    1,
                )

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        corner_len = max(8, min(w, h) // 5)
        thick = 2
        cv2.line(frame, (x, y), (x + corner_len, y), color, thick)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thick)
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thick)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thick)
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thick)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thick)
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thick)

        cx, cy = x + w // 2, y + h // 2
        cv2.line(frame, (cx - 4, cy), (cx + 4, cy), color, 1)
        cv2.line(frame, (cx, cy - 4), (cx, cy + 4), color, 1)

        if self.var_trajectory.get() and len(self.csrt_tracker.trajectory) > 1:
            pts = list(self.csrt_tracker.trajectory)
            for i in range(1, len(pts)):
                alpha = i / float(len(pts))
                thick_t = max(1, int(2.5 * alpha))
                color_t = (int(0 * alpha), int(210 * alpha), int(255 * alpha))
                cv2.line(frame, pts[i - 1], pts[i], color_t, thick_t)
            cv2.circle(frame, pts[-1], 3, (0, 255, 200), -1)

    def _render_placeholder(self):
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw > 10 and ch > 10:
            self.video_canvas.delete("all")
            self.canvas_img_id = None
            self.video_canvas.create_text(
                cw // 2,
                ch // 2,
                text="No video feed available.\nClick 'Webcam' or 'Video File' to begin.",
                fill="#6c7285",
                font=("Inter, -apple-system, sans-serif", 14),
                justify="center",
            )

    def _render_to_canvas(self, frame):
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        dw = int(fw * scale)
        dh = int(fh * scale)
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2

        display_frame = frame.copy()
        if self.mode == "CSRT Manual" and self.drag_start and self.drag_current:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 210, 255), 2)

        resized = cv2.resize(display_frame, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)

        if self.canvas_img_id is None:
            self.video_canvas.delete("all")
            self.canvas_img_id = self.video_canvas.create_image(ox, oy, anchor="nw", image=self.tk_image)
        else:
            self.video_canvas.coords(self.canvas_img_id, ox, oy)
            self.video_canvas.itemconfig(self.canvas_img_id, image=self.tk_image)

    def _on_close(self):
        self.is_running = False
        self.is_paused = True
        if self.cap is not None:
            self.cap.release()
        self.servo.release()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        sys.exit(0)
