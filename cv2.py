# =============================================================================
# FITFORM AI - MULTI-EXERCISE FORM DETECTION (ULTRA-ADVANCED UI & DASHBOARD)
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import warnings
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL VARIABLES AND INITIALIZATIONS
# =============================================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
EXERCISES = {
    "Bicep Curls": "bicep",
    "Squats": "squat",
    "Lateral Raises": "lateral"
}
EXERCISE_DESCRIPTIONS = {
    "Bicep Curls": "STRENGTH PROTOCOL: Optimize brachialis engagement and control elbow flexion.",
    "Squats": "POWER PROTOCOL: Analyze hip, knee, and back angles for deep, safe mobility.",
    "Lateral Raises": "ENDURANCE PROTOCOL: Monitor deltoid isolation and minimize body swing."
}

# --- UI Theme Colors ---
ACCENT_COLOR = "#00BFFF"  # Deep Sky Blue / Aqua for futuristic accents
ERROR_COLOR = "#FF4136"  # Red for warnings
WARN_COLOR = "#FF851B"  # Orange for caution
SUCCESS_COLOR = "#00FF7F"  # Spring Green for top scores
PRIMARY_FG = "#1E1E1E"  # Darker background for frames
SECONDARY_FG = "#2C2C2C"  # Even darker background
TEXT_COLOR = "#FFFFFF"


# =============================================================================
# MODEL AND FEATURE FUNCTIONS (MUST BE PRESENT)
# =============================================================================

def generate_synthetic_bicep_data(num_samples=1000):
    data = []
    labels = []
    for _ in range(num_samples):
        form_type = np.random.choice(['perfect', 'almost', 'needs_practice'])
        if form_type == 'perfect':
            elbow_angle = np.random.uniform(30, 60)
            upper_arm_swing = np.random.uniform(5, 15)
            label = 0
        elif form_type == 'almost':
            elbow_angle = np.random.uniform(60, 90)
            upper_arm_swing = np.random.uniform(15, 30)
            label = 1
        else:
            elbow_angle = np.random.uniform(90, 120)
            upper_arm_swing = np.random.uniform(30, 50)
            label = 2
        features = [elbow_angle, upper_arm_swing]
        data.append(features)
        labels.append(label)
    return np.array(data), np.array(labels)


def generate_synthetic_squat_data(num_samples=1000):
    data = []
    labels = []
    for _ in range(num_samples):
        form_type = np.random.choice(['perfect', 'almost', 'needs_practice'])
        if form_type == 'perfect':
            knee_angle = np.random.uniform(70, 90)
            hip_angle = np.random.uniform(40, 60)
            back_angle = np.random.uniform(-10, 10)
            knee_valgus = np.random.uniform(-0.02, 0.02)
            label = 0
        elif form_type == 'almost':
            knee_angle = np.random.uniform(100, 120)
            hip_angle = np.random.uniform(60, 80)
            back_angle = np.random.uniform(10, 25)
            knee_valgus = np.random.uniform(0.02, 0.05)
            label = 1
        else:
            knee_angle = np.random.uniform(130, 160)
            hip_angle = np.random.uniform(80, 100)
            back_angle = np.random.uniform(25, 45)
            knee_valgus = np.random.uniform(0.06, 0.12)
            label = 2
        features = [knee_angle, hip_angle, back_angle, knee_valgus]
        data.append(features)
        labels.append(label)
    return np.array(data), np.array(labels)


def generate_synthetic_lateral_data(num_samples=1000):
    data = []
    labels = []
    for _ in range(num_samples):
        form_type = np.random.choice(['perfect', 'almost', 'needs_practice'])
        if form_type == 'perfect':
            arm_angle = np.random.uniform(80, 100)
            body_swing = np.random.uniform(0, 5)
            label = 0
        elif form_type == 'almost':
            arm_angle = np.random.uniform(60, 80)
            body_swing = np.random.uniform(5, 15)
            label = 1
        else:
            arm_angle = np.random.uniform(30, 60)
            body_swing = np.random.uniform(15, 30)
            label = 2
        features = [arm_angle, body_swing]
        data.append(features)
        labels.append(label)
    return np.array(data), np.array(labels)


def train_bicep_model():
    print("--- Starting Bicep Model Training ---")
    X, y = generate_synthetic_bicep_data(num_samples=3000)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    model.save('bicep_curl_classifier.h5')
    joblib.dump(scaler, 'bicep_scaler.gz')
    print("--- Bicep Model Training Complete. Files saved. ---")
    return model, scaler


def train_squat_model():
    print("--- Starting Squat Model Training ---")
    X, y = generate_synthetic_squat_data(num_samples=3000)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    model.save('squat_form_classifier.h5')
    joblib.dump(scaler, 'squat_scaler.gz')
    print("--- Squat Model Training Complete. Files saved. ---")
    return model, scaler


def train_lateral_model():
    print("--- Starting Lateral Raises Model Training ---")
    X, y = generate_synthetic_lateral_data(num_samples=3000)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    model.save('lateral_raises_classifier.h5')
    joblib.dump(scaler, 'lateral_scaler.gz')
    print("--- Lateral Raises Model Training Complete. Files saved. ---")
    return model, scaler


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def extract_bicep_features(landmarks):
    try:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        upper_arm_angle = np.degrees(np.arctan2(elbow[1] - shoulder[1], elbow[0] - shoulder[0]))
        upper_arm_swing = abs(90 - upper_arm_angle)
        return np.array([[elbow_angle, upper_arm_swing]])
    except:
        return None


def extract_squat_features(landmarks):
    try:
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        back_angle = np.degrees(np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0])) - 90
        knee_valgus = knee[0] - (hip[0] + ankle[0]) / 2
        return np.array([[knee_angle, hip_angle, back_angle, knee_valgus]])
    except:
        return None


def extract_lateral_features(landmarks):
    try:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        arm_angle = (left_arm_angle + right_arm_angle) / 2
        body_swing = abs(left_shoulder[0] - right_shoulder[0] - 0.5) * 100
        return np.array([[arm_angle, body_swing]])
    except:
        return None


# --- ADVANCED REP COUNTER CLASSES (WITH PHASE DETECTION) ---
class BicepRepCounter:
    def __init__(self):
        self.state = "DOWN"
        self.count = 0
        self.phase = "RESTING"

    def update(self, elbow_angle):
        if elbow_angle < 60 and self.state == "DOWN":
            self.state = "UP"
            self.phase = "CONCENTRIC"
        elif elbow_angle > 160 and self.state == "UP":
            self.state = "DOWN"
            self.phase = "ECCENTRIC"
            self.count += 1
        elif 60 <= elbow_angle <= 160:
            if self.state == "UP":
                self.phase = "HOLD_TOP"
            else:
                self.phase = "HOLD_BOTTOM"


class SquatRepCounter:
    def __init__(self):
        self.state = "UP"
        self.count = 0
        self.phase = "RESTING"

    def update(self, knee_angle):
        if knee_angle > 160 and self.state == "UP":
            self.phase = "STANDBY"
        if knee_angle < 90 and self.state == "UP":
            self.state = "DOWN"
            self.phase = "DESCENDING"
        if knee_angle > 160 and self.state == "DOWN":
            self.state = "UP"
            self.phase = "ASCENDING"
            self.count += 1
        elif 90 <= knee_angle <= 160:
            if self.state == "DOWN":
                self.phase = "BOTTOM_HOLD"
            else:
                self.phase = "TRANSITION"


class LateralRepCounter:
    def __init__(self):
        self.state = "DOWN"
        self.count = 0
        self.phase = "RESTING"

    def update(self, arm_angle):
        if arm_angle > 90 and self.state == "DOWN":
            self.state = "UP"
            self.phase = "ASCENDING"
        elif arm_angle < 30 and self.state == "UP":
            self.state = "DOWN"
            self.phase = "DESCENDING"
            self.count += 1
        elif 30 <= arm_angle <= 90:
            if self.state == "UP":
                self.phase = "HOLD_TOP"
            else:
                self.phase = "TRANSITION"


def load_models():
    models = {}
    scalers = {}
    for ex in ['bicep', 'squat', 'lateral']:
        model_path = f'{ex}_curl_classifier.h5' if ex == 'bicep' else f'{ex}_form_classifier.h5' if ex == 'squat' else f'{ex}_raises_classifier.h5'
        scaler_path = f'{ex}_scaler.gz'

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"--- Training {ex.capitalize()} Model (Missing files) ---")
            if ex == 'bicep':
                models[ex], scalers[ex] = train_bicep_model()
            elif ex == 'squat':
                models[ex], scalers[ex] = train_squat_model()
            else:
                models[ex], scalers[ex] = train_lateral_model()
        else:
            print(f"--- Loading Pre-trained {ex.capitalize()} Model ---")
            try:
                models[ex] = tf.keras.models.load_model(model_path)
                scalers[ex] = joblib.load(scaler_path)
                print(f"--- {ex.capitalize()} Model Loaded. ---")
            except Exception as e:
                print(f"Error loading {ex} models: {e}. Retraining...")
                if ex == 'bicep':
                    models[ex], scalers[ex] = train_bicep_model()
                elif ex == 'squat':
                    models[ex], scalers[ex] = train_squat_model()
                else:
                    models[ex], scalers[ex] = train_lateral_model()
    return models, scalers


# =============================================================================
# MAIN APPLICATION CLASS (ULTRA-ADVANCED CUSTOMTKINTER UI)
# =============================================================================

class FitFormApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧬 FitForm AI - Protocol Hub")
        self.root.geometry("1000x650")
        self.root.resizable(True, True)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.accent_color = ACCENT_COLOR
        self.error_color = ERROR_COLOR
        self.warn_color = WARN_COLOR
        self.success_color = SUCCESS_COLOR
        self.primary_fg = PRIMARY_FG
        self.secondary_fg = SECONDARY_FG
        self.text_color = TEXT_COLOR

        self.models, self.scalers = load_models()

        # Performance Tracking
        self.history_data = []
        self.current_score_sum = 0
        self.current_score_count = 0
        self.best_session = {'score': 0, 'exercise': 'N/A', 'reps': 0, 'time': 'N/A'}  # New metric

        # Initialize variables
        self.current_exercise = None
        self.running = False
        self.cap = None
        self.pose = None
        self.rep_counter = None
        self.form_label_map = {0: "PERFECT", 1: "GOOD", 2: "POOR"}
        self.form_label_colors = {0: self.success_color, 1: self.warn_color, 2: self.error_color}
        self.current_image = None

        # UI Elements for Protocol Screen
        self.selected_exercise_name = ctk.StringVar(value="")
        self.card_widgets = {}

        self.create_protocol_screen()

    def get_score_color(self, score):
        """Returns a color based on the score value."""
        if score >= 90:
            return self.success_color
        elif score >= 70:
            return self.accent_color
        elif score >= 50:
            return self.warn_color
        else:
            return self.error_color

    def create_protocol_screen(self):
        self.root.title("🧬 FitForm AI - Protocol Hub")
        for widget in self.root.winfo_children():
            widget.destroy()

        self.start_frame = ctk.CTkFrame(self.root, fg_color=self.secondary_fg, corner_radius=0)
        self.start_frame.pack(fill="both", expand=True)
        self.start_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.start_frame.grid_rowconfigure((0, 1, 2), weight=1)

        # --- Header ---
        ctk.CTkLabel(self.start_frame, text="FITFORM AI PROTOCOL",
                     font=ctk.CTkFont(size=36, weight="bold"), text_color=self.accent_color).grid(row=0, column=0,
                                                                                                  columnspan=3,
                                                                                                  pady=(50, 0),
                                                                                                  sticky="s")
        ctk.CTkLabel(self.start_frame, text="CHOOSE YOUR TRAINING PROTOCOL",
                     font=ctk.CTkFont(size=16, weight="normal"), text_color="#AAAAAA").grid(row=1, column=0,
                                                                                            columnspan=3, pady=(0, 30),
                                                                                            sticky="n")

        # --- Exercise Cards ---
        exercises = list(EXERCISES.keys())
        for i, ex_name in enumerate(exercises):
            self._create_exercise_card(self.start_frame, ex_name, EXERCISE_DESCRIPTIONS[ex_name], i)

        # --- Control Buttons ---
        button_frame = ctk.CTkFrame(self.start_frame, fg_color="transparent")
        button_frame.grid(row=2, column=0, columnspan=3, pady=(0, 40), sticky="n")

        self.start_workout_btn = ctk.CTkButton(button_frame, text="INITIATE WORKOUT [>]",
                                               command=self.transition_to_live_session,
                                               fg_color=self.accent_color, hover_color="#00AEEF",
                                               text_color=self.secondary_fg, corner_radius=8,
                                               font=ctk.CTkFont(size=18, weight="bold"),
                                               state="disabled")
        self.start_workout_btn.pack(side="left", padx=20)

        self.dashboard_btn = ctk.CTkButton(button_frame, text="VIEW PERFORMANCE DASHBOARD",
                                           command=lambda: self.transition_to_live_session(dashboard_only=True),
                                           fg_color="transparent", border_width=2, border_color="#555555",
                                           font=ctk.CTkFont(size=18, weight="bold"))
        self.dashboard_btn.pack(side="left", padx=20)

        if exercises:
            self._select_card(exercises[0])

    def _create_exercise_card(self, parent, ex_name, description, column):
        card_frame = ctk.CTkFrame(parent, width=280, height=300, corner_radius=12,
                                  fg_color=self.primary_fg, border_width=2, border_color=self.secondary_fg)
        card_frame.grid(row=1, column=column, padx=20, pady=20, sticky="nsew")
        card_frame.grid_propagate(False)

        content_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        content_frame.pack(padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(content_frame, text="⚡", font=ctk.CTkFont(size=50), text_color=self.accent_color).pack(
            pady=(10, 5))

        ctk.CTkLabel(content_frame, text=ex_name.upper(),
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=5)

        ctk.CTkLabel(content_frame, text=description,
                     font=ctk.CTkFont(size=12), wraplength=240, justify="center", text_color="#AAAAAA").pack(pady=10)

        card_frame.bind("<Button-1>", lambda event, name=ex_name: self._select_card(name))
        card_frame.bind("<Enter>", lambda event, frame=card_frame: frame.configure(border_color=self.accent_color))
        card_frame.bind("<Leave>", lambda event, name=ex_name, frame=card_frame: frame.configure(
            border_color=self.accent_color if self.selected_exercise_name.get() == name else self.secondary_fg))

        self.card_widgets[ex_name] = card_frame

    def _select_card(self, ex_name):
        self.selected_exercise_name.set(ex_name)
        self.start_workout_btn.configure(state="normal")

        for name, card in self.card_widgets.items():
            if name == ex_name:
                card.configure(border_color=self.accent_color, border_width=3)
            else:
                card.configure(border_color=self.secondary_fg, border_width=2)

    def transition_to_live_session(self, dashboard_only=False):
        self.start_frame.destroy()
        self.root.geometry("1200x650")
        self.root.title("💪 FitForm AI - Live Session")

        self.tab_view = ctk.CTkTabview(self.root, width=1160, height=610, corner_radius=10, fg_color=self.primary_fg)
        self.tab_view.pack(padx=20, pady=20, fill="both", expand=True)

        self.live_tab = self.tab_view.add("🎥 Live Session")
        self.dashboard_tab = self.tab_view.add("📊 Performance Dashboard")

        self.create_live_session_widgets(self.live_tab)
        self.create_dashboard_widgets(self.dashboard_tab)

        if dashboard_only:
            self.tab_view.set("📊 Performance Dashboard")
        else:
            self.exercise_var.set(self.selected_exercise_name.get())
            self.tab_view.set("🎥 Live Session")

    def create_live_session_widgets(self, live_tab):
        live_tab.grid_columnconfigure(0, weight=4)  # WIDER CAMERA RATIO
        live_tab.grid_columnconfigure(1, weight=1)
        live_tab.grid_rowconfigure(0, weight=1)

        # --- Left Frame (Video Feed) ---
        self.video_frame = ctk.CTkFrame(live_tab, corner_radius=10, fg_color=self.secondary_fg)
        self.video_frame.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="nsew")

        self.video_label = ctk.CTkLabel(self.video_frame, text="Awaiting Protocol Initiation...",
                                        font=ctk.CTkFont(size=20), text_color="#AAAAAA")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Right Frame (Controls & Metrics) ---
        self.controls_frame = ctk.CTkFrame(live_tab, corner_radius=10, fg_color=self.primary_fg)
        self.controls_frame.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_rowconfigure(7, weight=1)  # Spacer row

        title_label = ctk.CTkLabel(self.controls_frame, text="LIVE ANALYSIS",
                                   font=ctk.CTkFont(size=26, weight="bold"), text_color=self.accent_color)
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        exercise_label = ctk.CTkLabel(self.controls_frame, text="PROTOCOL ACTIVE:",
                                      font=ctk.CTkFont(size=14, weight="bold"), text_color="#AAAAAA")
        exercise_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")

        self.exercise_var = ctk.StringVar(value=self.selected_exercise_name.get())
        self.exercise_menu = ctk.CTkOptionMenu(self.controls_frame,
                                               values=list(EXERCISES.keys()),
                                               variable=self.exercise_var,
                                               font=ctk.CTkFont(size=14), state="disabled")
        self.exercise_menu.grid(row=2, column=0, padx=20, pady=5, sticky="ew")

        # Start/Stop Buttons
        btn_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        btn_frame.grid(row=3, column=0, padx=20, pady=15, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.start_button = ctk.CTkButton(btn_frame, text="▶ START SESSION",
                                          command=self.start_exercise,
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          fg_color=self.accent_color, hover_color="#00AEEF",
                                          text_color=self.secondary_fg)
        self.start_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.stop_button = ctk.CTkButton(btn_frame, text="■ END SESSION",
                                         command=self.stop_exercise,
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         fg_color=self.error_color, hover_color="#C82828", state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # --- Metric Panel ---
        metric_panel = ctk.CTkFrame(self.controls_frame, corner_radius=8, fg_color=self.secondary_fg)
        metric_panel.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        metric_panel.grid_columnconfigure((0, 1), weight=1)

        # 1. Score Metric
        self.score_display = self._create_metric_widget(metric_panel, 0, "AVG SCORE", "N/A", self.accent_color)

        # 2. Reps Metric
        self.reps_display = self._create_metric_widget(metric_panel, 1, "REPS COUNT", "0", self.accent_color)

        # --- Advanced Status Panel ---
        self.status_panel = ctk.CTkFrame(self.controls_frame, corner_radius=8, fg_color=self.secondary_fg)
        self.status_panel.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.status_panel.columnconfigure((0, 1), weight=1)

        # 3. Form Status
        self.form_label = self._create_metric_widget(self.status_panel, 0, "FORM RATING", "READY", self.warn_color)
        # 4. Rep Phase
        self.phase_label = self._create_metric_widget(self.status_panel, 1, "REP PHASE", "IDLE", self.accent_color)

        # Detailed Feedback Text
        feedback_title = ctk.CTkLabel(self.controls_frame, text="NEURAL FEEDBACK LOG:",
                                      font=ctk.CTkFont(size=14, weight="bold"), text_color="#AAAAAA")
        feedback_title.grid(row=6, column=0, padx=20, pady=(10, 5), sticky="sw")

        self.feedback_text = ctk.CTkLabel(self.controls_frame, text="Protocol Initialized. Ready for biometric input.",
                                          font=ctk.CTkFont(size=16), wraplength=250, justify="left")
        self.feedback_text.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="nw")

    def _create_metric_widget(self, parent, column, title, initial_value, color):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=column, padx=5, pady=10, sticky="nsew")

        title_label = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#999999")
        title_label.pack(pady=(0, 2))

        value_label = ctk.CTkLabel(frame, text=initial_value, font=ctk.CTkFont(size=24, weight="bold"),
                                   text_color=color)
        value_label.pack()
        return value_label

    def create_dashboard_widgets(self, dashboard_tab):
        dashboard_tab.grid_columnconfigure(0, weight=1)
        dashboard_tab.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(dashboard_tab, text="PERFORMANCE DATA MATRIX", font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=self.accent_color).grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # --- 1. Overview Metrics Frame ---
        metrics_frame = ctk.CTkFrame(dashboard_tab, fg_color=self.secondary_fg, corner_radius=10)
        metrics_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="new")
        metrics_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.total_sessions_label = self._create_dashboard_metric(metrics_frame, 0, "TOTAL SESSIONS", "0")
        self.total_reps_label = self._create_dashboard_metric(metrics_frame, 1, "TOTAL REPS LOGGED", "0")
        self.avg_score_label = self._create_dashboard_metric(metrics_frame, 2, "AVERAGE SCORE", "N/A")
        self.best_session_label = self._create_dashboard_metric(metrics_frame, 3, "BEST SESSION SCORE",
                                                                "N/A")  # Updated metric

        # --- 2. Progress Chart Frame ---
        chart_frame = ctk.CTkFrame(dashboard_tab, fg_color=self.secondary_fg, corner_radius=10)
        chart_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        chart_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(chart_frame, text="PERFORMANCE TREND ANALYSIS", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        # Simple Text/Visual Placeholder for Chart
        chart_placeholder = ctk.CTkLabel(chart_frame,
                                         text="[DATA VISUALIZATION: Mock 7-Day Performance Trend]\n\nDay 1: 75% | Day 2: 78% | Day 3: 82% | Day 4: 80% | Day 5: 85% (New High) | Day 6: 83% | Day 7: 88%\n\n\u25b2 System detects a 13% score increase over the last week. Trend is positive. ",
                                         font=ctk.CTkFont(family="Courier", size=14), justify="left",
                                         fg_color=self.primary_fg, corner_radius=8, padx=10, pady=10)
        chart_placeholder.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")

        # --- 3. History Log Frame ---
        ctk.CTkLabel(dashboard_tab, text="SESSION HISTORY LOG", font=ctk.CTkFont(size=18, weight="bold")).grid(row=3,
                                                                                                               column=0,
                                                                                                               padx=20,
                                                                                                               pady=(
                                                                                                                   20,
                                                                                                                   10),
                                                                                                               sticky="w")

        self.history_text = ctk.CTkTextbox(dashboard_tab, height=200, width=800, state="disabled",
                                           fg_color=self.secondary_fg)
        self.history_text.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.update_dashboard()

    def _create_dashboard_metric(self, parent, column, title, initial_value):
        frame = ctk.CTkFrame(parent, corner_radius=8, fg_color=self.primary_fg)
        frame.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")

        title_label = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color="#999999")
        title_label.pack(pady=(10, 0))

        value_label = ctk.CTkLabel(frame, text=initial_value, font=ctk.CTkFont(size=36, weight="bold"),
                                   text_color=self.accent_color)
        value_label.pack(pady=(0, 10))
        return value_label

    def update_dashboard(self):
        total_sessions = len(self.history_data)
        total_reps = sum(d.get('reps', 0) for d in self.history_data)

        all_scores = [d['score'] for d in self.history_data if d.get('score') is not None and d['score'] != 'N/A']
        avg_score = round(np.mean(all_scores), 1) if all_scores else "N/A"

        # Update Best Session Metric
        for d in self.history_data:
            if d.get('score') is not None and d['score'] != 'N/A' and d['score'] > self.best_session['score']:
                self.best_session = {'score': d['score'], 'exercise': d['exercise'], 'reps': d['reps'],
                                     'time': d['time']}

        self.total_sessions_label.configure(text=str(total_sessions))
        self.total_reps_label.configure(text=str(total_reps))
        self.avg_score_label.configure(text=str(avg_score),
                                       text_color=self.get_score_color(avg_score if avg_score != 'N/A' else 0))

        best_text = f"{self.best_session['score']}% ({self.best_session['exercise']})" if self.best_session[
                                                                                              'score'] > 0 else "N/A"
        self.best_session_label.configure(text=best_text, text_color=self.get_score_color(self.best_session['score']))

        # Update History Log
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        if self.history_data:
            log_text = "\n".join([
                f"[{d['time'].split(' ')[1]}] {d['exercise']} | Reps: {d['reps']} | Score: {d['score'] if d['score'] is not None else 'N/A'}/100 | Status: {d['status']}"
                for d in reversed(self.history_data)
            ])
            self.history_text.insert("end", log_text)
        else:
            self.history_text.insert("end", ">> No operational data logged yet.")
        self.history_text.configure(state="disabled")

    # --- Live Session Logic ---

    def _update_gui_labels(self, rep_count, score, form_label, form_color, phase_label, phase_color, detail_feedback):
        if not self.running or not self.root.winfo_exists():
            return

        self.reps_display.configure(text=str(rep_count))
        self.score_display.configure(text=f"{score}/100", text_color=form_color)
        self.form_label.configure(text=form_label, text_color=form_color)
        self.phase_label.configure(text=phase_label, text_color=phase_color)
        self.feedback_text.configure(text=detail_feedback)

    def update_feedback(self, rep_count, form_label_index, detail_feedback="", current_score=None, rep_phase="IDLE"):
        if not self.running:
            return

        form_label = self.form_label_map.get(form_label_index, "DETECTING")
        form_color = self.form_label_colors.get(form_label_index, '#FFFFFF')
        score = int(current_score) if current_score is not None else "N/A"

        # Phase Color Logic
        if "HOLD" in rep_phase or "RESTING" in rep_phase or "STANDBY" in rep_phase:
            phase_color = self.warn_color
        elif "ASCENDING" in rep_phase or "CONCENTRIC" in rep_phase:
            phase_color = self.success_color
        else:
            phase_color = self.accent_color

        # Color the score display based on the score value itself
        score_color = self.get_score_color(score if score != 'N/A' else 0)

        if current_score is not None and form_label_index in [0, 1, 2]:
            self.current_score_sum += current_score
            self.current_score_count += 1

        self.root.after(10, lambda: self._update_gui_labels(rep_count, score, form_label, score_color, rep_phase,
                                                            phase_color, detail_feedback))

    def start_exercise(self):
        exercise_name = self.exercise_var.get()
        self.current_exercise = EXERCISES[exercise_name]

        self.current_score_sum = 0
        self.current_score_count = 0

        self.reps_display.configure(text="0")
        self.score_display.configure(text="N/A", text_color=self.accent_color)
        self.form_label.configure(text="SYNCING", text_color=self.warn_color)
        self.phase_label.configure(text="IDLE", text_color=self.accent_color)
        self.video_label.configure(text="Establishing Biometric Feed...")

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        # Initialize rep counter
        if self.current_exercise == "bicep":
            self.rep_counter = BicepRepCounter()
            instruction = "PROTOCOL: BICEP CURLS. Right Arm Focus. Wait for RESTING phase before starting."
        elif self.current_exercise == "squat":
            self.rep_counter = SquatRepCounter()
            instruction = "PROTOCOL: SQUATS. Maintain Upright Torso. Ensure full hip/knee break."
        else:
            self.rep_counter = LateralRepCounter()
            instruction = "PROTOCOL: LATERAL RAISES. Keep arms straight. Strict form, no body momentum."

        self.feedback_text.configure(text=instruction)

        self.running = True
        self.camera_thread = threading.Thread(target=self.run_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def stop_exercise(self):
        self.running = False

        total_reps = self.rep_counter.count if self.rep_counter else 0
        avg_score = round(self.current_score_sum / self.current_score_count) if self.current_score_count > 0 else None
        exercise_name = self.exercise_var.get()

        self.history_data.append({
            'time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'exercise': exercise_name,
            'reps': total_reps,
            'score': avg_score,
            'status': "COMPLETED" if total_reps > 0 and avg_score is not None else "INTERRUPTED"
        })
        self.update_dashboard()

        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

        final_status_text = "SESSION TERMINATED." if avg_score is None else "PROTOCOL COMPLETE."

        self.form_label.configure(text="ENDED", text_color=self.accent_color)
        self.phase_label.configure(text="---", text_color="#AAAAAA")
        final_score_color = self.get_score_color(avg_score if avg_score is not None else 0)
        self.score_display.configure(text=f"{avg_score}/100" if avg_score is not None else "N/A",
                                     text_color=final_score_color)
        self.feedback_text.configure(
            text=f"{final_status_text} Reps: {total_reps}. Final Score: {avg_score if avg_score is not None else 'N/A'}. Proceed to Dashboard tab.")

        self.video_label.configure(text=f"[{final_status_text}] Data Logged. Access Dashboard for analysis.",
                                   image=None)
        self.current_image = None

    def on_closing(self):
        self.stop_exercise()
        self.root.destroy()

    def _update_video_frame(self, ctk_image):
        if self.running:
            self.video_label.configure(image=ctk_image, text="")

    # ========================================================================
    # UPDATED: ROBUST CAMERA RUN METHOD
    # ========================================================================
    def run_camera(self):
        # --- Robust Camera Initialization ---
        self.cap = None
        camera_index = -1

        # Try to find a working camera index
        for i in range(3):  # Check indices 0, 1, and 2
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                temp_cap.release()  # Release it immediately
                camera_index = i
                print(f"Camera found at index {i}")
                break

        if camera_index == -1:
            # No camera found after checking all indices
            self.root.after(0, lambda: self.video_label.configure(
                text="ERROR: No camera found. Please check connection and close other apps.",
                text_color=self.error_color))
            self.root.after(0, self.stop_exercise)  # Gracefully stop the session
            return  # Exit the thread

        # --- If camera is found, proceed with initialization ---
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set a reasonable FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.root.update_idletasks()
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()

        while self.cap.isOpened() and self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # Handle end of stream or read error
                    print("Failed to grab frame from camera.")
                    self.root.after(0, lambda: self.video_label.configure(
                        text="ERROR: Lost camera feed. Check connection.",
                        text_color=self.error_color))
                    self.root.after(100, self.stop_exercise)  # Add a small delay before stopping
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                form_label_index = -1
                detail_feedback = ""
                current_score = None
                rep_phase = self.rep_counter.phase if self.rep_counter else "IDLE"

                try:
                    landmarks = results.pose_landmarks.landmark
                    features = None
                    angle = None

                    # --- Feature Extraction & Rep Counting ---
                    if self.current_exercise == "bicep":
                        features = extract_bicep_features(landmarks)
                        angle = calculate_angle([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                                                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                                                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
                        self.rep_counter.update(angle)

                    elif self.current_exercise == "squat":
                        features = extract_squat_features(landmarks)
                        angle = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
                        self.rep_counter.update(angle)

                    else:  # lateral
                        features = extract_lateral_features(landmarks)
                        angle = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                                                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
                        self.rep_counter.update(angle)

                    rep_phase = self.rep_counter.phase  # Get the updated phase

                    if features is not None:
                        features_scaled = self.scalers[self.current_exercise].transform(features)
                        prediction = self.models[self.current_exercise].predict(features_scaled, verbose=0)
                        form_label_index = np.argmax(prediction)

                        current_score = (100 - (form_label_index * 35)) if form_label_index < 2 else 30

                        # --- Detailed Feedback Logic ---
                        if form_label_index == 2:
                            detail_feedback = f"🛑 CRITICAL: {rep_phase} Phase Error. Check form immediately!"
                        elif form_label_index == 1:
                            detail_feedback = f"⚠️ WARNING: Sub-optimal form during {rep_phase}. Enhance control."
                        else:
                            detail_feedback = f"✅ OPTIMAL ALIGNMENT: Executing {rep_phase} perfectly."

                        self.update_feedback(self.rep_counter.count, form_label_index, detail_feedback, current_score,
                                             rep_phase)

                except Exception as e:
                    # Fallback for when landmarks are lost
                    rep_phase = "POSE LOST"
                    detail_feedback = "❌ BIOMETRIC ACQUISITION FAILURE: Re-establish full-body visibility."
                    self.update_feedback(self.rep_counter.count if self.rep_counter else 0, -1,
                                         detail_feedback, current_score=0, rep_phase=rep_phase)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                if target_w < 10 or target_h < 10:
                    target_w = self.video_label.winfo_width()
                    target_h = self.video_label.winfo_height()
                    if target_w < 10 or target_h < 10:
                        target_w, target_h = 800, 600

                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image,
                                         size=(target_w, target_h))

                self.root.after(5, self._update_video_frame, ctk_image)

            except Exception as e:
                print(f"An error occurred in the camera loop: {e}")
                self.root.after(0, lambda: self.video_label.configure(
                    text=f"ERROR: {e}",
                    text_color=self.error_color))
                self.root.after(100, self.stop_exercise)
                break

        # Ensure resources are released when the loop ends
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        print("Camera thread finished.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if 'main_app_running' not in locals():
        main_app_running = True
        try:
            root = ctk.CTk()
            app = FitFormApp(root)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()
        except Exception as e:
            print(f"An error occurred during application startup or runtime: {e}")
            if 'app' in locals():
                try:
                    app.on_closing()
                except:
                    pass