import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import json
import os
import sys

from collections import deque

from winotify import Notification
import pystray
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURAZIONE
# ============================================================

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pose_landmarker_full.task"
)
CALIBRATIONS_DIR = "calibrations"

CAMERA_INDEX = 0

MOVING_AVERAGE_SIZE = 30

CALIBRATION_POSITIONING_DURATION = 5.0
CALIBRATION_PHASE_DURATION = 10.0

BAD_POSTURE_DURATION = 10.0
ALARM_DURATION = 3.0

PROFILE_THRESHOLD = 0.10


# ============================================================
# COLORI UI
# ============================================================

WHITE = (255, 255, 255)
CYAN = (255, 230, 80)
LIGHT_CYAN = (255, 255, 180)

RED = (60, 60, 255)
GREEN = (80, 220, 100)
YELLOW = (80, 220, 255)
MAGENTA = (255, 80, 220)
BLUE = (255, 160, 60)

DARK_PANEL = (25, 25, 25)
DARK_BUTTON = (35, 35, 35)
DARK_BUTTON_HOVER = (55, 55, 55)


# ============================================================
# STATO GLOBALE
# ============================================================

latest_landmarks = None
landmarks_lock = threading.Lock()

running = True

# True quando la finestra è stata chiusa e l'app lavora in background
background_mode = False

# True quando la finestra OpenCV è attualmente disponibile
window_open = False

# Richiesta di mostrare nuovamente la finestra
show_window_requested = False

tray_icon = None

window_name = "PostureGuard"


# ============================================================
# STATO CALIBRAZIONE
# ============================================================

calibration_threshold = None
current_calibration_name = None

calibration_stats = {
    "normal_mean": None,
    "normal_median": None,
    "slight_mean": None,
    "slight_median": None,
    "very_mean": None,
    "very_median": None,
}


# ============================================================
# STATO MONITORAGGIO
# ============================================================

front_distance_history = deque(maxlen=MOVING_AVERAGE_SIZE)

bad_posture_start_time = None

alarm_active = False
alarm_start_time = None

# Evita di mandare infinite notifiche Windows
alarm_notification_sent = False


# ============================================================
# STATO UI
# ============================================================

WAITING = "WAITING"
POSITIONING = "POSITIONING"
NORMAL = "NORMAL"
SLIGHT_FORWARD = "SLIGHT_FORWARD"
VERY_FORWARD = "VERY_FORWARD"
SAVE_NAME = "SAVE_NAME"
LOAD_CALIBRATION = "LOAD_CALIBRATION"
MONITORING = "MONITORING"

ui_state = WAITING

state_start_time = None

normal_samples = []
slight_samples = []
very_samples = []

save_name_buffer = ""

selected_calibration_index = 0


# ============================================================
# MEDIAPIPE
# ============================================================

mp_tasks = mp.tasks
mp_vision = mp.tasks.vision

PoseLandmarker = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

BaseOptions = mp.tasks.BaseOptions


# ============================================================
# CALLBACK MEDIAPIPE
# ============================================================

def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks

    if result.pose_landmarks:
        with landmarks_lock:
            latest_landmarks = result.pose_landmarks[0]


# ============================================================
# NOTIFICA WINDOWS
# ============================================================

def send_system_notification():
    try:
        toast = Notification(
            app_id="PostureGuard",
            title="PostureGuard",
            msg="⚠️ Attenzione: postura scorretta mantenuta per più di 10 secondi."
        )

        toast.set_audio(
        "ms-winsoundevent:Notification.Mail",
        loop=False
        )

        toast.show()

    except Exception as e:
        print(f"[WARNING] Impossibile inviare la notifica Windows: {e}")


# ============================================================
# TRAY ICON
# ============================================================

def create_tray_image():
    """
    Crea una piccola icona direttamente in memoria.
    """

    image = Image.new("RGB", (64, 64), (25, 25, 25))
    draw = ImageDraw.Draw(image)

    # Scudo
    draw.polygon(
        [
            (32, 6),
            (53, 15),
            (49, 39),
            (32, 56),
            (15, 39),
            (11, 15),
        ],
        outline=(80, 230, 220),
        fill=(35, 80, 80)
    )

    # Persona stilizzata
    draw.ellipse(
        (27, 17, 37, 27),
        fill=(255, 255, 255)
    )

    draw.line(
        (32, 28, 32, 43),
        fill=(255, 255, 255),
        width=4
    )

    draw.line(
        (32, 31, 22, 38),
        fill=(255, 255, 255),
        width=3
    )

    draw.line(
        (32, 31, 42, 38),
        fill=(255, 255, 255),
        width=3
    )

    return image


def tray_show_window(icon, item):
    global show_window_requested

    show_window_requested = True


def tray_exit(icon, item):
    global running

    running = False


def start_tray():
    global tray_icon

    menu = pystray.Menu(
        pystray.MenuItem(
            "Mostra PostureGuard",
            tray_show_window
        ),
        pystray.MenuItem(
            "Esci",
            tray_exit
        )
    )

    tray_icon = pystray.Icon(
        "PostureGuard",
        create_tray_image(),
        "PostureGuard",
        menu
    )

    tray_icon.run()


# ============================================================
# UTILITY UI
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    color=WHITE,
    thickness=2
):
    """
    Testo con piccola ombra per renderlo leggibile
    anche sopra l'immagine della webcam.
    """

    x, y = position

    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_panel(frame, x1, y1, x2, y2, alpha=0.78):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        DARK_PANEL,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


def draw_button(
    frame,
    x1,
    y1,
    x2,
    y2,
    text,
    active=False
):
    color = DARK_BUTTON_HOVER if active else DARK_BUTTON

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        CYAN,
        2
    )

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        2
    )[0]

    text_x = x1 + ((x2 - x1) - text_size[0]) // 2
    text_y = y1 + ((y2 - y1) + text_size[1]) // 2

    draw_text(
        frame,
        text,
        (text_x, text_y),
        scale=0.65,
        color=WHITE,
        thickness=2
    )


def draw_title(frame, text):
    draw_text(
        frame,
        text,
        (30, 45),
        scale=1.0,
        color=CYAN,
        thickness=2
    )


# ============================================================
# GEOMETRIA
# ============================================================

def calculate_metrics(landmarks):
    """
    Calcola:
    - orientamento front/profile
    - head center
    - shoulder center
    - hip center
    - distanza testa-spalle normalizzata
    - angolo collo in profile
    """

    try:
        nose = landmarks[0]

        left_ear = landmarks[7]
        right_ear = landmarks[8]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

    except IndexError:
        return None

    shoulder_distance = abs(
        left_shoulder.x - right_shoulder.x
    )

    if shoulder_distance >= PROFILE_THRESHOLD:

        view = "FRONT"

        shoulder_center_x = (
            left_shoulder.x +
            right_shoulder.x
        ) / 2

        shoulder_center_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2

        head_center_x = (
            left_ear.x +
            right_ear.x
        ) / 2

        head_center_y = (
            left_ear.y +
            right_ear.y
        ) / 2

    else:

        # Profile.
        # Selezioniamo il lato più vicino alla camera
        # usando la coordinata z.

        if left_shoulder.z < right_shoulder.z:

            shoulder_center_x = left_shoulder.x
            shoulder_center_y = left_shoulder.y

            visible_ear = left_ear

        else:

            shoulder_center_x = right_shoulder.x
            shoulder_center_y = right_shoulder.y

            visible_ear = right_ear

        view = "PROFILE"

        head_center_x = visible_ear.x
        head_center_y = visible_ear.y

    hip_center_x = (
        left_hip.x +
        right_hip.x
    ) / 2

    hip_center_y = (
        left_hip.y +
        right_hip.y
    ) / 2

    # --------------------------------------------------------
    # FRONT
    # --------------------------------------------------------

    head_shoulder_distance = math.sqrt(
        (head_center_x - shoulder_center_x) ** 2 +
        (head_center_y - shoulder_center_y) ** 2
    )

    torso_length = math.sqrt(
        (shoulder_center_x - hip_center_x) ** 2 +
        (shoulder_center_y - hip_center_y) ** 2
    )

    if torso_length > 0:
        front_head_distance = (
            head_shoulder_distance /
            torso_length
        )
    else:
        front_head_distance = None

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    dx = head_center_x - shoulder_center_x
    dy = head_center_y - shoulder_center_y

    angle_rad = math.atan2(
        dx,
        -dy
    )

    profile_angle = math.degrees(angle_rad)

    return {
        "view": view,

        "head_center": (
            head_center_x,
            head_center_y
        ),

        "shoulder_center": (
            shoulder_center_x,
            shoulder_center_y
        ),

        "hip_center": (
            hip_center_x,
            hip_center_y
        ),

        "front_head_distance": front_head_distance,

        "profile_angle": profile_angle,
    }


# ============================================================
# DISEGNO SKELETON
# ============================================================

def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),

        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),

        (9, 10),

        (11, 12),

        (11, 13),
        (13, 15),

        (12, 14),
        (14, 16),

        (11, 23),
        (12, 24),

        (23, 24),

        (23, 25),
        (25, 27),

        (24, 26),
        (26, 28),
    ]

    # Skeleton
    for a, b in connections:

        if a >= len(landmarks) or b >= len(landmarks):
            continue

        p1 = landmarks[a]
        p2 = landmarks[b]

        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            GREEN,
            2
        )

    important_indices = [
        0,
        7,
        8,
        11,
        12,
        23,
        24,
    ]

    for index in important_indices:

        if index >= len(landmarks):
            continue

        point = landmarks[index]

        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(
            frame,
            (x, y),
            5,
            RED,
            -1
        )


def draw_centers(frame, metrics):
    height, width = frame.shape[:2]

    head_x = int(metrics["head_center"][0] * width)
    head_y = int(metrics["head_center"][1] * height)

    shoulder_x = int(metrics["shoulder_center"][0] * width)
    shoulder_y = int(metrics["shoulder_center"][1] * height)

    hip_x = int(metrics["hip_center"][0] * width)
    hip_y = int(metrics["hip_center"][1] * height)

    # Head center
    cv2.circle(
        frame,
        (head_x, head_y),
        8,
        YELLOW,
        -1
    )

    # Shoulder center
    cv2.circle(
        frame,
        (shoulder_x, shoulder_y),
        8,
        CYAN,
        -1
    )

    # Hip center
    cv2.circle(
        frame,
        (hip_x, hip_y),
        8,
        MAGENTA,
        -1
    )

    # Head -> Shoulder
    cv2.line(
        frame,
        (head_x, head_y),
        (shoulder_x, shoulder_y),
        YELLOW,
        3
    )

    # Vertical reference
    cv2.line(
        frame,
        (shoulder_x, shoulder_y - 120),
        (shoulder_x, shoulder_y + 120),
        BLUE,
        2
    )


# ============================================================
# METRICHE
# ============================================================

def draw_metrics(frame, metrics):
    if metrics is None:
        return

    draw_panel(
        frame,
        15,
        15,
        390,
        180,
        alpha=0.72
    )

    draw_text(
        frame,
        f"Vista: {metrics['view']}",
        (30, 45),
        scale=0.65,
        color=CYAN
    )

    if metrics["view"] == "FRONT":

        value = metrics["front_head_distance"]

        if value is not None:

            front_distance_history.append(value)

            moving_average = (
                sum(front_distance_history) /
                len(front_distance_history)
            )

            draw_text(
                frame,
                f"Distanza: {value:.3f}",
                (30, 80),
                scale=0.58
            )

            draw_text(
                frame,
                f"Media: {moving_average:.3f}",
                (30, 112),
                scale=0.58,
                color=LIGHT_CYAN
            )

            draw_text(
                frame,
                f"Campioni: {len(front_distance_history)}",
                (30, 144),
                scale=0.55
            )

            if calibration_threshold is not None:

                draw_text(
                    frame,
                    f"Soglia: {calibration_threshold:.3f}",
                    (30, 175),
                    scale=0.55,
                    color=YELLOW
                )

    else:

        front_distance_history.clear()

        angle = metrics["profile_angle"]

        draw_text(
            frame,
            f"Angolo collo: {angle:.1f}°",
            (30, 90),
            scale=0.62,
            color=LIGHT_CYAN
        )

        draw_text(
            frame,
            "Modalita profilo",
            (30, 125),
            scale=0.55
        )


# ============================================================
# CALIBRAZIONE
# ============================================================

def calculate_statistics(samples):
    if not samples:
        return None, None

    array = np.array(samples)

    return (
        float(np.mean(array)),
        float(np.median(array))
    )


def finish_calibration():
    global calibration_threshold
    global calibration_stats

    normal_mean, normal_median = calculate_statistics(
        normal_samples
    )

    slight_mean, slight_median = calculate_statistics(
        slight_samples
    )

    very_mean, very_median = calculate_statistics(
        very_samples
    )

    calibration_stats = {
        "normal_mean": normal_mean,
        "normal_median": normal_median,

        "slight_mean": slight_mean,
        "slight_median": slight_median,

        "very_mean": very_mean,
        "very_median": very_median,
    }

    if normal_mean is not None and slight_mean is not None:

        calibration_threshold = (
            normal_mean +
            slight_mean
        ) / 2.0

    else:
        calibration_threshold = None


# ============================================================
# SALVATAGGIO CALIBRAZIONE
# ============================================================

def ensure_calibration_directory():
    os.makedirs(
        CALIBRATIONS_DIR,
        exist_ok=True
    )


def save_calibration(name):
    global current_calibration_name

    ensure_calibration_directory()

    clean_name = name.strip()

    if not clean_name:
        return False

    forbidden = '<>:"/\\|?*'

    for char in forbidden:
        clean_name = clean_name.replace(char, "_")

    path = os.path.join(
        CALIBRATIONS_DIR,
        clean_name + ".json"
    )

    data = {
        "name": clean_name,

        "normal_mean":
            calibration_stats["normal_mean"],

        "normal_median":
            calibration_stats["normal_median"],

        "slight_mean":
            calibration_stats["slight_mean"],

        "slight_median":
            calibration_stats["slight_median"],

        "very_mean":
            calibration_stats["very_mean"],

        "very_median":
            calibration_stats["very_median"],

        "threshold":
            calibration_threshold,
    }

    try:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

        current_calibration_name = clean_name

        print(
            f"[INFO] Calibrazione salvata: {path}"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Salvataggio calibrazione: {e}"
        )

        return False


# ============================================================
# CARICAMENTO CALIBRAZIONE
# ============================================================

def get_calibration_files():

    ensure_calibration_directory()

    files = []

    for filename in os.listdir(CALIBRATIONS_DIR):

        if filename.lower().endswith(".json"):
            files.append(filename)

    files.sort()

    return files


def load_calibration(filename):

    global calibration_threshold
    global calibration_stats
    global current_calibration_name

    path = os.path.join(
        CALIBRATIONS_DIR,
        filename
    )

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        calibration_threshold = data["threshold"]

        calibration_stats = {
            "normal_mean":
                data.get("normal_mean"),

            "normal_median":
                data.get("normal_median"),

            "slight_mean":
                data.get("slight_mean"),

            "slight_median":
                data.get("slight_median"),

            "very_mean":
                data.get("very_mean"),

            "very_median":
                data.get("very_median"),
        }

        current_calibration_name = data.get(
            "name",
            filename[:-5]
        )

        print(
            f"[INFO] Calibrazione caricata: "
            f"{current_calibration_name}"
        )

        print(
            f"[INFO] Threshold: "
            f"{calibration_threshold:.3f}"
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Caricamento calibrazione: {e}"
        )

        return False


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global ui_state
    global selected_calibration_index
    global show_window_requested

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # --------------------------------------------------------
    # WAITING
    # --------------------------------------------------------

    if ui_state == WAITING:

        # INIZIA CALIBRAZIONE
        if (
            30 <= x <= 300 and
            80 <= y <= 140
        ):

            start_calibration()

        # CARICA CALIBRAZIONE
        elif (
            330 <= x <= 600 and
            80 <= y <= 140
        ):

            ui_state = LOAD_CALIBRATION

            selected_calibration_index = 0

    # --------------------------------------------------------
    # SAVE NAME
    # --------------------------------------------------------

    elif ui_state == SAVE_NAME:

        # SALVA
        if (
            30 <= x <= 300 and
            500 <= y <= 560
        ):

            if save_calibration(save_name_buffer):
                ui_state = MONITORING

        # NON SALVARE
        elif (
            330 <= x <= 600 and
            500 <= y <= 560
        ):

            ui_state = MONITORING

    # --------------------------------------------------------
    # LOAD CALIBRATION
    # --------------------------------------------------------

    elif ui_state == LOAD_CALIBRATION:

        files = get_calibration_files()

        start_y = 105

        for index, filename in enumerate(files):

            y1 = start_y + index * 45
            y2 = y1 + 35

            if (
                30 <= x <= 610 and
                y1 <= y <= y2
            ):

                selected_calibration_index = index

                break


# ============================================================
# AVVIO CALIBRAZIONE
# ============================================================

def start_calibration():

    global ui_state
    global state_start_time

    global normal_samples
    global slight_samples
    global very_samples

    global save_name_buffer

    normal_samples = []
    slight_samples = []
    very_samples = []

    save_name_buffer = ""

    front_distance_history.clear()

    ui_state = POSITIONING

    state_start_time = time.time()


# ============================================================
# RESET MONITORAGGIO
# ============================================================

def reset_monitoring():

    global bad_posture_start_time
    global alarm_active
    global alarm_start_time
    global alarm_notification_sent

    bad_posture_start_time = None

    alarm_active = False
    alarm_start_time = None

    alarm_notification_sent = False

    front_distance_history.clear()


# ============================================================
# DISEGNO START SCREEN
# ============================================================

def draw_waiting_screen(frame):

    draw_panel(
        frame,
        20,
        20,
        620,
        360,
        alpha=0.80
    )

    draw_title(
        frame,
        "POSTUREGUARD"
    )

    draw_text(
        frame,
        "Monitoraggio intelligente della postura",
        (32, 78),
        scale=0.60,
        color=LIGHT_CYAN
    )

    draw_button(
        frame,
        30,
        100,
        300,
        160,
        "INIZIA CALIBRAZIONE"
    )

    draw_button(
        frame,
        330,
        100,
        600,
        160,
        "CARICA CALIBRAZIONE"
    )

    if current_calibration_name:

        draw_text(
            frame,
            f"Calibrazione attiva: {current_calibration_name}",
            (32, 215),
            scale=0.55,
            color=GREEN
        )

        if calibration_threshold is not None:

            draw_text(
                frame,
                f"Soglia: {calibration_threshold:.3f}",
                (32, 250),
                scale=0.55
            )

    draw_text(
        frame,
        "Chiudi la finestra per continuare",
        (32, 305),
        scale=0.48,
        color=(210, 210, 210)
    )

    draw_text(
        frame,
        "il monitoraggio in background.",
        (32, 335),
        scale=0.48,
        color=(210, 210, 210)
    )


# ============================================================
# POSITIONING
# ============================================================

def draw_positioning_screen(frame):

    elapsed = time.time() - state_start_time

    remaining = max(
        0,
        CALIBRATION_POSITIONING_DURATION - elapsed
    )

    draw_panel(
        frame,
        20,
        20,
        620,
        300,
        alpha=0.80
    )

    draw_title(
        frame,
        "CALIBRAZIONE"
    )

    draw_text(
        frame,
        "Posizionati davanti al PC",
        (32, 95),
        scale=0.75,
        color=LIGHT_CYAN
    )

    draw_text(
        frame,
        "Assumi la tua normale distanza di lavoro.",
        (32, 135),
        scale=0.55
    )

    draw_text(
        frame,
        "Mantieni una postura naturale.",
        (32, 170),
        scale=0.55
    )

    draw_text(
        frame,
        f"Inizio tra {remaining:.1f} s",
        (32, 230),
        scale=0.85,
        color=YELLOW
    )


# ============================================================
# CALIBRATION PHASE SCREEN
# ============================================================

def draw_calibration_phase(
    frame,
    title,
    instruction,
    elapsed,
    samples
):

    remaining = max(
        0,
        CALIBRATION_PHASE_DURATION - elapsed
    )

    draw_panel(
        frame,
        20,
        20,
        620,
        270,
        alpha=0.80
    )

    draw_title(
        frame,
        title
    )

    draw_text(
        frame,
        instruction,
        (32, 100),
        scale=0.60,
        color=LIGHT_CYAN
    )

    draw_text(
        frame,
        f"Tempo rimanente: {remaining:.1f} s",
        (32, 155),
        scale=0.70,
        color=YELLOW
    )

    draw_text(
        frame,
        f"Campioni raccolti: {len(samples)}",
        (32, 200),
        scale=0.55
    )


# ============================================================
# SAVE SCREEN
# ============================================================

def draw_save_screen(frame):

    draw_panel(
        frame,
        20,
        20,
        620,
        590,
        alpha=0.84
    )

    draw_title(
        frame,
        "SALVA CALIBRAZIONE"
    )

    draw_text(
        frame,
        "Inserisci un nome:",
        (32, 100),
        scale=0.60,
        color=LIGHT_CYAN
    )

    cv2.rectangle(
        frame,
        (30, 125),
        (610, 180),
        (50, 50, 50),
        -1
    )

    cv2.rectangle(
        frame,
        (30, 125),
        (610, 180),
        CYAN,
        2
    )

    draw_text(
        frame,
        save_name_buffer + "_",
        (45, 162),
        scale=0.65,
        color=WHITE
    )

    draw_text(
        frame,
        "Risultati calibrazione",
        (32, 225),
        scale=0.62,
        color=CYAN
    )

    if calibration_stats["normal_mean"] is not None:

        draw_text(
            frame,
            f"Normale: {calibration_stats['normal_mean']:.3f}",
            (45, 260),
            scale=0.55
        )

        draw_text(
            frame,
            f"Leggermente avanti: "
            f"{calibration_stats['slight_mean']:.3f}",
            (45, 295),
            scale=0.55
        )

        draw_text(
            frame,
            f"Molto avanti: "
            f"{calibration_stats['very_mean']:.3f}",
            (45, 330),
            scale=0.55
        )

        draw_text(
            frame,
            f"SOGLIA: {calibration_threshold:.3f}",
            (45, 375),
            scale=0.70,
            color=YELLOW
        )

    draw_button(
        frame,
        30,
        500,
        300,
        560,
        "SALVA"
    )

    draw_button(
        frame,
        330,
        500,
        600,
        560,
        "NON SALVARE"
    )

    draw_text(
        frame,
        "ENTER = salva    ESC = annulla",
        (32, 470),
        scale=0.50,
        color=(220, 220, 220)
    )


# ============================================================
# LOAD SCREEN
# ============================================================

def draw_load_screen(frame):

    draw_panel(
        frame,
        20,
        20,
        620,
        590,
        alpha=0.84
    )

    draw_title(
        frame,
        "CARICA CALIBRAZIONE"
    )

    files = get_calibration_files()

    if not files:

        draw_text(
            frame,
            "Nessuna calibrazione salvata.",
            (32, 115),
            scale=0.62,
            color=YELLOW
        )

        draw_text(
            frame,
            "Premi ESC per tornare.",
            (32, 165),
            scale=0.55
        )

        return

    draw_text(
        frame,
        "Seleziona una calibrazione:",
        (32, 90),
        scale=0.58,
        color=LIGHT_CYAN
    )

    start_y = 115

    for index, filename in enumerate(files):

        y1 = start_y + index * 45
        y2 = y1 + 35

        if y2 > 440:
            break

        selected = (
            index ==
            selected_calibration_index
        )

        background = (
            (70, 80, 80)
            if selected
            else (40, 40, 40)
        )

        cv2.rectangle(
            frame,
            (30, y1),
            (610, y2),
            background,
            -1
        )

        cv2.rectangle(
            frame,
            (30, y1),
            (610, y2),
            CYAN if selected else (80, 80, 80),
            2
        )

        name = filename[:-5]

        draw_text(
            frame,
            name,
            (45, y1 + 25),
            scale=0.55,
            color=WHITE if selected else (210, 210, 210)
        )

    draw_text(
        frame,
        "ENTER = carica    ESC = torna indietro",
        (32, 475),
        scale=0.55,
        color=LIGHT_CYAN
    )

    draw_text(
        frame,
        "Clicca una calibrazione per selezionarla.",
        (32, 515),
        scale=0.48,
        color=(210, 210, 210)
    )


# ============================================================
# ALARM OVERLAY
# ============================================================

def draw_alarm(frame):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        (0, 0, 180),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.45,
        frame,
        0.55,
        0,
        frame
    )

    draw_panel(
        frame,
        70,
        height // 2 - 100,
        width - 70,
        height // 2 + 100,
        alpha=0.88
    )

    text = "ATTENZIONE"

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        3
    )[0]

    x = (width - text_size[0]) // 2

    draw_text(
        frame,
        text,
        (x, height // 2 - 30),
        scale=1.2,
        color=RED,
        thickness=3
    )

    text2 = "Postura scorretta rilevata"

    text_size = cv2.getTextSize(
        text2,
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        2
    )[0]

    x = (width - text_size[0]) // 2

    draw_text(
        frame,
        text2,
        (x, height // 2 + 30),
        scale=0.65,
        color=WHITE
    )


# ============================================================
# MONITORAGGIO
# ============================================================

def process_monitoring(metrics):

    global bad_posture_start_time
    global alarm_active
    global alarm_start_time
    global alarm_notification_sent

    if metrics is None:
        return

    if metrics["view"] != "FRONT":
        return

    value = metrics["front_head_distance"]

    if value is None:
        return

    front_distance_history.append(value)

    if not front_distance_history:
        return

    moving_average = (
        sum(front_distance_history) /
        len(front_distance_history)
    )

    if calibration_threshold is None:
        return

    # --------------------------------------------------------
    # POSTURA CORRETTA
    # --------------------------------------------------------

    if moving_average >= calibration_threshold:

        bad_posture_start_time = None

        alarm_active = False
        alarm_start_time = None

        alarm_notification_sent = False

        return

    # --------------------------------------------------------
    # POSTURA SCORRETTA
    # --------------------------------------------------------

    if bad_posture_start_time is None:

        bad_posture_start_time = time.time()

    bad_duration = (
        time.time() -
        bad_posture_start_time
    )

    # --------------------------------------------------------
    # ALLARME
    # --------------------------------------------------------

    if bad_duration >= BAD_POSTURE_DURATION:

        if not alarm_notification_sent:

            send_system_notification()

            alarm_notification_sent = True

            print(
                "[ALARM] Notifica Windows inviata."
            )

        if not alarm_active:

            alarm_active = True
            alarm_start_time = time.time()

    # L'allarme visivo rimane per alcuni secondi
    if alarm_active and alarm_start_time is not None:

        if (
            time.time() -
            alarm_start_time
        ) >= ALARM_DURATION:

            alarm_active = False


# ============================================================
# INPUT TASTIERA
# ============================================================

def handle_save_name_key(key):

    global save_name_buffer
    global ui_state

    if key == 13:  # ENTER

        if save_calibration(save_name_buffer):

            ui_state = MONITORING

        return

    if key == 27:  # ESC

        ui_state = MONITORING

        return

    if key == 8:  # BACKSPACE

        save_name_buffer = save_name_buffer[:-1]

        return

    # ASCII normale
    if 32 <= key <= 126:

        if len(save_name_buffer) < 30:

            save_name_buffer += chr(key)


# ============================================================
# APERTURA FINESTRA
# ============================================================

def create_window():

    global window_open
    global background_mode

    try:

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            window_name,
            640,
            480
        )

        cv2.setMouseCallback(
            window_name,
            mouse_callback
        )

        window_open = True
        background_mode = False

        print(
            "[INFO] Finestra PostureGuard aperta."
        )

    except Exception as e:

        window_open = False
        background_mode = True

        print(
            f"[WARNING] Errore creazione finestra: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global running
    global background_mode
    global window_open
    global show_window_requested

    global ui_state
    global state_start_time

    global save_name_buffer
    global selected_calibration_index

    global normal_samples
    global slight_samples
    global very_samples

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "[ERROR] Impossibile aprire la webcam."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=VisionRunningMode.LIVE_STREAM,

        num_poses=1,

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

        min_tracking_confidence=0.5,

        result_callback=result_callback
    )

    landmarker = PoseLandmarker.create_from_options(
        options
    )

    # --------------------------------------------------------
    # TRAY
    # --------------------------------------------------------

    tray_thread = threading.Thread(
        target=start_tray,
        daemon=True
    )

    tray_thread.start()

    # --------------------------------------------------------
    # FINESTRA
    # --------------------------------------------------------

    create_window()

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while running:

        # ====================================================
        # RICHIESTA MOSTRA FINESTRA
        # ====================================================

        if show_window_requested:

            show_window_requested = False

            create_window()

        # ====================================================
        # WEBCAM
        # ====================================================

        ret, frame = cap.read()

        if not ret:

            time.sleep(0.01)

            continue

        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # MEDIAPIPE INPUT
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            time.time() * 1000
        )

        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )

        # ====================================================
        # LANDMARKS
        # ====================================================

        with landmarks_lock:

            landmarks = latest_landmarks

        metrics = None

        if landmarks is not None:

            metrics = calculate_metrics(
                landmarks
            )

            draw_pose(
                frame,
                landmarks
            )

            if metrics is not None:

                draw_centers(
                    frame,
                    metrics
                )

        # ====================================================
        # UI
        # ====================================================

        if ui_state == WAITING:

            draw_waiting_screen(
                frame
            )

        elif ui_state == POSITIONING:

            elapsed = (
                time.time() -
                state_start_time
            )

            draw_positioning_screen(
                frame
            )

            if elapsed >= CALIBRATION_POSITIONING_DURATION:

                ui_state = NORMAL
                state_start_time = time.time()

                normal_samples.clear()

        elif ui_state == NORMAL:

            elapsed = (
                time.time() -
                state_start_time
            )

            if (
                metrics is not None and
                metrics["view"] == "FRONT" and
                metrics["front_head_distance"] is not None
            ):

                normal_samples.append(
                    metrics["front_head_distance"]
                )

            draw_calibration_phase(
                frame,
                "POSTURA NORMALE",
                "Mantieni la tua postura corretta.",
                elapsed,
                normal_samples
            )

            if elapsed >= CALIBRATION_PHASE_DURATION:

                ui_state = SLIGHT_FORWARD

                state_start_time = time.time()

                slight_samples.clear()

        elif ui_state == SLIGHT_FORWARD:

            elapsed = (
                time.time() -
                state_start_time
            )

            if (
                metrics is not None and
                metrics["view"] == "FRONT" and
                metrics["front_head_distance"] is not None
            ):

                slight_samples.append(
                    metrics["front_head_distance"]
                )

            draw_calibration_phase(
                frame,
                "LEGGERMENTE IN AVANTI",
                "Porta leggermente la testa in avanti.",
                elapsed,
                slight_samples
            )

            if elapsed >= CALIBRATION_PHASE_DURATION:

                ui_state = VERY_FORWARD

                state_start_time = time.time()

                very_samples.clear()

        elif ui_state == VERY_FORWARD:

            elapsed = (
                time.time() -
                state_start_time
            )

            if (
                metrics is not None and
                metrics["view"] == "FRONT" and
                metrics["front_head_distance"] is not None
            ):

                very_samples.append(
                    metrics["front_head_distance"]
                )

            draw_calibration_phase(
                frame,
                "MOLTO IN AVANTI",
                "Porta la testa molto in avanti.",
                elapsed,
                very_samples
            )

            if elapsed >= CALIBRATION_PHASE_DURATION:

                finish_calibration()

                ui_state = SAVE_NAME

                state_start_time = time.time()

                save_name_buffer = ""

        elif ui_state == SAVE_NAME:

            draw_save_screen(
                frame
            )

        elif ui_state == LOAD_CALIBRATION:

            draw_load_screen(
                frame
            )

        elif ui_state == MONITORING:

            process_monitoring(
                metrics
            )

            draw_metrics(
                frame,
                metrics
            )

            # Nome calibrazione
            if current_calibration_name:

                draw_panel(
                    frame,
                    15,
                    200,
                    390,
                    245,
                    alpha=0.70
                )

                draw_text(
                    frame,
                    f"Calibrazione: {current_calibration_name}",
                    (30, 230),
                    scale=0.50,
                    color=LIGHT_CYAN
                )

            # Stato
            if alarm_active:

                draw_alarm(
                    frame
                )

            else:

                draw_panel(
                    frame,
                    15,
                    260,
                    300,
                    305,
                    alpha=0.65
                )

                if (
                    metrics is not None and
                    metrics["view"] == "FRONT"
                ):

                    if (
                        calibration_threshold is not None and
                        metrics["front_head_distance"] is not None
                    ):

                        moving_average = (
                            sum(front_distance_history) /
                            len(front_distance_history)
                            if front_distance_history
                            else 0
                        )

                        if moving_average >= calibration_threshold:

                            draw_text(
                                frame,
                                "POSTURA OK",
                                (30, 290),
                                scale=0.55,
                                color=GREEN
                            )

                        else:

                            if bad_posture_start_time is not None:

                                duration = (
                                    time.time() -
                                    bad_posture_start_time
                                )

                                remaining = max(
                                    0,
                                    BAD_POSTURE_DURATION -
                                    duration
                                )

                                draw_text(
                                    frame,
                                    "POSTURA DA CORREGGERE",
                                    (30, 290),
                                    scale=0.45,
                                    color=YELLOW
                                )

                                draw_text(
                                    frame,
                                    f"Allarme tra {remaining:.1f}s",
                                    (30, 315),
                                    scale=0.45,
                                    color=YELLOW
                                )

        # ====================================================
        # GESTIONE FINESTRA
        # ====================================================
        #
        # IMPORTANTE:
        #
        # waitKey() viene chiamato PRIMA del controllo della
        # finestra e di imshow().
        #
        # Questo permette a OpenCV di processare l'evento
        # della X. Se la finestra è stata chiusa, impostiamo
        # window_open = False e NON chiamiamo imshow().
        #
        # In questo modo imshow() non può ricreare la finestra.
        # ====================================================

        if window_open and not background_mode:

            try:

                # ------------------------------------------------
                # PROCESSA GLI EVENTI DELLA FINESTRA
                # ------------------------------------------------

                key = cv2.waitKey(1) & 0xFF

                # ------------------------------------------------
                # CONTROLLA SE LA X È STATA PREMUTA
                # ------------------------------------------------

                try:

                    window_visible = cv2.getWindowProperty(
                        window_name,
                        cv2.WND_PROP_VISIBLE
                    )

                except cv2.error:

                    window_visible = -1

                if window_visible < 1:

                    print(
                        "[INFO] Finestra chiusa. "
                        "PostureGuard continua in background."
                    )

                    window_open = False
                    background_mode = True

                    continue

                # ------------------------------------------------
                # TASTIERA
                # ------------------------------------------------

                if key == ord("q"):

                    running = False

                    break

                # ------------------------------------------------
                # SAVE NAME
                # ------------------------------------------------

                if ui_state == SAVE_NAME:

                    handle_save_name_key(
                        key
                    )

                # ------------------------------------------------
                # LOAD
                # ------------------------------------------------

                elif ui_state == LOAD_CALIBRATION:

                    if key == 13:

                        files = get_calibration_files()

                        if files:

                            if (
                                selected_calibration_index
                                < len(files)
                            ):

                                filename = files[
                                    selected_calibration_index
                                ]

                                if load_calibration(
                                    filename
                                ):

                                    reset_monitoring()

                                    ui_state = MONITORING

                    elif key == 27:

                        ui_state = WAITING

                # ------------------------------------------------
                # MONITORING
                # ------------------------------------------------

                elif ui_state == MONITORING:

                    if key == 27:

                        reset_monitoring()

                        ui_state = WAITING

                # ------------------------------------------------
                # MOSTRA FRAME
                # ------------------------------------------------

                cv2.imshow(
                    window_name,
                    frame
                )

            except cv2.error:

                print(
                    "[INFO] Finestra chiusa. "
                    "PostureGuard continua in background."
                )

                window_open = False
                background_mode = True

                continue

            except Exception as e:

                print(
                    f"[WARNING] Errore gestione GUI: {e}"
                )

                window_open = False
                background_mode = True

                continue

        else:

            # ====================================================
            # MODALITÀ BACKGROUND
            # ====================================================
            #
            # La webcam, MediaPipe e il monitoraggio continuano.
            # Non viene mostrata nessuna finestra OpenCV.
            # ====================================================

            time.sleep(0.01)

    # ========================================================
    # CLEANUP
    # ========================================================

    running = False

    try:

        if tray_icon is not None:
            tray_icon.stop()

    except Exception:
        pass

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print(
        "[INFO] PostureGuard terminato."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()