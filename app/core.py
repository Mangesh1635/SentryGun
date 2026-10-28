"""Pure logic for SentryGun v2. No cv2 / torch / serial imports here,
so this module is unit-testable anywhere.

Box format everywhere: (x1, y1, x2, y2) in frame pixels.
A person is a FRIEND only when a badge box overlaps that person.
Otherwise the person is an ENEMY. Colour/shape alone never makes a
person a friend: the badge must be on that person.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------- config

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env reader (KEY=VALUE lines, # comments). Never overwrites
    a variable already set in the real environment."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Config:
    # Camera: "0" = MacBook camera, or an IP-camera URL / video file path.
    camera_source: str = "0"
    # COCO-pretrained person model. yolov8n.pt auto-downloads on first run.
    person_model: str = "yolov8n.pt"
    person_conf: float = 0.5
    # Trained badge model, produced by train_badge.py. If this file does
    # not exist, the colour+shape fallback in vision.py is used instead.
    badge_model: str = "runs/badge_detector/weights/best.pt"
    # Serial link to the Arduino pan/tilt mount (old project values).
    serial_port: str = "/dev/cu.usbserial-120"
    serial_baud: int = 115200
    pan_min: int = 0
    pan_max: int = 180
    tilt_min: int = 0
    tilt_max: int = 180
    smoothing_alpha: float = 0.2
    servo_update_seconds: float = 0.1
    # Evidence clip length. The brief says 5-10 seconds; 7 is the default.
    clip_seconds: float = 7.0
    cooldown_seconds: float = 10.0
    evidence_dir: str = "evidence"
    # Fallback badge check (interim only, see vision.py).
    badge_red_ratio: float = 0.02
    host: str = "127.0.0.1"
    port: int = 8000
    demo: bool = False

    def __post_init__(self) -> None:
        if not 5.0 <= self.clip_seconds <= 10.0:
            raise ValueError("clip_seconds must be between 5 and 10")

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            camera_source=os.environ.get("CAMERA_SOURCE", "0"),
            person_model=os.environ.get("PERSON_MODEL", "yolov8n.pt"),
            person_conf=_env_float("PERSON_CONF", 0.5),
            badge_model=os.environ.get(
                "BADGE_MODEL", "runs/badge_detector/weights/best.pt"),
            serial_port=os.environ.get("SERIAL_PORT", "/dev/cu.usbserial-120"),
            serial_baud=_env_int("SERIAL_BAUD", 115200),
            smoothing_alpha=_env_float("SMOOTHING_ALPHA", 0.2),
            clip_seconds=_env_float("CLIP_SECONDS", 7.0),
            cooldown_seconds=_env_float("COOLDOWN_SECONDS", 10.0),
            evidence_dir=os.environ.get("EVIDENCE_DIR", "evidence"),
            host=os.environ.get("HOST", "127.0.0.1"),
            port=_env_int("PORT", 8000),
            demo=os.environ.get("DEMO", "0") == "1",
        )

    def cloudinary_configured(self) -> bool:
        return bool(
            os.environ.get("CLOUDINARY_CLOUD_NAME")
            and os.environ.get("CLOUDINARY_API_KEY")
            and os.environ.get("CLOUDINARY_API_SECRET")
        )


# ---------------------------------------------------------------- geometry

def box_center(box) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (int(x1 + x2) // 2, int(y1 + y2) // 2)


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def intersection_area(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def map_to_angle(value: float, src_min: float, src_max: float,
                 dst_min: int = 0, dst_max: int = 180) -> int:
    """Map a pixel coordinate to a servo angle, clamped to the range."""
    if src_max == src_min:
        return (dst_min + dst_max) // 2
    ratio = (value - src_min) / (src_max - src_min)
    ratio = min(1.0, max(0.0, ratio))
    return int(dst_min + ratio * (dst_max - dst_min))


def face_to_servo(face_center, frame_width: int, frame_height: int,
                  cfg: Config) -> tuple[int, int]:
    cx, cy = face_center
    pan = map_to_angle(cx, 0, frame_width, cfg.pan_min, cfg.pan_max)
    tilt = map_to_angle(cy, 0, frame_height, cfg.tilt_min, cfg.tilt_max)
    return pan, tilt


class Smoother:
    """Exponential smoothing so the servos do not jitter frame to frame."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value = None

    def update(self, point):
        if self.value is None:
            self.value = (float(point[0]), float(point[1]))
        else:
            self.value = (
                self.alpha * point[0] + (1 - self.alpha) * self.value[0],
                self.alpha * point[1] + (1 - self.alpha) * self.value[1],
            )
        return (int(self.value[0]), int(self.value[1]))

    def reset(self):
        self.value = None


# ---------------------------------------------------------------- classification

GREEN = (0, 255, 0)   # BGR, friend
RED = (0, 0, 255)     # BGR, enemy


def badge_on_person(person_box, badge_boxes) -> bool:
    """True only if a badge box overlaps this person.

    A badge centre inside the person box counts. A badge mostly
    overlapping the person also counts. A badge on someone else, or
    elsewhere in the frame, does not make this person a friend.
    """
    for badge in badge_boxes:
        cx, cy = box_center(badge)
        if person_box[0] <= cx <= person_box[2] and \
                person_box[1] <= cy <= person_box[3]:
            return True
        badge_area = box_area(badge)
        if badge_area > 0 and \
                intersection_area(person_box, badge) / badge_area >= 0.5:
            return True
    return False


def classify_person(person_box, badge_boxes) -> str:
    return "FRIEND" if badge_on_person(person_box, badge_boxes) else "ENEMY"


def servo_line(pan: int, tilt: int, is_enemy: bool) -> str:
    """Arduino protocol. Servos move ONLY for an enemy.

    ENEMY -> "PAN:90;TILT:90;ENEMY\\n"  (mount moves, LED/buzzer on)
    FRIEND -> "FRIEND\\n"               (LED/buzzer off, servos hold)
    """
    if is_enemy:
        return f"PAN:{pan};TILT:{tilt};ENEMY\n"
    return "FRIEND\n"


# ---------------------------------------------------------------- evidence clips

@dataclass
class EvidenceState:
    """Timing state machine for 5-10 second enemy clips.

    idle -> recording (first enemy frame) -> cooldown (after the clip
    finishes, so one person does not produce a clip every second).
    The caller supplies `now` (seconds) so tests can control time.
    """

    clip_seconds: float = 7.0
    cooldown_seconds: float = 10.0
    state: str = "idle"
    started_at: float = 0.0
    finished_at: float = 0.0

    def update(self, enemy_seen: bool, now: float) -> str:
        """Returns 'started', 'finished', or '' (no change)."""
        if self.state == "idle" and enemy_seen:
            self.state = "recording"
            self.started_at = now
            return "started"
        if self.state == "recording" and \
                now - self.started_at >= self.clip_seconds:
            self.state = "cooldown"
            self.finished_at = now
            return "finished"
        if self.state == "cooldown" and \
                now - self.finished_at >= self.cooldown_seconds:
            self.state = "idle"
            if enemy_seen:
                self.state = "recording"
                self.started_at = now
                return "started"
        return ""

    @property
    def recording(self) -> bool:
        return self.state == "recording"
