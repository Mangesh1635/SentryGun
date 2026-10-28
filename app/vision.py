"""Vision: COCO person detection, badge detection, face detection.

Imports of cv2 / ultralytics are lazy, inside the functions that need
them, so importing this module (and running the pure-logic tests)
never requires a model download.

Person model: YOLO pretrained on COCO, class 0 = person only.
Badge model: a YOLO model trained by train_badge.py on the badge's
actual colour and shape. Until that file exists, badge detection
falls back to the old project's red colour check, tightened with a
compact-contour shape check, inside each person box only. That
fallback is interim: any red object on a person can fool it, which
is exactly why the trained model is the intended path.
"""

from __future__ import annotations

import os


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            "OpenCV is not installed. Run: "
            "python3 -m pip install -r requirements.txt") from exc
    return cv2


class PersonDetector:
    def __init__(self, model_path: str, conf: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed. Run: "
                "python3 -m pip install -r requirements.txt") from exc
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Return [{"box": (x1,y1,x2,y2), "conf": float}] for persons."""
        results = self.model.predict(frame, classes=[0], conf=self.conf,
                                     verbose=False)
        persons = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            persons.append({"box": (x1, y1, x2, y2),
                            "conf": float(box.conf[0])})
        return persons


class BadgeDetector:
    """Trained badge model if present, else None (caller uses fallback)."""

    def __init__(self, model_path: str, conf: float = 0.5):
        self.model = None
        if os.path.exists(model_path):
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        if self.model is None:
            return []
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        badges = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            badges.append((x1, y1, x2, y2))
        return badges


class FaceDetector:
    def __init__(self):
        cv2 = _cv2()
        cascade = os.path.join(cv2.data.haarcascades,
                               "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade)

    def detect_in_person(self, frame, person_box):
        """Largest face inside a person box, in frame coordinates."""
        cv2 = _cv2()
        x1, y1, x2, y2 = person_box
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return (x1 + int(fx), y1 + int(fy),
                x1 + int(fx + fw), y1 + int(fy + fh))


def fallback_badge_boxes(frame, person_box, red_ratio: float = 0.02):
    """Interim badge check: red + compact contour, inside a person box.

    Returns [badge_box] or []. Kept deliberately strict: the red
    pixel ratio must clear the threshold AND the largest red contour
    must fill a reasonable part of its own bounding box (a badge is
    a solid coloured shape, not scattered red pixels).
    """
    cv2 = _cv2()
    import numpy as np

    x1, y1, x2, y2 = person_box
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return []
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    ratio = cv2.countNonZero(mask) / float(roi.shape[0] * roi.shape[1])
    if ratio <= red_ratio:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(contour)
    if bw * bh == 0:
        return []
    fill = cv2.contourArea(contour) / float(bw * bh)
    if fill < 0.4:
        return []
    return [(x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)]


def annotate(frame, classified, target):
    """Draw boxes, labels, and (enemy only) the face-centre point."""
    cv2 = _cv2()
    for person in classified:
        x1, y1, x2, y2 = person["box"]
        color = person["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, person["label"], (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if target is not None and target.get("smooth_center"):
        cx, cy = target["smooth_center"]
        # Crosshair + filled dot at the exact face centre.
        cv2.circle(frame, (cx, cy), 6, (255, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 12, (255, 0, 255), 2)
        cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (255, 0, 255), 1)
        cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (255, 0, 255), 1)
    return frame
