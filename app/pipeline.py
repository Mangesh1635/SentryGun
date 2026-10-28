"""Frame-by-frame orchestration. No camera, video or serial IO here:
the caller passes detections in and gets decisions out, so this is
fully testable with fake detections.

Detection dicts (all boxes are (x1, y1, x2, y2) frame pixels):
    persons: [{"box": ..., "conf": ...}]
    badges:  [box, ...]              (trained model, or fallback boxes)
    faces:   {person_index: box}     (face box inside that person)
"""

from __future__ import annotations

import time

from .core import (
    Config, EvidenceState, GREEN, RED, Smoother, box_center,
    classify_person, face_to_servo, servo_line,
)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.smoother = Smoother(cfg.smoothing_alpha)
        self.evidence = EvidenceState(cfg.clip_seconds, cfg.cooldown_seconds)
        self.last_servo_at = 0.0

    def process(self, persons, badges, faces, frame_width: int,
                frame_height: int, now: float | None = None):
        now = time.monotonic() if now is None else now
        classified = []
        for i, person in enumerate(persons):
            label = classify_person(person["box"], badges)
            item = {
                "box": person["box"],
                "conf": person.get("conf", 0.0),
                "label": label,
                "color": GREEN if label == "FRIEND" else RED,
                "face_box": faces.get(i),
                "face_center": None,
            }
            if label == "ENEMY" and faces.get(i) is not None:
                item["face_center"] = box_center(faces[i])
            classified.append(item)

        enemies = [p for p in classified if p["label"] == "ENEMY"]
        evidence_event = self.evidence.update(bool(enemies), now)

        # Servo target: the largest enemy with a face (closest person).
        # Friends never move the mount. No enemy face -> no servo move.
        servo = None
        target = None
        if enemies:
            with_face = [p for p in enemies if p["face_center"] is not None]
            if with_face:
                target = max(
                    with_face,
                    key=lambda p: (p["box"][2] - p["box"][0])
                    * (p["box"][3] - p["box"][1]),
                )
                smooth = self.smoother.update(target["face_center"])
                target = dict(target)
                target["smooth_center"] = smooth
                if now - self.last_servo_at >= self.cfg.servo_update_seconds:
                    self.last_servo_at = now
                    pan, tilt = face_to_servo(
                        smooth, frame_width, frame_height, self.cfg)
                    servo = servo_line(pan, tilt, True)
        else:
            self.smoother.reset()
            # Tell the Arduino the scene is clear (LED/buzzer off).
            # Sent at most once per second, not every frame.
            if now - self.last_servo_at >= 1.0:
                self.last_servo_at = now
                servo = servo_line(0, 0, False)

        return {
            "persons": classified,
            "target": target,
            "servo": servo,
            "evidence_event": evidence_event,
            "evidence_recording": self.evidence.recording,
        }
