"""Minimal dashboard: live footage + evidence list. Nothing else.

Routes:
    GET /               the dashboard page (video + evidence only)
    GET /video          MJPEG stream of annotated frames
    GET /api/evidence   JSON list of enemy clips
    GET /evidence/<f>   a local-only clip file (Cloudinary clips link
                        straight to Cloudinary from the page)

Run:  python3 run.py
Then open http://127.0.0.1:8000 in a browser.
DEMO=1 python3 run.py  -> synthetic moving person, no camera/model,
so you can check the dashboard before YOLO downloads.
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .arduino import ArduinoLink
from .core import Config
from .evidence import load_evidence, record_evidence
from .pipeline import Pipeline

app = FastAPI(title="SentryGun")
cfg = Config.from_env()
pipeline = Pipeline(cfg)
_state = {
    "jpeg": None,
    "lock": threading.Lock(),
    "arduino": None,
    "entries": load_evidence(cfg.evidence_dir),
    "started": False,
}

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SentryGun</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:16px}
img{width:100%;max-width:960px;background:#000}
a{color:#7dd3fc}
li{margin:6px 0}
</style></head><body>
<h1>Live</h1>
<img src="/video" alt="Live footage">
<h2>Evidence</h2>
<ul id="evidence"><li>Loading...</li></ul>
<script>
async function load(){
  const r = await fetch('/api/evidence');
  const items = await r.json();
  const ul = document.getElementById('evidence');
  if(!items.length){ ul.innerHTML = '<li>No enemy clips yet.</li>'; return; }
  ul.innerHTML = items.slice().reverse().map(e => {
    const href = e.cloudinary_url || ('/evidence/' + e.local_file);
    return '<li>' + e.time + ' - <a href="' + href + '" target="_blank">' +
      (e.cloudinary_url ? 'Cloudinary clip' : 'Local clip (' + e.status + ')') +
      '</a></li>';
  }).join('');
}
load(); setInterval(load, 5000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/evidence")
def evidence():
    return _state["entries"]


@app.get("/evidence/{filename}")
def evidence_file(filename: str):
    # Basename only: never let a URL walk out of the evidence folder.
    safe = os.path.basename(filename)
    path = os.path.join(cfg.evidence_dir, safe)
    if not os.path.exists(path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="video/mp4")


def _mjpeg():
    while True:
        with _state["lock"]:
            jpeg = _state["jpeg"]
        if jpeg is None:
            time.sleep(0.1)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg + b"\r\n")
        time.sleep(0.03)


@app.get("/video")
def video():
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


def _camera_loop():
    import cv2
    import numpy as np

    from .vision import (BadgeDetector, FaceDetector, PersonDetector,
                         annotate, fallback_badge_boxes)

    _state["arduino"] = ArduinoLink(cfg.serial_port, cfg.serial_baud)
    person_detector = None
    badge_detector = None
    face_detector = None
    if not cfg.demo:
        person_detector = PersonDetector(cfg.person_model, cfg.person_conf)
        badge_detector = BadgeDetector(cfg.badge_model)
        face_detector = FaceDetector()

    source = int(cfg.camera_source) if cfg.camera_source.isdigit() \
        else cfg.camera_source
    cap = None if cfg.demo else cv2.VideoCapture(source)
    if cap is not None and not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {cfg.camera_source}")
        return

    writer = None
    clip_path = None
    t0 = time.time()
    try:
        while True:
            if cfg.demo:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                x = int(200 + 150 * np.sin((time.time() - t0) * 0.5))
                person_box = (x, 120, x + 160, 420)
                persons = [{"box": person_box, "conf": 0.9}]
                # 20s enemy, 20s friend, repeating, so both colours show.
                cycle = int(time.time() - t0) % 40
                badges = [] if cycle < 20 else [(
                    person_box[0] + 40, person_box[1] + 60,
                    person_box[0] + 90, person_box[1] + 110)]
                faces = {0: (person_box[0] + 50, person_box[1] + 10,
                             person_box[0] + 110, person_box[1] + 70)}
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                persons = person_detector.detect(frame)
                badges = badge_detector.detect(frame)
                if badge_detector.model is None:
                    badges = []
                    for person in persons:
                        badges += fallback_badge_boxes(
                            frame, person["box"], cfg.badge_red_ratio)
                faces = {}
                for i, person in enumerate(persons):
                    face = face_detector.detect_in_person(
                        frame, person["box"])
                    if face is not None:
                        faces[i] = face

            h, w = frame.shape[:2]
            result = pipeline.process(persons, badges, faces, w, h)
            if result["servo"] and _state["arduino"] is not None:
                _state["arduino"].send(result["servo"])

            if result["evidence_event"] == "started":
                stamp = time.strftime("%Y%m%d_%H%M%S")
                clip_path = os.path.join(
                    cfg.evidence_dir, f"enemy_{stamp}.mp4")
                os.makedirs(cfg.evidence_dir, exist_ok=True)
                writer = cv2.VideoWriter(
                    clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            if result["evidence_recording"] and writer is not None:
                writer.write(annotate(frame.copy(),
                                      result["persons"], result["target"]))
            elif result["evidence_event"] == "finished" and writer is not None:
                writer.release()
                writer = None
                record_evidence(cfg.evidence_dir, _state["entries"],
                                clip_path)
                clip_path = None

            annotated = annotate(frame, result["persons"],
                                 result["target"])
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                with _state["lock"]:
                    _state["jpeg"] = buf.tobytes()
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()


@app.on_event("startup")
def startup():
    if _state["started"]:
        return
    _state["started"] = True
    threading.Thread(target=_camera_loop, daemon=True).start()
