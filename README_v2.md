# SentryGun v2 — simple version

Camera pan/tilt mount only. No trigger, no weapon. The mount only ever
aims a camera at an enemy face centre.

## What it does

- Person detection: YOLO pretrained on COCO, class 0 = person only.
- Badge detection: a YOLO model trained on your badge (see below).
  Until that model exists, an interim red-colour + compact-shape check
  inside each person box is used instead.
- Friend = person with a badge on them -> **green** box, mount does not move.
- Enemy = person with no badge -> **red** box, face tracked, exact face
  centre shown as a dot + crosshair, servos aim at it in real time.
- Enemy clip: a 7-second clip (5–10 second range, `CLIP_SECONDS`) is
  recorded when an enemy is first seen, uploaded to Cloudinary if
  configured, and listed on the dashboard.
- Dashboard: live footage + evidence list only. Nothing else.
  Open http://127.0.0.1:8000

## Files (new, old files untouched)

- `run.py` — start the dashboard
- `app/core.py` — config, mapping, friend/enemy rule, clip timing
- `app/pipeline.py` — per-frame decisions (testable without a camera)
- `app/vision.py` — YOLO person/badge, Haar face, colour/shape fallback
- `app/server.py` — dashboard + MJPEG stream + evidence list
- `app/arduino.py` — serial link
- `app/evidence.py` — Cloudinary upload + local fallback
- `app/label_badge.py` — draw boxes on badge photos
- `app/train_badge.py` — train the badge detector
- `arduino/sentrygun_v2.ino` — upload to the Arduino (pan pin 7,
  tilt pin 10, LED 11, buzzer 12, 115200 baud). Servos move ONLY on
  an ENEMY command; FRIEND only turns the LED/buzzer off.

Old files (`sentry2.py`, `sentry3.py`, `sentry4.py`, `stream_viewer.py`,
`ardino/`, `badge_Images/`, `model/`, `yolov5s.pt`) are not changed.

## Run on your Mac

```bash
cd ~/Desktop/sentrygun
python3 -m pip install -r requirements.txt
python3 run.py
```

Then open http://127.0.0.1:8000

First run downloads `yolov8n.pt` (COCO) automatically. If your Arduino
is on a different port, set it before starting, e.g.
`SERIAL_PORT=/dev/cu.usbserial-110 python3 run.py`.
The old port was `/dev/cu.usbserial-120`.

Check the dashboard without a camera or model first:

```bash
DEMO=1 python3 run.py
```

## Train the badge detector (the real badge model)

Your badge photos are already here: 281 photos in
`badge_Images/badge` and 126 in `badge_Images/without badge`.
They are full photos, not labelled boxes, so labelling comes first:

```bash
python3 -m app.label_badge   # drag a box on the badge, s=save, n=next, q=quit
python3 -m app.train_badge   # needs at least 20 labelled photos
```

This creates `runs/badge_detector/weights/best.pt`. Restart the
dashboard and it will use the trained model instead of the colour
fallback. If you change the badge's colour or shape, label the new
badge and retrain — no code change is needed, because the model
learns colour and shape from your labels.

## Cloudinary evidence

Create a file named `.env` in this folder (it is local, never
committed anywhere):

```
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

Without these, clips are still saved in `evidence/` and listed on
the dashboard as local-only. Never put real keys in code or chat.

## Tests (in this workspace copy)

```bash
python3 -m pytest tests/ -q
```

10 tests pass: mapping, friend/enemy rule, badge-on-person rule,
servo-only-for-enemy, clip timing and cooldown, largest-enemy target.

## Honest limits — not yet proven on your Mac

- Real YOLO inference on your MacBook camera (needs the install above).
- Servo hardware moving (needs the Arduino sketch uploaded + the
  correct serial port).
- Cloudinary upload (needs your Cloudinary credentials in `.env`).
- Badge accuracy before training: the colour fallback will mistake
  any solid red object worn by a person for a badge. Train the badge
  model before relying on friend/enemy decisions.
- A person facing away, or with their face hidden, gives no face
  centre, so the mount holds its last position instead of guessing.
