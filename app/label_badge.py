"""Label badge photos for training. Drag a box around the badge.

Keys:  drag mouse to draw a box, 's' save this photo's label,
       'n' next photo without saving, 'q' quit.
Labels are YOLO format (class 0, centre x/y, width, height, all
normalised) in labels/<photo-stem>.txt, next to the sentrygun folder.
Run from the sentrygun folder:  python3 -m app.label_badge
"""

from __future__ import annotations

import os
import sys


def main():
    import cv2

    badge_dir = os.path.join("badge_Images", "badge")
    label_dir = "labels"
    os.makedirs(label_dir, exist_ok=True)
    photos = sorted(
        f for f in os.listdir(badge_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not photos:
        print(f"[ERROR] No photos in {badge_dir}")
        sys.exit(1)

    state = {"start": None, "box": None, "done": False}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"]:
            x0, y0 = state["start"]
            state["box"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            state["start"] = None

    cv2.namedWindow("Label badge - drag a box, s=save, n=next, q=quit")
    cv2.setMouseCallback("Label badge - drag a box, s=save, n=next, q=quit",
                         on_mouse)
    for name in photos:
        stem = os.path.splitext(name)[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        if os.path.exists(label_path):
            continue  # already labelled
        img = cv2.imread(os.path.join(badge_dir, name))
        if img is None:
            continue
        state["box"] = None
        while True:
            show = img.copy()
            if state["box"]:
                x1, y1, x2, y2 = state["box"]
                cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Label badge - drag a box, s=save, n=next, q=quit",
                       show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break
            if key == ord("s") and state["box"]:
                h, w = img.shape[:2]
                x1, y1, x2, y2 = state["box"]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                with open(label_path, "w") as fh:
                    fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print(f"[INFO] Saved {label_path}")
                break
    cv2.destroyAllWindows()
    print("[INFO] All badge photos labelled (or already labelled).")


if __name__ == "__main__":
    main()
