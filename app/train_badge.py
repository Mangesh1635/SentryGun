"""Train a YOLO badge detector from the badge's own photos.

Step 1 - label:  python3 -m app.label_badge
    Opens each photo in badge_Images/badge, you drag a box around
    the badge, press 's' to save, 'n' for next, 'q' to quit.
    Labels are saved as YOLO .txt files next to a labels/ folder.
    Label at least 20-30 photos; more, from different angles and
    distances, is better. The model learns the badge's actual colour
    AND shape from these boxes - nothing about colour or shape is
    hardcoded here.

Step 2 - train:  python3 -m app.train_badge [--epochs 50]
    Photos with a label become positive examples. Photos in
    badge_Images/without badge become negative examples (no badge).
    Split is 80% train / 20% validation, deterministic by filename.
    Output: runs/badge_detector/weights/best.pt - the file the
    dashboard looks for. Once it exists, the dashboard uses it and
    stops using the red colour fallback.

Run from the sentrygun folder (or pass --root <folder>).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys


def _link_or_copy(src: str, dst: str):
    # Re-running the script must not fail on, or copy over, the
    # symlink left by the previous run. Remove it first.
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_dataset(root: str, out: str = "dataset"):
    badge_dir = os.path.join(root, "badge_Images", "badge")
    no_badge_dir = os.path.join(root, "badge_Images", "without badge")
    label_dir = os.path.join(root, "labels")
    labelled = []
    for name in sorted(os.listdir(badge_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(name)[0]
        label = os.path.join(label_dir, stem + ".txt")
        if os.path.exists(label):
            labelled.append((os.path.join(badge_dir, name), label))
    if len(labelled) < 20:
        print(f"[ERROR] Only {len(labelled)} labelled badge photos found "
              f"in {label_dir}. Label at least 20 first:\n"
              "  python3 -m app.label_badge")
        sys.exit(1)

    negatives = []
    if os.path.isdir(no_badge_dir):
        for name in sorted(os.listdir(no_badge_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                negatives.append(os.path.join(no_badge_dir, name))

    rng = random.Random(42)
    rng.shuffle(labelled)
    split = int(len(labelled) * 0.8)
    train_pos, val_pos = labelled[:split], labelled[split:]
    rng.shuffle(negatives)
    n_val_neg = len(negatives) // 5
    val_neg, train_neg = negatives[:n_val_neg], negatives[n_val_neg:]

    for split_name in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, split_name, sub), exist_ok=True)
    for split_name, pos, neg in (("train", train_pos, train_neg),
                                 ("val", val_pos, val_neg)):
        for src, label in pos:
            dst = os.path.join(out, split_name, "images",
                               os.path.basename(src))
            _link_or_copy(src, dst)
            shutil.copyfile(
                label,
                os.path.join(out, split_name, "labels",
                             os.path.basename(label)))
        for src in neg:
            dst = os.path.join(out, split_name, "images",
                               os.path.basename(src))
            _link_or_copy(src, dst)
            # Negative example: an empty label file means "no badge".
            open(os.path.join(out, split_name, "labels",
                              os.path.splitext(os.path.basename(src))[0]
                              + ".txt"), "w").close()
    # BUG FIX: "path: ." was the failure. Ultralytics resolves that
    # relative to the current working directory (sentrygun/), so it
    # looked for sentrygun/val/images and crashed. Use the absolute
    # dataset folder instead, so train/val resolve inside it
    # regardless of where the command is run from.
    out_abs = os.path.abspath(out)
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write(f"path: {out_abs}\ntrain: train/images\n"
                 "val: val/images\nnames:\n  0: badge\n")
    print(f"[INFO] Dataset: {len(train_pos)} train + {len(val_pos)} val "
          f"badge photos, {len(train_neg)} train + {len(val_neg)} val "
          f"without-badge photos.")
    return os.path.join(out, "data.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    data_yaml = build_dataset(args.root)
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # COCO-pretrained, fine-tuned for badge
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz,
                project="runs", name="badge_detector")
    # Ultralytics versions differ in where they actually put the run
    # folder (your log showed runs/detect/runs/badge_detector).
    # The dashboard only looks in one place, so copy the result
    # there explicitly instead of hoping the folders line up.
    expected = os.path.join("runs", "badge_detector", "weights",
                            "best.pt")
    if not os.path.exists(expected):
        found = None
        for dirpath, _dirnames, filenames in os.walk("runs"):
            if "best.pt" in filenames and "badge_detector" in dirpath:
                found = os.path.join(dirpath, "best.pt")
                break
        if found is not None and os.path.abspath(found) != \
                os.path.abspath(expected):
            os.makedirs(os.path.dirname(expected), exist_ok=True)
            shutil.copyfile(found, expected)
            print(f"[INFO] Copied trained model from {found} to "
                  f"{expected}")
    print("[INFO] Done. Best model: "
          "runs/badge_detector/weights/best.pt")


if __name__ == "__main__":
    main()
