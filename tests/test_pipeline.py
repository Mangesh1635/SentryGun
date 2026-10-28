import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core import Config
from app.pipeline import Pipeline


def make_pipeline():
    cfg = Config(servo_update_seconds=0.0)
    return Pipeline(cfg)


def test_friend_gets_green_box_no_servo_no_evidence():
    p = make_pipeline()
    result = p.process(
        [{"box": (100, 100, 300, 400), "conf": 0.9}],
        badges=[(150, 150, 180, 180)],
        faces={0: (150, 110, 210, 170)},
        frame_width=640, frame_height=480, now=100.0)
    assert result["persons"][0]["label"] == "FRIEND"
    assert result["persons"][0]["color"] == (0, 255, 0)
    assert result["persons"][0]["face_center"] is None  # face only for enemy
    assert result["target"] is None
    assert result["evidence_event"] == ""


def test_enemy_gets_red_box_face_point_servo_and_clip():
    p = make_pipeline()
    result = p.process(
        [{"box": (100, 100, 300, 400), "conf": 0.9}],
        badges=[],
        faces={0: (150, 110, 210, 170)},
        frame_width=640, frame_height=480, now=100.0)
    person = result["persons"][0]
    assert person["label"] == "ENEMY"
    assert person["color"] == (0, 0, 255)
    assert person["face_center"] == (180, 140)
    assert result["target"]["smooth_center"] == (180, 140)
    assert result["servo"] == "PAN:50;TILT:52;ENEMY\n"
    assert result["evidence_event"] == "started"
    assert result["evidence_recording"] is True


def test_servo_targets_largest_enemy_not_a_friend():
    p = make_pipeline()
    result = p.process(
        [
            {"box": (0, 0, 50, 50), "conf": 0.9},       # small enemy
            {"box": (100, 100, 400, 400), "conf": 0.9},  # large enemy
            {"box": (500, 100, 600, 300), "conf": 0.9},  # friend
        ],
        badges=[(520, 150, 550, 180)],
        faces={0: (10, 10, 30, 30), 1: (200, 120, 260, 180),
               2: (520, 110, 580, 170)},
        frame_width=640, frame_height=480, now=100.0)
    # Largest enemy face centre is (230, 150).
    assert result["target"]["face_center"] == (230, 150)
    assert result["persons"][2]["label"] == "FRIEND"
