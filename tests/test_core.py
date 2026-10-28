import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.core import (
    Config, EvidenceState, Smoother, badge_on_person, box_center,
    classify_person, face_to_servo, map_to_angle, servo_line,
)


def test_box_center_and_mapping():
    assert box_center((10, 20, 30, 60)) == (20, 40)
    assert map_to_angle(0, 0, 640) == 0
    assert map_to_angle(640, 0, 640) == 180
    assert map_to_angle(320, 0, 640) == 90
    assert map_to_angle(-50, 0, 640) == 0   # clamped
    assert map_to_angle(9999, 0, 640) == 180  # clamped


def test_face_center_maps_to_servo_angles():
    cfg = Config()
    assert face_to_servo((320, 240), 640, 480, cfg) == (90, 90)
    assert face_to_servo((0, 0), 640, 480, cfg) == (0, 0)


def test_badge_must_be_on_the_person():
    person = (100, 100, 300, 400)
    assert classify_person(person, [(150, 150, 180, 180)]) == "FRIEND"
    assert classify_person(person, []) == "ENEMY"
    # Badge on someone else in the frame does NOT make this person a friend.
    assert classify_person(person, [(500, 500, 530, 530)]) == "ENEMY"
    assert badge_on_person(person, [(500, 500, 530, 530)]) is False


def test_servo_moves_only_for_enemy():
    assert servo_line(90, 90, True) == "PAN:90;TILT:90;ENEMY\n"
    assert servo_line(90, 90, False) == "FRIEND\n"


def test_smoother_converges_and_resets():
    s = Smoother(0.5)
    assert s.update((100, 100)) == (100, 100)
    assert s.update((200, 200)) == (150, 150)
    s.reset()
    assert s.update((10, 20)) == (10, 20)


def test_clip_seconds_must_be_5_to_10():
    with pytest.raises(ValueError):
        Config(clip_seconds=4)
    with pytest.raises(ValueError):
        Config(clip_seconds=11)
    assert Config(clip_seconds=5).clip_seconds == 5
    assert Config(clip_seconds=10).clip_seconds == 10


def test_evidence_state_machine():
    ev = EvidenceState(clip_seconds=7, cooldown_seconds=10)
    assert ev.update(False, 0) == ""
    assert ev.update(True, 0) == "started"
    assert ev.recording is True
    assert ev.update(True, 5) == ""       # still recording
    assert ev.update(True, 7) == "finished"
    assert ev.recording is False
    assert ev.update(True, 10) == ""      # cooldown, no new clip yet
    assert ev.update(True, 17) == "started"  # cooldown over, enemy again
