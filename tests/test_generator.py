from __future__ import annotations

import numpy as np

from studio.editing.generator import blank_trajectory, generate_from_keyframes


def test_blank_trajectory_and_keyframe_generator() -> None:
    blank = blank_trajectory({"q0": 0.2, "q1": -0.1}, duration=1.0, hz=10.0)
    assert blank.duration == np.float64(1.0) or blank.duration == 1.0
    assert float(blank.duration) == 1.0
    assert blank.channels["q0"][0] == 0.2
    generated = generate_from_keyframes(
        [
            (0.0, {"q0": 0.0, "q1": 1.0}),
            (1.0, {"q0": 1.0, "q1": 1.0}),
        ],
        hz=20.0,
    )
    assert generated.channels["q0"][0] == 0.0
    assert generated.channels["q0"][-1] == 1.0
    assert generated.channels["q1"][-1] == 1.0
    assert generated.metadata["generator"] == "keyframes"
