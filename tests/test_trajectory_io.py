from __future__ import annotations

import numpy as np
import pytest

from studio.core.trajectory import TrajectoryData, infer_frequency
from studio.io.trajectory_io import export_trajectory, load_trajectory
import studio.io.trajectory_io as trajectory_io


@pytest.mark.parametrize("hz", [50.0, 100.0, 250.0, 1000.0])
def test_frequency_inference(hz: float) -> None:
    times = np.arange(1001) / hz
    info = infer_frequency(times)
    assert info.hz == pytest.approx(hz)
    assert not info.irregular


def test_irregular_frequency_is_reported() -> None:
    times = np.arange(100, dtype=float) / 100
    times[50:] += 0.003
    info = infer_frequency(times)
    assert info.irregular
    assert info.max_jitter > 0.002


def test_csv_without_time_requires_frequency(tmp_path) -> None:
    source = tmp_path / "untimed.csv"
    source.write_text("q0,q1\n0,1\n1,2\n2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control frequency"):
        load_trajectory(source)
    loaded = load_trajectory(source, 50)
    np.testing.assert_allclose(loaded.times, [0, 0.02, 0.04])


def test_csv_unedited_roundtrip(tmp_path) -> None:
    times = np.arange(501) / 250
    source = TrajectoryData(
        times,
        {
            "q0": np.sin(times),
            "q1": np.cos(times),
            "dq0": np.cos(times),
            "custom": np.arange(len(times), dtype=float),
        },
        ["q0", "q1", "custom"],
        {"q0": "dq0"},
    )
    path = export_trajectory(source, tmp_path / "roundtrip.csv")
    loaded = load_trajectory(path)
    np.testing.assert_array_equal(loaded.times, source.times)
    for name in source.channels:
        np.testing.assert_array_equal(loaded.channels[name], source.channels[name])


def test_npz_matrix_and_key_mapping(tmp_path) -> None:
    path = tmp_path / "matrix.npz"
    positions = np.arange(30, dtype=float).reshape(10, 3)
    np.savez(
        path,
        q=positions,
        joint_names=np.asarray(["shoulder", "elbow", "wrist"]),
        hz=np.asarray(100.0),
        sensor=np.arange(10, dtype=float),
    )
    loaded = load_trajectory(path)
    assert loaded.position_channels == ["shoulder", "elbow", "wrist"]
    assert loaded.frequency.hz == pytest.approx(100)
    np.testing.assert_array_equal(loaded.channels["elbow"], positions[:, 1])
    np.testing.assert_array_equal(loaded.channels["sensor"], np.arange(10))


def test_resampling_preserves_duration_and_endpoints() -> None:
    times = np.arange(51) / 50
    data = TrajectoryData(times, {"q0": times**2}, ["q0"])
    output = data.resample(250)
    assert output.frequency.hz == pytest.approx(250)
    assert output.duration == pytest.approx(data.duration)
    assert output.channels["q0"][0] == pytest.approx(data.channels["q0"][0])
    assert output.channels["q0"][-1] == pytest.approx(data.channels["q0"][-1])


@pytest.mark.parametrize("suffix", [".csv", ".npz"])
def test_atomic_export_roundtrip_precision(tmp_path, suffix: str) -> None:
    times = np.arange(101) / 1000
    data = TrajectoryData(
        times,
        {"q0": np.sin(times), "custom": np.sqrt(times)},
        ["q0", "custom"],
    )
    path = export_trajectory(data, tmp_path / f"motion{suffix}")
    loaded = load_trajectory(path)
    np.testing.assert_allclose(loaded.times, data.times, rtol=0, atol=1e-15)
    for name in data.channels:
        np.testing.assert_allclose(
            loaded.channels[name],
            data.channels[name],
            rtol=0,
            atol=1e-15,
        )


def test_export_refuses_source_overwrite(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("time,q0\n0,0\n.01,.1\n", encoding="utf-8")
    data = load_trajectory(source)
    with pytest.raises(ValueError, match="source"):
        export_trajectory(data, source)


def test_failed_atomic_export_keeps_existing_file(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "motion.csv"
    destination.write_text("original", encoding="utf-8")
    data = TrajectoryData(
        np.arange(3) / 100,
        {"q0": np.arange(3, dtype=float)},
        ["q0"],
    )

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(trajectory_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        export_trajectory(data, destination)
    assert destination.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".motion.csv.*")) == []


def test_npz_export_rejects_reserved_channel_names(tmp_path) -> None:
    data = TrajectoryData(
        np.arange(3) / 100,
        {"q0": np.zeros(3), "frequency_hz": np.ones(3)},
        ["q0", "frequency_hz"],
    )
    with pytest.raises(ValueError, match="reserved metadata"):
        export_trajectory(data, tmp_path / "reserved.npz")


def test_csv_treats_dexterous_hand_names_as_positions(tmp_path) -> None:
    source = tmp_path / "o6_hands.csv"
    source.write_text(
        "time,q0,lh_index_mcp_pitch,rh_thumb_cmc_yaw\n"
        "0,0.1,0.2,0.3\n"
        "0.02,0.1,0.4,0.3\n",
        encoding="utf-8",
    )
    loaded = load_trajectory(source)
    assert loaded.position_channels == [
        "q0",
        "lh_index_mcp_pitch",
        "rh_thumb_cmc_yaw",
    ]
    np.testing.assert_allclose(loaded.channels["lh_index_mcp_pitch"], [0.2, 0.4])
