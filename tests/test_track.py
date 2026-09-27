import math
from itertools import pairwise

import numpy as np
import pytest

from racing.track import Track, TrackError, box_sum, canonical_normal, geodesic_distance
from tests.conftest import RING_CENTER, RING_INNER, RING_OUTER, make_ring_rgb

MID_RADIUS = (RING_INNER + RING_OUTER) / 2


def ring_point(angle_deg: float, radius: float = MID_RADIUS) -> tuple[float, float]:
    """Ekran koordinatında (y aşağı) açıya göre halka noktası; 0° = tepe, saat yönünde artar."""
    a = math.radians(angle_deg)
    return RING_CENTER + radius * math.sin(a), RING_CENTER - radius * math.cos(a)


def test_start_line_found_and_heading_points_right(ring_track):
    x, y = ring_track.start_center
    assert abs(x - RING_CENTER) < 2
    assert RING_CENTER - RING_OUTER < y < RING_CENTER - RING_INNER
    assert math.isclose(ring_track.start_heading, 0.0, abs_tol=0.05)


def test_lap_length_is_between_inner_and_mid_circumference(ring_track):
    assert 2 * math.pi * RING_INNER * 0.95 < ring_track.length < 2 * math.pi * MID_RADIUS


def test_progress_increases_clockwise(ring_track):
    values = [float(ring_track.progress_at(*ring_point(a))) for a in range(10, 351, 20)]
    assert all(b > a for a, b in pairwise(values))
    assert values[0] < 0.1 * ring_track.length
    assert values[-1] > 0.9 * ring_track.length


@pytest.mark.parametrize("radius", [RING_INNER + 6, MID_RADIUS, RING_OUTER - 6])
def test_direction_is_tangent_to_ring_across_width(ring_track, radius):
    for angle in range(20, 350, 30):
        d = ring_track.direction_at(*ring_point(angle, radius))
        tangent = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        assert float(d @ tangent) > 0.97, f"açı {angle}°, yarıçap {radius}"


def test_reverse_flips_heading_and_progress(config):
    cfg = {**config["track"], "reverse": True}
    track = Track(make_ring_rgb(), cfg)
    assert math.isclose(abs(track.start_heading), math.pi, abs_tol=0.05)
    assert track.progress_at(*ring_point(30)) > track.progress_at(*ring_point(330))


def test_on_track_handles_out_of_bounds(ring_track):
    xs = np.array([-5.0, 1e6, *ring_point(90)[:1]])
    ys = np.array([10.0, 10.0, ring_point(90)[1]])
    assert ring_track.on_track(xs, ys).tolist() == [False, False, True]


def test_works_without_start_line(config):
    track = Track(make_ring_rgb(start_line=False), config["track"])
    assert track.length > 2 * math.pi * RING_INNER * 0.9


def test_open_track_raises(config):
    with pytest.raises(TrackError, match="döngü"):
        Track(make_ring_rgb(closed=False), config["track"])


def test_blank_image_raises(config):
    with pytest.raises(TrackError, match="sürülebilir"):
        Track(np.full((100, 100, 3), 255, dtype=np.uint8), config["track"])


def test_from_file_missing_raises(config, tmp_path):
    with pytest.raises(TrackError, match="bulunamadı"):
        Track.from_file(tmp_path / "yok.png", config["track"])


def test_geodesic_distance_straight_corridor():
    passable = np.ones((3, 10), dtype=bool)
    seeds = np.zeros_like(passable)
    seeds[:, 0] = True
    dist = geodesic_distance(passable, seeds)
    assert dist[1, 9] == pytest.approx(9.0)


def test_box_sum_matches_naive():
    a = np.arange(30, dtype=float).reshape(5, 6)
    got = box_sum(a, 1)
    padded = np.pad(a, 1)
    naive = np.array([[padded[y:y + 3, x:x + 3].sum() for x in range(6)] for y in range(5)])
    np.testing.assert_allclose(got, naive)


def test_canonical_normal_points_right_or_down():
    np.testing.assert_allclose(canonical_normal(np.array([-1.0, 0.0])), [1.0, 0.0])
    np.testing.assert_allclose(canonical_normal(np.array([0.0, -1.0])), [0.0, 1.0])
