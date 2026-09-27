import numpy as np
import pytest

from racing import trackgen
from racing.config import DEFAULTS
from racing.track import Track


@pytest.mark.parametrize("difficulty", ["kolay", "orta", "zor"])
def test_generated_track_passes_analysis(difficulty):
    g = trackgen.generate(difficulty, seed=7, track_cfg=DEFAULTS["track"])
    track = Track(g.rgb, DEFAULTS["track"], name=g.name)
    assert track.length > 800
    assert g.rgb.shape == (trackgen.IMAGE_SIZE[1], trackgen.IMAGE_SIZE[0], 3)
    assert g.difficulty.width[0] <= g.width <= g.difficulty.width[1]


def test_same_seed_gives_same_track():
    a = trackgen.generate("orta", seed=42)
    b = trackgen.generate("orta", seed=42)
    np.testing.assert_array_equal(a.rgb, b.rgb)
    assert a.name == "orta_42"


def test_different_seeds_differ():
    assert not np.array_equal(trackgen.generate("orta", seed=1).rgb, trackgen.generate("orta", seed=2).rgb)


def test_harder_tracks_are_narrower():
    widths = [trackgen.DIFFICULTIES[k].width[1] for k in ("kolay", "orta", "zor")]
    assert widths == sorted(widths, reverse=True)


def test_random_difficulty_resolves_to_known_level():
    g = trackgen.generate("rastgele", seed=3)
    assert g.difficulty.key in trackgen.DIFFICULTIES


def test_unknown_difficulty_raises():
    with pytest.raises(ValueError, match="Bilinmeyen"):
        trackgen.generate("imkansız", seed=1)


def test_well_formed_rejects_overlapping_loop():
    # Kendine çok yaklaşan (8 biçimine yakın) bir döngü
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    curve = np.stack([np.sin(2 * t) * 300 + 600, np.sin(t) * 20 + 400], axis=1)
    assert not trackgen.is_well_formed(curve, width=60)


def test_well_formed_accepts_big_circle():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    curve = np.stack([np.cos(t) * 300 + 600, np.sin(t) * 300 + 400], axis=1)
    assert trackgen.is_well_formed(curve, width=60)


def test_save_writes_png(tmp_path):
    g = trackgen.generate("kolay", seed=5)
    path = trackgen.save(g, tmp_path)
    assert path.name == "kolay_5.png" and path.stat().st_size > 0
    loaded = Track.from_file(path, DEFAULTS["track"])
    assert loaded.name == "kolay_5"
