import pytest

from racing.__main__ import parse_args
from racing.ui import SETTINGS, Setting, adjust, format_value


def setting(key: str) -> Setting:
    return next(s for s in SETTINGS if s.key == key)


def test_adjust_linear_respects_bounds():
    s = setting("num_cars")
    assert adjust(s, 16, +1) == 17
    assert adjust(s, 1, -1) == 1
    assert adjust(s, 64, +1) == 64


def test_adjust_log_multiplies_and_divides():
    s = setting("learning_rate")
    assert adjust(s, 0.001, +1) == pytest.approx(0.002)
    assert adjust(s, 0.001, -1) == pytest.approx(0.0005)


def test_adjust_bool_toggles():
    assert adjust(setting("reverse"), False, +1) is True


def test_adjust_float_keeps_float():
    assert isinstance(adjust(setting("max_speed"), 7.0, +1), float)


def test_every_setting_exists_in_defaults(defaults):
    for s in SETTINGS:
        assert s.key in defaults[s.section]


@pytest.mark.parametrize("value, text", [(True, "AÇIK"), (1500, "1.500"), (0.0005, "0.0005"), (0.99, "0.99")])
def test_format_value(value, text):
    assert format_value(value) == text


def test_cli_defaults():
    args = parse_args([])
    assert not args.headless and args.episodes == 200


def test_cli_new_track_choices():
    assert parse_args(["--new-track", "zor", "--seed", "3"]).new_track == "zor"
    with pytest.raises(SystemExit):
        parse_args(["--new-track", "imkansız"])


def test_cli_rejects_non_positive_episodes():
    with pytest.raises(SystemExit):
        parse_args(["--episodes", "0"])
