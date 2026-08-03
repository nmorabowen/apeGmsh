"""The exact-pixel classifiers still fail on a real regression.

``_frames_match_or_skip`` / ``_cleared_or_skip`` let three render
tests survive GL stacks that are not bit-deterministic (see the
conftest note). A tolerance that swallowed everything would delete
those contracts silently, so the band itself is pinned here: noise
skips, divergence FAILS, and an exact match stays a plain pass.
"""
from __future__ import annotations

import numpy as np
import pytest

_SHAPE = (40, 40, 3)


def _frame(fill: int = 0) -> np.ndarray:
    return np.full(_SHAPE, fill, dtype=np.uint8)


def _painted_frame() -> np.ndarray:
    """A frame with a known amount of ink (1000 painted px)."""
    img = _frame()
    img.reshape(-1, 3)[:1000] = 200
    return img


# ---------------------------------------------------------------
# frames_match_or_skip
# ---------------------------------------------------------------

def test_identical_frames_pass_without_skipping(frames_match_or_skip):
    img = _painted_frame()
    frames_match_or_skip(img, img.copy(), what="identical")


def test_one_pixel_one_level_skips(frames_match_or_skip):
    a = _painted_frame()
    b = a.copy()
    b[0, 0, 0] = np.uint8(int(b[0, 0, 0]) - 1)      # 1 px, 1 level
    with pytest.raises(pytest.skip.Exception, match="1 px differ"):
        frames_match_or_skip(a, b, what="noise")


def test_two_intensity_levels_is_not_noise(frames_match_or_skip):
    """Delta > 1 level is a real difference even on one pixel."""
    a = _painted_frame()
    b = a.copy()
    b[0, 0, 0] = np.uint8(int(b[0, 0, 0]) - 5)
    with pytest.raises(AssertionError, match="too large"):
        frames_match_or_skip(a, b, what="banding")


def test_many_pixels_fail_even_at_one_level(frames_match_or_skip):
    """The shape of a real fast-path break: lots of pixels move."""
    a = _painted_frame()
    b = a.copy()
    b.reshape(-1, 3)[:500, 0] -= 1                   # 500 px, 1 level
    with pytest.raises(AssertionError, match="too large"):
        frames_match_or_skip(a, b, what="edges lost")


def test_shape_mismatch_always_fails(frames_match_or_skip):
    with pytest.raises(AssertionError, match="frame shape"):
        frames_match_or_skip(
            np.zeros((4, 4, 3), np.uint8), np.zeros((5, 5, 3), np.uint8),
            what="shape",
        )


# ---------------------------------------------------------------
# cleared_or_skip
# ---------------------------------------------------------------

def test_fully_cleared_passes(cleared_or_skip):
    cleared_or_skip(0, 4900, what="clean")


def test_faint_residue_skips(cleared_or_skip):
    with pytest.raises(pytest.skip.Exception, match="22 px remain"):
        cleared_or_skip(22, 4900, what="residue")


def test_an_actor_still_drawn_fails(cleared_or_skip):
    """Half the ink left is a live actor, not driver residue."""
    with pytest.raises(AssertionError, match="still present"):
        cleared_or_skip(2450, 4900, what="not removed")
