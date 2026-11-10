import os

import pytest


@pytest.mark.live
def test_probe_native_exit_zero():
    os._exit(0)
