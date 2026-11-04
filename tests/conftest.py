"""Load the numbered analysis scripts as modules so they can be unit-tested.

Script file names start with digits, so they cannot be imported normally.
Importing a script must not run the analysis: every script guards its entry
point with ``if __name__ == "__main__"``.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(filename):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(
        "script_" + path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def bench():
    return load_script("06_corum_benchmark.py")


@pytest.fixture(scope="session")
def boot():
    return load_script("10_paired_bootstrap.py")


@pytest.fixture(scope="session")
def heldout():
    return load_script("08_heldout_perturbation_prediction.py")


@pytest.fixture(scope="session")
def adjust():
    return load_script("03_adjust_effect_score.py")


@pytest.fixture(scope="session")
def effect():
    pytest.importorskip("scanpy")
    return load_script("02_effect_matrix.py")


@pytest.fixture(scope="session")
def autoenc():
    pytest.importorskip("torch")
    return load_script("09_autoencoder_benchmark.py")
