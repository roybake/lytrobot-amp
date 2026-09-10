"""Training entry points must not import a neighboring editable checkout."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_multimotion_resolves_local_package_before_first_import():
    environment = dict(os.environ, PYTHONPATH='')
    result = subprocess.run(
        [sys.executable, '-c',
         'import runpy, sys; runpy.run_path(sys.argv[1]); '
         'import lytrobot_amp.env; print(lytrobot_amp.env.__file__)',
         str(ROOT / 'scripts/train_multimotion.py')],
        cwd='/tmp', env=environment, check=True, text=True, capture_output=True,
    )
    assert Path(result.stdout.strip()).resolve() == ROOT / 'src/lytrobot_amp/env.py'
