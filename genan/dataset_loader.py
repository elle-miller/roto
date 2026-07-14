"""Isaac-free import shim for `uan_shadowlite`'s `AlignedTrajectoryDataset`.

`roto.tasks.uan_shadowlite.__init__` imports `task.py`, which imports
`isaaclab` -- so a normal `from roto.tasks.uan_shadowlite.dataset import
...` would transitively boot Isaac Lab just to load a plain-torch dataset
class. `dataset.py` itself has zero internal package dependencies (see its
own module docstring), so it can be imported directly, bypassing the
package `__init__.py` -- the same trick
`roto/tests/uan_shadowlite/test_dataset.py` already uses.
"""

from __future__ import annotations

import os
import sys

_UAN_SHADOWLITE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roto", "tasks", "uan_shadowlite")
)
if _UAN_SHADOWLITE_DIR not in sys.path:
    sys.path.insert(0, _UAN_SHADOWLITE_DIR)

from dataset import COUPLED_JOINT_PAIRS, AlignedTrajectoryDataset, DatasetKeys, TrajectoryDataset  # noqa: E402

__all__ = ["AlignedTrajectoryDataset", "TrajectoryDataset", "DatasetKeys", "COUPLED_JOINT_PAIRS"]
