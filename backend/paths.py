"""Where things live.

Every path is worked out from this file's own location, so the commands behave
the same whether you run them from the repo root, from inside backend/, or
from a container with a different working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FRONTEND = ROOT / "frontend"
MODEL_DIR = ROOT / "model"
CHECKPOINTS = ROOT / "checkpoints"
TB_LOGS = ROOT / "tb_logs"
