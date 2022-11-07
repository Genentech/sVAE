"""scvi-tools-skeleton."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from . import metrics
from ._model import SpikeSlabVAE
from .baselines._model import sVAE
from .simulation import sparse_shift

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "svae"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("svae: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = ["sVAE", "SpikeSlabVAE", "sparse_shift", "metrics"]
