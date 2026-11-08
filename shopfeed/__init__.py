"""Shop feed recommendation system prototype."""
import warnings as _w

__version__ = "0.1.0"

# numpy 2.0 on macOS Accelerate emits spurious RuntimeWarnings from matmul; results are correct.
_w.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
_w.filterwarnings("ignore", message=".*requires_grad=True to a scalar", category=UserWarning)
_w.filterwarnings("ignore", message=".*concatenation with empty or all-NA", category=FutureWarning)
