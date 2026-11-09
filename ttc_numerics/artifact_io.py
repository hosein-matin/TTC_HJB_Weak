"""Publish complete numerical archives after closing and validating each member."""
import os
from pathlib import Path
import tempfile
import numpy as np


def save_npz_atomic(path, **arrays):
    """Close and read every saved array before publishing a complete archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz',
                dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as check:
            for key in check.files:
                value = check[key]
                if value.shape != np.asarray(arrays[key]).shape:
                    raise IOError(f'Incomplete saved array: {key}')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

