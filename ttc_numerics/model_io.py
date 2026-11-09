"""CPU NumPy evaluation of the supplied PyTorch state-dict checkpoints.

The restricted loader accepts only the storage/tensor constructors used by
these float32 state dictionaries; it never resolves arbitrary pickle globals.
Training remains available in train.py with PyTorch installed. Inference here
uses the same physical-coordinate affine normalization as the supplied run.
"""
from collections import OrderedDict
from pathlib import Path
import io
import pickle
import zipfile
import numpy as np
import ttc as T
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(*args, **kwargs): return nullcontext()

LO = np.array([T.BOX["s"][0], T.BOX["Xi"][0], T.BOX["Lam"][0], T.BOX["zeta"][0]], np.float32)
HI = np.array([T.BOX["s"][1], T.BOX["Xi"][1], T.BOX["Lam"][1], T.BOX["zeta"][1]], np.float32)


def norm(z):
    return 2 * (np.asarray(z, dtype=np.float32) - LO) / (HI - LO) - 1


def _rebuild(storage, offset, size, stride, requires_grad, hooks, *unused):
    size, stride = tuple(size), tuple(stride)
    if offset < 0 or any(v < 0 for v in size + stride):
        raise ValueError("Unsupported tensor indexing")
    top = offset + sum((n - 1) * st for n, st in zip(size, stride))
    if top >= len(storage):
        raise ValueError("Tensor exceeds its storage")
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
           strides=tuple(st * storage.itemsize for st in stride), writeable=False).copy()


def load_state_dict(path):
    """Read these plain float32 Torch zip state dictionaries without Torch."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pk = [x for x in names if x.endswith("/data.pkl")]
        if len(pk) != 1:
            raise ValueError("Expected a single state-dict pickle")
        prefix = pk[0][:-len("data.pkl")]
        byteorder = archive.read(prefix + "byteorder").decode().strip()
        dtype = np.dtype("<f4" if byteorder == "little" else ">f4")
        cache = {}
        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                allow = {("collections", "OrderedDict"): OrderedDict,
                         ("torch._utils", "_rebuild_tensor_v2"): _rebuild,
                         ("torch", "FloatStorage"): np.float32}
                if (module, name) not in allow:
                    raise pickle.UnpicklingError("Unsupported checkpoint global: " + module + "." + name)
                return allow[(module, name)]
            def persistent_load(self, key):
                kind, typ, name, location, count = key
                if kind != "storage" or typ is not np.float32:
                    raise pickle.UnpicklingError("Unsupported storage")
                if name not in cache:
                    buf = archive.read(prefix + "data/" + name)
                    if len(buf) != count * 4:
                        raise ValueError("Storage size mismatch")
                    cache[name] = np.frombuffer(buf, dtype=dtype).astype(np.float32, copy=False)
                return cache[name]
        state = Reader(io.BytesIO(archive.read(pk[0]))).load()
    expected = {f"net.{k}.{suffix}" for k in (0,2,4,6) for suffix in ("weight","bias")}
    if set(state) != expected:
        raise ValueError("Checkpoint architecture differs from the supplied 4-400-400-400-1 MLP")
    return state


class NumpyMLP:
    def __init__(self, checkpoint, batch_size=8192):
        self.state = load_state_dict(checkpoint)
        self.layers = [(self.state[f"net.{k}.weight"],self.state[f"net.{k}.bias"]) for k in (0,2,4,6)]
        self.batch_size = batch_size
        self.n_parameters = sum(x.size for x in self.state.values())
        self.n_relu_units = sum(b.size for _,b in self.layers[:-1])
    def __call__(self, z):
        z = np.asarray(z)
        result = np.empty(len(z), dtype=np.float64)
        with threadpool_limits(limits=2, user_api="blas"):
            for lo in range(0, len(z), self.batch_size):
                hi = min(lo+self.batch_size,len(z))
                x=norm(z[lo:hi])
                for k,(w,b) in enumerate(self.layers):
                    x = x @ w.T + b
                    if k < len(self.layers)-1:
                        np.maximum(x,0,out=x)
                result[lo:hi]=x[:,0]
        return result


def load_model(tag, directory="out"):
    return NumpyMLP(Path(directory)/f"model_{tag}.pt")
