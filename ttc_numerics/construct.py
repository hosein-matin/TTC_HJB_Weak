"""Complete tensor grids and an explicitly evaluated ReLU realization.

The grid is fixed before query points are drawn. Every grid node is retained:
there is no selection of supports using validation or test locations.

TensorGrid.evaluate(k=...) evaluates approximate ReLU products, not the exact
interpolant. It accelerates the same finite ReLU sum using its compact support:
only the at most 2**d nonzero hats contribute. full_sum verifies equivalence
on small grids. This acceleration is not a materialized dense graph timing.

For x in [0,1], q_k(x)=x-sum(4**(-j)*g_j(x),j=1..k), where
g(x)=2 relu(x)-4 relu(x-.5)+2 relu(x-1), interpolates x**2 on the dyadic
mesh. Thus 0 <= q_k(x)-x*x <= 4**(-(k+1)). The multiplier
clip(q_k((x+y)/2)-q_k(abs(x-y)/2),0,1) preserves a zero factor exactly,
has range [0,1], and error at most e_k=4**(-(k+1)) on [0,1]**2.
Sequential d-factor products have error at most (d-1)*e_k. All block
operations are affine or ReLU. Floating-point roundoff is not certified.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np
from artifact_io import save_npz_atomic



def relu(x):
    return np.maximum(x, 0.0)


def square_relu(x, k):
    """Dyadic square interpolant, implemented by k tent-map ReLU layers."""
    x = np.asarray(x, dtype=float)
    t, out = x.copy(), x.copy()
    for j in range(1, int(k) + 1):
        t = 2.0 * relu(t) - 4.0 * relu(t - 0.5) + 2.0 * relu(t - 1.0)
        out -= (4.0 ** (-j)) * t
    return out


def multiply_relu(x, y, k):
    """Zero-preserving approximate product on [0,1] inputs."""
    a = 0.5 * (x + y)
    b = 0.5 * (relu(x - y) + relu(y - x))
    p = square_relu(a, k) - square_relu(b, k)
    return relu(p) - relu(p - 1.0)


def product_relu(factors, k):
    factors = np.asarray(factors, dtype=float)
    if factors.shape[-1] == 0:
        raise ValueError("At least one factor is required")
    out = factors[..., 0]
    for j in range(1, factors.shape[-1]):
        out = multiply_relu(out, factors[..., j], k)
    return out


def realization_metadata(d, n_nodes, k, coefficient_bound=1.0):
    """Literal DAG counts, without sharing hats or product blocks.

    A univariate hat uses two absolute-value ReLUs and one outer ReLU.
    A multiplier uses two absolute-value ReLUs, 6*k square-block ReLUs,
    and two clipping ReLUs. Affine output nodes are not counted as ReLUs.
    These gate counts allow affine skip connections. The separately reported
    strictly layered realization also counts nonnegative identity/carry
    units: each multiplier needs 8*k+3 units, and each later factor is
    carried through k+2 layers until it is used. Neither count is a
    trainable-parameter count or a dense execution-time measurement.
    """
    e = 4.0 ** (-(int(k) + 1))
    per_multiplier = 6 * int(k) + 4
    per_node = 3 * int(d) + (int(d) - 1) * per_multiplier
    layered_per_node = (3 * int(d) + (int(d) - 1) * (8 * int(k) + 3)
                        + (int(k) + 2) * (int(d) - 1) * (int(d) - 2) // 2)
    return dict(square_layers=int(k), square_error_bound=e,
        product_error_bound=(int(d) - 1) * e,
        relu_units_per_multiplier=per_multiplier, relu_units_per_node=per_node,
        literal_relu_units=int(n_nodes) * per_node,
        literal_hidden_depth=2 + (int(d) - 1) * (int(k) + 2),
        gate_count_convention="unshared ReLU DAG with affine skip connections",
        strictly_layered_relu_units_per_node=layered_per_node,
        strictly_layered_relu_units=int(n_nodes) * layered_per_node,
        nodal_coefficients=int(n_nodes), coefficient_bound=float(coefficient_bound),
        realization_error_bound=float((2 ** int(d)) * coefficient_bound * (int(d) - 1) * e),
        nodal_error_amplification=float(1.0 + (2 ** int(d)) * (int(d) - 1) * e),
        evaluation="support-aware evaluation of explicit zero-preserving ReLU blocks")


@dataclass
class TensorGrid:
    lo: np.ndarray
    hi: np.ndarray
    nseg: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        self.lo, self.hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        self.nseg = np.asarray(self.nseg, int)
        if not (self.lo.shape == self.hi.shape == self.nseg.shape):
            raise ValueError("Grid geometry shapes do not agree")
        if np.any(self.hi <= self.lo) or np.any(self.nseg < 1):
            raise ValueError("Positive grid side lengths and segment counts required")
        self.dim = len(self.lo)
        self.mesh = (self.hi - self.lo) / self.nseg
        self.shape = tuple(self.nseg + 1)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.size != int(np.prod(self.shape)):
            raise ValueError("A complete grid value is required at EVERY node")
        self.values = self.values.reshape(self.shape)
        if not np.isfinite(self.values).all():
            raise ValueError("All complete-grid nodal coefficients must be finite")
        self.strides = np.array([np.prod(self.nseg[i+1:] + 1) for i in range(self.dim)], dtype=np.int64)
        self.offsets = np.array(list(itertools.product([0, 1], repeat=self.dim)), dtype=int)

    @classmethod
    def geometry(cls, lo, hi, h0):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        nseg = np.maximum(np.ceil((hi - lo) / float(h0) - 1e-12).astype(int), 1)
        return lo, hi, nseg

    @staticmethod
    def node_batches(lo, hi, nseg, batch_size=100000):
        lo, hi, nseg = np.asarray(lo), np.asarray(hi), np.asarray(nseg)
        shape = tuple(nseg + 1)
        n = int(np.prod(shape))
        mesh = (hi - lo) / nseg
        for a in range(0, n, batch_size):
            ids = np.arange(a, min(a + batch_size, n))
            multi = np.column_stack(np.unravel_index(ids, shape))
            yield a, lo + multi * mesh

    def active(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError("Queries must have shape (N, grid dimension)")
        tol = 32 * np.finfo(float).eps * max(1.0, float(np.abs(self.hi).max()))
        if np.any(z < self.lo - tol) or np.any(z > self.hi + tol):
            raise ValueError("Query outside the declared complete-grid box")
        u = (z - self.lo) / self.mesh
        cell = np.minimum(np.maximum(np.floor(u).astype(int), 0), self.nseg - 1)
        frac = np.minimum(np.maximum(u - cell, 0.0), 1.0)
        return cell, frac

    def evaluate(self, z, k=None, batch_size=4096):
        """Exact multilinear interpolant (k=None), or actual ReLU products."""
        z = np.asarray(z, dtype=float)
        out = np.zeros(len(z), dtype=float)
        flat = self.values.ravel()
        for a in range(0, len(z), batch_size):
            cell, frac = self.active(z[a:a + batch_size])
            ans = np.zeros(len(cell))
            for off in self.offsets:
                coeff = flat[((cell + off) * self.strides).sum(axis=1)]
                factors = np.where(off[None] == 1, frac, 1.0 - frac)
                weight = np.prod(factors, axis=1) if k is None else product_relu(factors, k)
                ans += coeff * weight
            out[a:a + len(ans)] = ans
        return out

    def full_sum(self, z, k, max_nodes=20000):
        """Literal global ReLU sum, also defined outside the approximation box."""
        if self.values.size > max_nodes:
            raise ValueError("Full-sum verification is intended for small grids")
        z = np.asarray(z, float)
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError("Queries must have shape (N, grid dimension)")
        out = np.zeros(len(z))
        for a, nodes in self.node_batches(self.lo, self.hi, self.nseg, batch_size=512):
            u = (z[:, None, :] - nodes[None, :, :]) / self.mesh
            hats = relu(1.0 - relu(u) - relu(-u))
            weights = product_relu(hats, k)
            out += weights @ self.values.ravel()[a:a + len(nodes)]
        return out

    def save(self, path, **extra):
        save_npz_atomic(path, lo=self.lo, hi=self.hi, nseg=self.nseg, values=self.values, **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as x:
            return cls(x['lo'], x['hi'], x['nseg'], x['values'])
