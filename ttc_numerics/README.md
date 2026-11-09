# TTC / stopped-step Bellman experiments

This package accompanies **A Derivative-Free Bellman Equation for Dynamic
Time-to-Collision: Two-Sided Certificates and Explicit ReLU Approximation**.
The accompanying manuscript uses the results in `out/`. The supplied learned
checkpoints are unchanged; the experiments evaluate them without
retraining. All model and construction comparisons use independent queries.

## Reproduce the reported results

Python 3.12 was used. The exact evaluation dependencies are in
`requirements.txt`; the environment record is `out/runtime_environment.json`.
PyTorch is **not required** to evaluate the supplied checkpoints. `model_io.py`
reads the restricted state-dictionary format with NumPy and uses the original
float32 weights and normalization.

From this directory:

```bash
python3 -m pip install -r requirements.txt
sh run_all.sh
```

To check the complete saved outputs without rerunning the simulations:

```bash
python3 verify_results.py
```

The verifier writes `out/reproduction_checks.json`. The standalone numerical
archive includes the same complete saved outputs as the repository.

The default pipeline runs, in order:

```bash
python3 reference_checks.py
python3 evaluate_models.py
python3 construction_experiment.py
python3 iteration_experiment.py
python3 figures.py
python3 verify_results.py
```

`experiments.py` sets two BLAS/OpenMP threads for its subprocesses. To reuse the
saved complete construction grids, run `sh run_all.sh --reuse-grids`. Every
other calculation and independent query evaluation still runs. Runtime varies
with hardware; the full evaluation pipeline is intended to take minutes,
not the time required for retraining.

The reference solver, model evaluation, construction and sweep experiments
are separate scripts so each can be rerun independently. Outputs are written
to `out/`. Construction and figure commands accept alternative checkpoint
paths; the default is `provenance/checkpoints`. No file under `provenance/`
is modified by the default pipeline.

## What is being computed

The prediction uses `alpha=2`, horizon 1, and leader acceleration
`(0.5,-1.2,0.8,-0.4,0)` on successive fifths of the horizon. The sampled
physical coordinates are independent uniforms in
`(s,Xi,vF,zeta_L) in [0,1) x (0,1] x [0,2*tanh(2)] x [.25,1.75]`, with
`Lambda=zeta_L-vF`. The sampling cap `2*tanh(2)` is distinct from the true
global optimal-velocity supremum `1+tanh(2)`.

The event-aware RK4 reference preserves the initial time and splits steps at
forcing discontinuities, snapshots and the horizon. Initial contact and
terminal states stop immediately. Endpoint crossings and closing-to-opening
interior minima are checked. The solver is numerical, not an interval or
validated event solver; arbitrary grazing classifications are not certified.

### Broad-domain learned-model test

`evaluate_models.py` first reconstructs the original rounded-time test
coordinates, then evaluates a new 300,000-point continuous-time holdout with
seed 20261103. Its selected collision subset has 2,725 points with TTC at
least .1, impact relative velocity at most -.5, and at least .2 horizon
margin. These are finite test points, not a verified thickened compact region.

| Quantity | MSE checkpoint | MSE + residual checkpoint |
|---|---:|---:|
| Full-sample maximum error | 0.8881085953 | 0.9333088836 |
| Selected-sample maximum error | 0.0092413662 | 0.0135282936 |
| Selected-sample RMSE | 0.0022885278 | 0.0022056405 |
| Sampled chain residual, h=.05 | 0.0171202540 | 0.0169957139 |
| Sampled reached-boundary error | 0.0033180863 | 0.0028703799 |
| Largest sampled N delta + b | 0.1745206267 | 0.1728275185 |

The residual-trained model improves the broad selected RMSE slightly but
worsens the maximum error. The finite-chain envelope uses all chain nodes
and reached impacts. Its validity on these points is not a continuum
certificate, because uniform flow, event and roundoff errors are unavailable.

### Complete four-dimensional ReLU construction

`construction_experiment.py` fixes, before drawing queries,

- `B_c=[.15,.27] x [.05,.12] x [-1.4,-1.1] x [.27,.52]`;
- `D_c=[.15,.25] x [.08,.12] x [-1.4,-1.2] x [.30,.50]`;
- coordinate order `(s,Xi,Lambda,zeta_L)` and stopped duration `.02`.

Every grid node is retained. Analytically, all of `B_c` collides transversally
before `T*=.5*log(2.26/1.46)` with closing speed above .3, and the entire
one-step tube from `D_c` lies in `B_c`. An analytical Lipschitz constant is
`L=3.8284415309...`. The complete proof, actual ReLU multiplier, error budget
and architecture conventions are in `CONSTRUCTION_NOTES.md`.

`construct.TensorGrid.evaluate(..., k=k)` evaluates the actual approximate
ReLU products. Omitting `k` evaluates the exact multilinear interpolant;
these are distinct routines and both errors are recorded. The implementation
sums the at most 16 active hats, which is algebraically equivalent to the
full ReLU sum. It is not a materialized dense-network latency measurement.

| Maximum mesh | Complete nodes | ReLU gates (affine-skip DAG) | Maximum error on the same 10,000 D_c points | Exact-node analytical bound |
|---|---:|---:|---:|---:|
| .02 | 7,840 | 1,034,880 | 8.57452369e-4 | .07418379785 |
| .01 | 83,824 | 12,573,600 | 2.10575508e-5 | .03844442326 |
| .005 | 1,166,625 | 195,993,000 | 5.42929888e-6 | .01918220964 |

The last column assumes exact TTC nodal coefficients and real arithmetic.
A uniform nodal error eta adds `(1+48*4**(-(k+1)))*eta`; numerical nodal errors
were checked independently at sampled points, not uniformly certified.
Both unchanged learned models are also evaluated on precisely the same
`D_c` points. This is a common-domain comparison, not equal-budget training:
the learned models were trained over a much larger domain.

### Interpolated stopped-step sweeps

`iteration_experiment.py` uses the invariant computational coordinates
`(s,q,vF,omega)`, `q=Xi/(1+2s)`, `omega=zeta_L-integral_0^s a_L`, with contact
and the terminal plane absorbing. The two grids contain 188,139 and 1,395,009
nodes. No escaped state is projected into the domain; only roundoff-level
clipping after an explicit domain check is allowed and recorded.

The common independent selected test has 288 points. Under refinement,
final maximum error falls from .0487260 to .0258426 and RMSE from .0173454
to .0060121. The zero and .5 initializations reach the same discrete result
by 20 and 21 sweeps. Spatial interpolation error remains: the finite-sweep
exactness theorem concerns the continuum operator.

## Files and provenance

- `out/evaluation_results.json`, `out/test_holdout.npz`: fresh learned-model test.
- `out/baseline_reproduction.json`: agreement with the original rounded test.
- `out/holdout_D_chains.npz`: all cached points needed for the sampled chains.
- `out/construction_results.json` and `construction_*.npz`: complete grids,
  held-out queries, model comparisons, actual ReLU values and reference checks.
- `out/iteration_results.json`, `out/iteration_common_test.npz`: grid refinement.
- `out/reference_checks.json`, `out/reference_check_cases.npz`: independent
  solver comparisons and stopping checks.
- `out/fig_slice.pdf`, `out/fig_iter_cert.pdf`: regenerated manuscript figures.
- `out/reproduction_checks.json`: archive validation and recomputed table metrics.
- `provenance/checkpoints/`: unchanged learned checkpoints and supplied training metadata.
- `provenance/results.json`: the original rounded-time baseline used by the
  reproduction check.

The active source files in this release implement the experiments reported
in the accompanying manuscript.

## Training is optional and separate

The supplied checkpoints came from 100-epoch runs with 1,000,000 training
points, 100,000 validation points, initial times rounded to .001, and 44,959
training impact points appended with target zero. They have not been retrained
or overwritten. The full original platform/PyTorch versions were not supplied,
so bitwise reproduction of those weights is not claimed.

For future training, install an appropriate CPU PyTorch distribution separately.
The included training script was inspected and compiled, but was not executed
when producing this release. To regenerate the documented rounded-time **protocol**
using the corrected reference solver, write to separate directories:

```bash
python3 gen_data.py --legacy-rounded-time --out-dir regenerated_legacy
python3 train.py --data-dir regenerated_legacy --out-dir retrained_legacy
```

Omit `--legacy-rounded-time` for continuous initial times.
`train.py --terminal-boundary` adds terminal-boundary supervision; this is off
by default because it was absent from the supplied training protocol. Changing
labels or training options produces a new experiment, not the supplied model.

## Interpretation limits

The paper and code study a prescribed prediction flow. They do not simulate
or certify closed-loop collision avoidance. Exact-node analytical bounds,
finite sampled envelopes and observed errors are deliberately kept separate.
The experiments contain no dense-network real-time latency claim and do not
establish a universal refinement rate from the three construction meshes.

## Licenses

The original software, software documentation, supplied checkpoints, and
numerical arrays/JSON results are available under the MIT License. The
original PDF/PNG figures and `CONSTRUCTION_NOTES.md` are available under
CC BY-NC-ND 4.0. See `LICENSES.md` in the repository root, or the corresponding
file at the root of the standalone numerical archive, for exact scope and
attribution. That archive includes both complete license texts.
