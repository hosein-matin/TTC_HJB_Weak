# Dynamic time-to-collision and the stopped-step Bellman equation

Companion code, numerical artifacts, and author preprint for:

**A Derivative-Free Bellman Equation for Dynamic Time-to-Collision:
Two-Sided Certificates and Explicit ReLU Approximation**  
**Hossein Nick Zinat Matin and Yuneil Yeo**

[Read the manuscript](manuscript/TTC_Bellman_Preprint.pdf) ·
[Numerical documentation](ttc_numerics/README.md) ·
[Construction supplement](ttc_numerics/CONSTRUCTION_NOTES.md) ·
[License scope](LICENSES.md)

**Manuscript status:** author preprint. This release does not identify the
paper as accepted or published. Release package version: **1.0.0**.

## Research overview

The paper studies the capped time-to-collision (TTC) of a prescribed
Caratheodory prediction flow. Its stopped-step Bellman equation advances the
flow for a fixed duration or until contact and adds the elapsed time.

The theoretical results establish local Lipschitz regularity on the
transversal collision domain, uniqueness for the stopped-step equation,
two-sided error bounds from complete-chain residuals and boundary errors,
and finite-sweep recovery for the exact continuum operator. They also
characterize the nonuniqueness of an optional-duration infimum operator and
give a local, optimization-free ReLU approximation with nodal-error bounds.

The numerical examples use the optimal velocity model. They evaluate two
supplied learned checkpoints, implement complete four-dimensional ReLU
constructions, and study interpolated stopped-step sweeps under grid
refinement.

## Quick start: inspect the supplied results

The complete numerical outputs and trained checkpoints are included. Python
3.12 and the pinned versions below were used for the reported evaluation.
PyTorch is **not required** to evaluate the supplied checkpoints.

From the repository root:

```bash
python3 -m pip install -r ttc_numerics/requirements.txt
cd ttc_numerics
python3 verify_results.py
```

The verifier checks the saved NumPy archives and recomputes learned-model
and construction error metrics. It writes `out/reproduction_checks.json`.
This verifies saved artifacts; it does not rerun the reference solver or
independently establish every assumption in the mathematical results.

## Reproduce the numerical experiments

From `ttc_numerics/`:

```bash
python3 experiments.py
```

On systems with a POSIX shell, `sh run_all.sh` runs the same pipeline.
To reuse the supplied complete construction grids while recomputing the
remaining experiments:

```bash
python3 experiments.py --reuse-grids
```

The pipeline runs the reference-solver checks, checkpoint evaluation, ReLU
construction, grid-sweep experiment, figure generation, and saved-result
verification. Results are written to `ttc_numerics/out/`. See the
[numerical README](ttc_numerics/README.md) for parameters, seeds, interpretation,
and optional training instructions.

The default workflow **evaluates the supplied trained weights**. Retraining
is separate, requires PyTorch, and is not claimed to reproduce the original
weights bit for bit. Original training-platform details are unavailable.

## Repository contents

| Path | Contents |
|---|---|
| `manuscript/` | Eight-page author preprint, buildable LaTeX, bibliography, and figures |
| `ttc_numerics/` | Numerical source, requirements, and experiment documentation |
| `ttc_numerics/out/` | Complete saved arrays, JSON metrics, environment record, and figures |
| `ttc_numerics/provenance/checkpoints/` | Two unchanged learned checkpoints and supplied metadata |
| `ttc_numerics/provenance/results.json` | Original rounded-time baseline used by the reproduction check |
| `ttc_numerics/CONSTRUCTION_NOTES.md` | Detailed analytical construction and architecture conventions |
| `tools/set_repository_url.py` | Inserts the actual GitHub URL into the manuscript configuration and citation metadata |
| `LICENSE`, `LICENSES.md`, `LICENSES/` | MIT text, explicit scope, CC BY 4.0, and third-party licenses |
| `CITATION.cff` | Machine-readable citation metadata |
| `SHA256SUMS.txt` | File-integrity manifest for this release snapshot |

A standalone `TTC_Numerical_Methods.zip` contains the complete numerical
directory and its licenses. Extract it before running the commands in its
README. The repository source tree and this archive contain the same
numerical implementation, checkpoints, and result data.

## Build the manuscript and set the repository URL

A TeX distribution with `latexmk` and BibTeX is required. From `manuscript/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error TTC_Bellman_Preprint.tex
```

The opening paragraph of the numerical section uses
`manuscript/repository_url.tex`. Until a real URL is supplied, the compiled
paper refers to the accompanying reproducibility archive. After creating
the repository, run this from the repository root, replacing the example
owner and repository with the actual values:

```bash
python3 tools/set_repository_url.py https://github.com/OWNER/REPOSITORY --build
```

This updates the URL, citation metadata, compiled PDF, and release checksums.
Without `--build`, the script updates the source and metadata only; compile
the manuscript separately before distributing a new PDF. In Overleaf, edit
the value in `manuscript/repository_url.tex` and recompile.

## Scope of the numerical evidence

The exact-node approximation bound is analytical. Computed ODE labels,
finite test errors, and sampled chain envelopes are numerical evidence;
they are not uniform validated certificates. Spatial interpolation error
remains in the discrete sweep experiment. The construction evaluator uses
active hats and is not a dense-network latency benchmark.

The paper studies a prescribed prediction flow. It does not establish
closed-loop collision avoidance or a real-time control implementation.
These distinctions are also stated in the manuscript and numerical README.

## Citation

Please cite the accompanying manuscript when referring to its results:

```bibtex
@unpublished{zinatmatin_yeo_ttc_bellman_2026,
  author = {Zinat Matin, Hossein Nick and Yeo, Yuneil},
  title = {A Derivative-Free Bellman Equation for Dynamic
           Time-to-Collision: Two-Sided Certificates and Explicit
           {ReLU} Approximation},
  year = {2026},
  note = {Author preprint}
}
```

Update this citation when a publication DOI or an archived release DOI
becomes available. No DOI has been assigned in this package.

## Licenses

- **MIT:** original software, software documentation, supplied model weights,
  and numerical data.
- **CC BY 4.0:** the author preprint, original figures, and mathematical
  construction supplement.
- **Existing third-party licenses:** the unchanged LaTeX class and BibTeX style.

See [LICENSES.md](LICENSES.md) for the exact file coverage and attribution.
