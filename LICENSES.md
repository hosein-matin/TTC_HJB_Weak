# License scope

Copyright (c) 2026 Hossein Nick Zinat Matin for the author's
original material in this repository. This repository contains separately
licensed software and manuscript material. No single license applies to every
file.

## MIT: software, software documentation and numerical artifacts

The [MIT License](LICENSE) applies to the author's original:

- Python and shell programs in `ttc_numerics/` and `tools/`;
- dependency specifications, repository configuration, `CITATION.cff`, and
  software documentation, including both `README.md` files;
- supplied model weights and their metadata in `ttc_numerics/provenance/checkpoints/`;
- numerical arrays and JSON results in `ttc_numerics/out/`, and the baseline
  summary `ttc_numerics/provenance/results.json`.

The author makes the listed weights and numerical data available under the
same MIT terms as the software. Keep the copyright and permission notices
when redistributing these materials. A scholarly citation is appreciated;
it is not an additional condition of the MIT license.

## CC BY-NC-ND 4.0: author preprint, original figures and mathematical supplement

The following original material is licensed under **Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**:

- `manuscript/TTC_Bellman_Preprint.pdf` and `manuscript/TTC_Bellman_Preprint.tex`;
- the author's original content in `manuscript/references.bib` and the
  manuscript configuration `manuscript/repository_url.tex`;
- the author's original figures in `manuscript/Figures/` and their corresponding
  PDF/PNG versions in `ttc_numerics/out/`;
- `ttc_numerics/CONSTRUCTION_NOTES.md`, the mathematical supplement.

Attribution: Hossein Nick Zinat Matin,
*A Derivative-Free Bellman Equation for Dynamic Time-to-Collision: Two-Sided
Certificates and Explicit ReLU Approximation*, author preprint, 2026.

Reuse must comply with the full license, including appropriate attribution,
a license reference, and an indication of changes where applicable. The
[complete legal text](LICENSES/CC-BY-NC-ND-4.0.txt) is included; the
[official license page](https://creativecommons.org/licenses/by-nc-nd/4.0/) provides
the summary. The grant covers the author's material in this preprint release.
It does not grant rights in a future publisher's version of record, referenced
publications, third-party material, names, trademarks, or rights the author
does not hold.

## Third-party LaTeX files

The following files retain their existing licenses and copyright notices;
they are excluded from the author's MIT and CC BY-NC-ND grants:

| File | Existing license | Included license text |
|---|---|---|
| `manuscript/ieeeconf.cls` | Perl Artistic License, as stated in the file header | [Artistic](LICENSES/Artistic-1.0.txt) |
| `manuscript/IEEEtran.bst` | LaTeX Project Public License version 1.3, as stated in the file header | [LPPL 1.3](LICENSES/LPPL-1.3.txt) |

Both files are redistributed unchanged from the supplied manuscript package.
Their original contribution notices and credits remain in the files.

External software installed from `requirements.txt`, optional PyTorch, Python,
and the TeX distribution retain their own licenses. They are not bundled or
relicensed by this repository.
