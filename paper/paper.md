---
title: 'StrainTrace: Virtual extensometry and fixed-reference digital image correlation'
tags:
  - Python
  - digital image correlation
  - experimental mechanics
  - strain measurement
authors:
  - name: Delun Gong
    orcid: 0000-0001-7877-7707
    email: dlgong17s@imr.ac.cn
    corresponding: true
    affiliation: 1
affiliations:
  - name: Institute of Metal Research, Chinese Academy of Sciences, Shenyang 110016, China
    index: 1
date: 27 September 2026
bibliography: paper.bib
---

# Summary

Image sequences recorded during mechanical tests can provide both gauge-length
changes and spatially resolved deformation. These measurements require different
analysis choices: a virtual extensometer follows selected image regions, whereas
digital image correlation estimates a displacement field and differentiates that
field to obtain strain. StrainTrace is a Python desktop application and headless
analysis tool that exposes both workflows while retaining their distinct output
semantics. It provides a two-region virtual extensometer and a fixed-reference,
local-subset, in-plane two-dimensional digital image correlation workflow. Its
outputs include numerical tables, diagnostic fields, configuration records and
verifiable file manifests. StrainTrace retains the historical ezDIC module and
executable names for compatibility. The current repository source is MIT licensed;
older ezDIC archives remain separate historical records.

# Statement of need

Materials researchers often need an image-derived strain history to accompany
a mechanical test, and may subsequently need to determine whether deformation
varies across the observed surface. Moving between gauge measurements, correlation
software and plotting tools can obscure which reference image, region, validity
criterion and strain convention generated a reported value. A failed correlation
must also remain distinguishable from a measured small displacement. StrainTrace
addresses this recording and inspection problem through a common local workbench
and a versioned configuration interface.

The virtual extensometer reports engineering and true strain from selected region
pairs, with diagnostic information for the tracking step. The full-field workflow
correlates each deformation image to the selected fixed reference, evaluates a
grid of points within a rectangular region, and exports displacement and strain
maps. Its measurements are sampled on that grid, not at every image pixel. The
software is intended for in-plane image sequences with suitable texture; it does
not implement stereo reconstruction, volumetric correlation, or correction for
out-of-plane motion. These scope limits matter when an image-derived strain is
used to interpret a mechanical experiment.

# State of the field

Digital image correlation is an established measurement approach. Ncorr provides
an open-source two-dimensional implementation in MATLAB [@blaber2015ncorr]. DICe
provides full-field displacement and strain analysis, object tracking, and support
for stereo applications and parallel execution [@dice]. These projects already
address extensive correlation needs. StrainTrace does not claim a new correlation
theory or general accuracy or speed superiority over them.

The contribution is a focused Python workflow that combines a virtual
extensometer, local full-field analysis and explicit recording of the decisions
needed to interpret their outputs. The desktop and headless interfaces share the
numerical core. Configuration normalization, separate displacement and strain
validity, failed-frame handling and output verification are part of the application
contract. This choice supports scripted laboratory workflows without requiring
MATLAB, while preserving a desktop interface for selecting image regions. It
trades the scope of a general stereo or high-performance correlation framework
for a narrower implementation that can be exercised through generated regression
cases. Researchers requiring the broader capabilities should use the appropriate
established package rather than interpret this scope as a substitute.

# Software design

StrainTrace separates its numerical and export functions from the Tk desktop
interface. The command-line interface validates a versioned JSON configuration,
normalizes defaults and rejects unknown fields or non-finite values. The same
core can therefore be invoked without importing the GUI. Existing ezDIC imports
remain available, while the installable package exposes StrainTrace command-line
and desktop entry points. Origin export is optional; numerical tables and normal
analysis do not require Origin.

For full-field correlation, every selected deformation frame is compared with one
fixed reference. Reference-based normalization and bounded multiscale
initialization precede the implemented inverse-compositional Gauss--Newton or
Levenberg--Marquardt solution. Solver diagnostics accompany accepted and rejected
points. A displacement-valid point need not have a valid local strain estimate,
so the outputs retain separate validity fields. Coordinates and displacements are
in pixels; strain is dimensionless. Green--Lagrange and infinitesimal strain
components are identified separately, and off-diagonal components use tensor
shear rather than engineering shear. Unsuccessful measurements remain missing
values, rather than being interpolated into apparently complete fields.

Full-field outputs are prepared in a staging directory. Completed results are
published with recorded input identities, configuration and source fingerprints;
earlier successful outputs are retained under a previous-run directory. A frame
with no valid strain field is recorded as failed and skipped, while other usable
frames can still produce results. A run without any valid strain field fails.
These distinctions prevent a directory containing some output files from being
interpreted automatically as a successful scientific analysis.

The automated tests exercise the configuration interface, numerical and export
behavior, failure conditions and preservation of previous results. A locked
synthetic benchmark supplies small and large translations, a prescribed affine
strain, and a near-one-dimensional periodic texture case. Its generated inputs,
fixed conditions, source hashes and numerical diagnostics make the verification
repeatable from an installed distribution. Translation and strain cases test
specified numerical behavior; the ambiguous texture case tests failure reporting.
The benchmark also distinguishes quality ranking from calibration of a quality
threshold. Passing it does not establish uncertainty for an experimental camera,
specimen or texture, and the paper makes no such claim.

# Research impact statement

On 27 September 2026, the author confirmed using StrainTrace in the research
underlying the Ti-24Nb-4Zr-8Sn study reported by Gong et al. [@gong2026acta]. This
is an author-confirmed research application, not a claim that the article cites
StrainTrace or that the current full-field implementation was used in that study.
The historical source revision and the specific operations and outputs must be
mapped to the author's processing records before submission. The repository's
research-use record separates that outstanding mapping from the present synthetic
verification suite. No independent-user adoption or experimental accuracy is
inferred from the repository's availability or benchmark results.

# AI usage disclosure

OpenAI Codex (GPT-6) assisted the September 2026 preparation with repository
inspection, Python packaging, tests, documentation, bibliography and manuscript
drafting, and automated verification. The complete history of earlier AI
assistance has not been confirmed for this software. The sole author must review,
edit and validate the assisted outputs and confirm the complete disclosure and
human responsibility for scientific and design decisions before submission.

# Acknowledgements

No external funding was received for this software. There was no sponsor
involvement. The author declares no competing interests.

# References
