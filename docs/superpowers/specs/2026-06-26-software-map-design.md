# Software Map Reorganization Design

## Goal

Reorganize the software section of tensor4all.org around package status and
user workflow instead of the old "Julia vs C++/Python" split.

The new public message should make three points clear:

- `xfac` with Python bindings is a current recommended route for C++ and Python
  users, and it remains the original implementation.
- The stable Julia packages remain the recommended route for Julia users today.
- `tensor4all-rs` and `Tensor4all.jl` are the actively developed future stack,
  with potential to unify or replace parts of the current Julia ecosystem, but
  they should not be presented as the default stable path yet.

## Page Structure

Create a single `software.md` page and link to it from the home page `Code`
section. The page should start with a compact software map, then expand each
category.

### Use Now

This section should appear first.

- `xfac`: original C++ implementation with Python bindings. Recommended for
  current C++ and Python workflows.
- `TensorCrossInterpolation.jl`: stable Julia TCI core package.
- `QuanticsTCI.jl`: stable Julia QTCI convenience layer over
  `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`.
- `QuanticsGrids.jl`: stable Julia quantics-grid and coordinate-conversion
  utilities.
- `InterpolativeQTT.jl`: current Julia implementation of multiscale
  interpolative construction of quantized tensor trains.

### Active Development / Future Stack

- `tensor4all-rs`: Rust implementation covering TCI, QTT, TreeTN, and C API
  work. Present as under active development.
- `Tensor4all.jl`: Julia frontend for `tensor4all-rs`. Present as an active
  development frontend, not the default stable Julia recommendation.

Mention that this stack is likely to replace or unify parts of the existing
Julia ecosystem over time.

### Maintenance / Legacy

- `Quantics.jl`
- `FastMPOContractions.jl`
- `TCIITensorConversion.jl`

Mention that `TCIITensorConversion.jl` functionality has been absorbed into
`TensorCrossInterpolation.jl` as an extension.

### Experimental Libraries

Do not list `T4A****.jl` experimental libraries in the map or as user-facing
software choices. Add only a short note that old experimental repositories are
not recommended for new users and are not planned as maintained public
libraries.

## Existing Pages To Update

- `index.md`: Replace the old two-library wording with a link to the new
  software map.
- `julia.md`: Either keep as a Julia-specific detail page or redirect its role
  by pointing users to `software.md` first.
- `faq.md`: Replace the old "C++/Python vs Julia" getting-started wording with
  a link to the software map, while preserving specific FAQ answers where they
  still mention language-specific APIs.

## Constraints

- Keep the site simple: Jekyll Markdown, no new build tooling.
- Avoid overstating stability of `tensor4all-rs` and `Tensor4all.jl`.
- Make `xfac/Python` visibly current, not merely historical.
- Use English on the website.
