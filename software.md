---
title: Software
---
# Software

The tensor4all software ecosystem is organized by status and workflow. For new
users today, we recommend `xfac` for C++ and Python workflows, and the stable
Julia packages for Julia workflows. Active development is focused on the
next-generation Rust stack: `tenferro-rs`, a tensor and autodiff engine that
is ready for production use with a relatively stable API, and `tensor4all-rs`
(TCI, quantics tensor trains, and tree tensor networks), which is under active
development on top of it.

## Software map

| Status | Library | Best for |
| --- | --- | --- |
| Use now | [`xfac`](https://github.com/tensor4all/xfac) / [Python tutorials](https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html) | C++ and Python workflows; original TCI implementation |
| Use now | [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl/) | Core TCI algorithms in Julia |
| Use now | [`QuanticsTCI.jl`](https://github.com/tensor4all/QuanticsTCI.jl/) | Convenient QTCI interface in Julia |
| Use now | [`QuanticsGrids.jl`](https://github.com/tensor4all/QuanticsGrids.jl/) | Quantics grids and coordinate transformations |
| Use now | [`InterpolativeQTT.jl`](https://github.com/tensor4all/InterpolativeQTT.jl/) | Multiscale interpolative QTT construction in Julia |
| Use now | [`tenferro-rs`](https://github.com/tensor4all/tenferro-rs) | Rust tensor and autodiff engine: einsum, linear algebra, FFT, CPU/CUDA backends |
| Active development | [`tensor4all-rs`](https://github.com/tensor4all/tensor4all-rs) | Next-generation Rust implementation of TCI, QTT, and tree tensor networks |
| Active development | [`Tensor4all.jl`](https://github.com/tensor4all/Tensor4all.jl) | Julia frontend for `tensor4all-rs` |
| Maintenance | [`Quantics.jl`](https://github.com/tensor4all/Quantics.jl/) | Existing QTT workflows built on ITensors.jl |
| Maintenance | [`FastMPOContractions.jl`](https://github.com/tensor4all/FastMPOContractions.jl/) | Existing MPO contraction workflows |
| Maintenance | [`TCIITensorConversion.jl`](https://github.com/tensor4all/TCIITensorConversion.jl/) | Historical ITensors conversion package |

## Use now: C++ and Python

[`xfac`](https://github.com/tensor4all/xfac) is the original C++
implementation of tensor cross interpolation in the tensor4all ecosystem. It
includes Python bindings and remains the recommended current path for C++ and
Python users.

- Documentation: [xfac readthedocs](https://xfac.readthedocs.io/en/latest/intro.html)
- Python tutorials: [tutorial-python](https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html)
- Source: [tensor4all/xfac](https://github.com/tensor4all/xfac)

## Use now: Julia

These Julia packages are the current recommended route for Julia users who
need stable workflows.

- [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl/) provides the core TCI algorithms.
- [`QuanticsTCI.jl`](https://github.com/tensor4all/QuanticsTCI.jl/) provides a convenient interface for quantics TCI, built on `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`.
- [`QuanticsGrids.jl`](https://github.com/tensor4all/QuanticsGrids.jl/) provides quantics grids and coordinate transformations.
- [`InterpolativeQTT.jl`](https://github.com/tensor4all/InterpolativeQTT.jl/) implements multiscale interpolative construction of quantized tensor trains.

For Julia examples, see the [Julia tutorials](juliatutorials/index.html).

## The Rust stack: tenferro-rs and tensor4all-rs

The next generation of the tensor4all ecosystem is written in Rust, in two
layers.

### tenferro-rs: tensor and autodiff engine (use now)

[`tenferro-rs`](https://github.com/tensor4all/tenferro-rs) is a Rust-native
tensor computation stack with opt-in automatic differentiation for scientific
workloads. It provides typed and dynamic dense tensors, explicit CPU/CUDA
backend dispatch, linear algebra, NumPy-style einsum, FFT, PyTorch-style eager
autodiff, and JAX-style traced transforms. It sits between low-level array
crates and full deep-learning frameworks, targeting scientific code that needs
column-major storage, dynamic shapes, and extensible operations.

tenferro-rs is ready for production use, with a relatively stable API.

- Crates are published on [crates.io](https://crates.io/crates/tenferro-runtime); start with `tenferro-runtime`, `tenferro-cpu`, and the operation crates you need.
- Documentation: [tensor4all.org/tenferro-rs](https://tensor4all.org/tenferro-rs/)
- Performance is tracked openly against PyTorch and JAX in [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark).
- Background and design rationale: [Introducing tenferro-rs](https://tensor4all.org/blog/introducing-tenferro-rs/) (also in [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) and [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)).

### tensor4all-rs: tensor networks (active development)

[`tensor4all-rs`](https://github.com/tensor4all/tensor4all-rs) is the
next-generation implementation of the tensor4all algorithms, built on
`tenferro-rs`. It covers tensor cross interpolation (TCI), quantics tensor
trains, tree tensor networks with arbitrary topology, and an ITensors.jl-like
dynamic tensor API, plus the C API used by language bindings.

- Documentation: [tensor4all.org/tensor4all-rs](https://tensor4all.org/tensor4all-rs/)
- The crates are not yet published on crates.io; use git dependencies as described in the repository README.
- [`Tensor4all.jl`](https://github.com/tensor4all/Tensor4all.jl) is the Julia frontend for this stack, aimed at light use and teaching. Performance-critical applications should use `tensor4all-rs` directly.

New features and performance work land in this stack first. tensor4all-rs is
ready for early adopters comfortable with Rust; users who need a stable,
documented workflow today can start with `xfac` or the stable Julia packages
above and migrate later.

## Maintenance and legacy packages

These packages are useful for existing workflows, but they are not the primary
recommendation for new users.

- [`Quantics.jl`](https://github.com/tensor4all/Quantics.jl/) provides a high-level QTT interface built on ITensors.jl.
- [`FastMPOContractions.jl`](https://github.com/tensor4all/FastMPOContractions.jl/) supports existing MPO contraction workflows.
- [`TCIITensorConversion.jl`](https://github.com/tensor4all/TCIITensorConversion.jl/) is historical. Its conversion functionality has been absorbed into `TensorCrossInterpolation.jl` as an extension.

## Experimental repositories

Some old `T4A****.jl` repositories were used for experiments. They are not
recommended for new users and are not planned as maintained public libraries.
