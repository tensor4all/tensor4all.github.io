---
title: Software
---
# Software

The tensor4all software ecosystem is organized by status and workflow. For new
users today, we recommend `xfac` for C++ and Python workflows, and the stable
Julia packages for Julia workflows.

## Software map

| Status | Library | Best for |
| --- | --- | --- |
| Use now | [`xfac`](https://github.com/tensor4all/xfac) / [Python tutorials](https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html) | C++ and Python workflows; original TCI implementation |
| Use now | [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl/) | Core TCI algorithms in Julia |
| Use now | [`QuanticsTCI.jl`](https://github.com/tensor4all/QuanticsTCI.jl/) | Convenient QTCI interface in Julia |
| Use now | [`QuanticsGrids.jl`](https://github.com/tensor4all/QuanticsGrids.jl/) | Quantics grids and coordinate transformations |
| Use now | [`InterpolativeQTT.jl`](https://github.com/tensor4all/InterpolativeQTT.jl/) | Multiscale interpolative QTT construction in Julia |
| Active development | [`tensor4all-rs`](https://github.com/tensor4all/tensor4all-rs) | Next-generation Rust implementation |
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

These Julia packages are the current recommended route for Julia users.

- [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl/) provides the core TCI algorithms.
- [`QuanticsTCI.jl`](https://github.com/tensor4all/QuanticsTCI.jl/) provides a convenient interface for quantics TCI, built on `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`.
- [`QuanticsGrids.jl`](https://github.com/tensor4all/QuanticsGrids.jl/) provides quantics grids and coordinate transformations.
- [`InterpolativeQTT.jl`](https://github.com/tensor4all/InterpolativeQTT.jl/) implements multiscale interpolative construction of quantized tensor trains.

For Julia examples, see the [Julia tutorials](juliatutorials/index.html).

## Active development: Rust and future Julia frontend

[`tensor4all-rs`](https://github.com/tensor4all/tensor4all-rs) is the
next-generation Rust implementation for TCI, quantics tensor trains, tree tensor
networks, and language-binding infrastructure.
[`Tensor4all.jl`](https://github.com/tensor4all/Tensor4all.jl) is the Julia
frontend for this Rust stack.

This stack is under active development. It may eventually unify or replace
parts of the current Julia ecosystem, but new users who need a stable workflow
today should start with `xfac` or the stable Julia packages above.

## Maintenance and legacy packages

These packages are useful for existing workflows, but they are not the primary
recommendation for new users.

- [`Quantics.jl`](https://github.com/tensor4all/Quantics.jl/) provides a high-level QTT interface built on ITensors.jl.
- [`FastMPOContractions.jl`](https://github.com/tensor4all/FastMPOContractions.jl/) supports existing MPO contraction workflows.
- [`TCIITensorConversion.jl`](https://github.com/tensor4all/TCIITensorConversion.jl/) is historical. Its conversion functionality has been absorbed into `TensorCrossInterpolation.jl` as an extension.

## Experimental repositories

Some old `T4A****.jl` repositories were used for experiments. They are not
recommended for new users and are not planned as maintained public libraries.
