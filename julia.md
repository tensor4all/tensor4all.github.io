# Julia libraries

For the full cross-language overview, see the [software map](software.html).
This page focuses on Julia packages.

## Overview of libraries
The tensor4all group hosts several Julia libraries for tensor cross
interpolation and quantics tensor trains.

Current recommended packages:

- [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl/) provides implementations of TCI.
- [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl/) provides utilities for handling quantics representations, e.g., creating a quantics grid and transformation between the original coordinate system and the quantics representation.
- [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl/) is a thin wrapper around `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`, providing valuable functionalities for non-expert users' performing quantics TCI (QTCI).
- [InterpolativeQTT.jl](https://github.com/tensor4all/InterpolativeQTT.jl/) implements multiscale interpolative construction of quantized tensor trains.

Maintenance-mode packages:

- [Quantics.jl](https://github.com/tensor4all/Quantics.jl/) provides a high-level API for QTT operations built on `ITensors.jl`.
- [FastMPOContractions.jl](https://github.com/tensor4all/FastMPOContractions.jl/) supports existing MPO contraction workflows.
- [TCIITensorConversion.jl](https://github.com/tensor4all/TCIITensorConversion.jl/) is historical. Its conversion functionality has been absorbed into `TensorCrossInterpolation.jl` as an extension.

## Tutorials
Are you interested in learning how to use these libraries?
If so, please visit our [Julia tutorials](/juliatutorials/index.html) page.
