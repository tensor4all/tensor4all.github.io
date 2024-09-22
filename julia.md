# Julia libraries

## Overview of libraries
The T4A group hosts various Julia libraries for performing such operations.
The folowing list is given in the order of low-level to high-level libraries:

- [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl/) provides implementations of TCI.
- [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl/) provides utilities for handling quantics representations, e.g., creating a quantics grid and transformation between the original coordinate system and the quantics representation.
- [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl/) is a thin wrapper around `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`, providing valuable functionalities for non-expert users' performing quantics TCI (QTCI).
- [TCIITensorConversion.jl](https://github.com/tensor4all/TCIITensorConversion.jl/) provides conversions of tensor trains between `TensorCrossInterpolation.jl` and `ITensors.jl`.
- [Quantics.jl](https://github.com/tensor4all/Quantics.jl/) is an experimental library providing a high-level API for performing operations in QTT. This library is under development and its API may be subject to change. 

## Tutorials
Are you interested in learning how to use these libraries?
If so, please visit our [Julia tutorials](/juliatutorials/index.html) page.