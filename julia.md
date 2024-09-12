# Julia libraries

## For users
Our Julia ecosystem consists of the following packages:

* [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl) provides implementations of TCI.
* [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl) provides utilities for handing quantics representations, e.g., creating a quantics grid and transformation between the original coordinate system and the quantics representation.
* [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl) is a thin wrapper around `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`, providing valuable functionalities for non-expert users' performing quantics TCI (QTCI).

### Tutorials
Online documentation for each package is available at the links above.
But, we recommend starting with [our Julia tutorials](/juliatutorials/index.html).

## For developers & internal users
Some public packages are not registered in the Julia General registry.
To use them, you can add them via [the T4A registry](https://github.com/tensor4all/T4ARegistry).
