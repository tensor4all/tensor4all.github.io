# Julia tutorials

This documentation provides a comprehensive tutorials/examples
on quantics and tensor cross interpolation (TCI) and their combinations (QTCI).
These technologies allow us to reveal low-rank tensor network representation (TNR) hidden in data or a function,
and perform computation such as Fourier transform and convolution.
Please refer [xfacpaper](https://arxiv.org/abs/2407.02454) for a more detailed introduction of these concepts.

The T4A group hosts various Julia libraries for performing such operations.
The folowing list is given in the order of low-level to high-level libraries:

- [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl/) provides implementations of TCI.
- [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl/) provides utilities for handling quantics representations, e.g., creating a quantics grid and transformation between the original coordinate system and the quantics representation.
- [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl/) is a thin wrapper around `TensorCrossInterpolation.jl` and `QuanticsGrids.jl`, providing valuable functionalities for non-expert users' performing quantics TCI (QTCI).
- [TCIITensorConversion.jl](https://github.com/tensor4all/TCIITensorConversion.jl/) provides conversions of tensor trains between `TensorCrossInterpolation.jl` and `ITensors.jl`.
- [Quantics.jl](https://github.com/tensor4all/Quantics.jl/) is an experimental library providing a high-level API for performing operations in QTT. This library is under development and its API may be subject to change. The library is not yet registered in the Julia package registry.

Additionally, we provide some topics on Julia packages such as:

- [Plots.jl](plots.ipynb). Basic tutorial for plotting using Plots.jl.

This documentation provides examples of using these libraries to perform QTCI and other operations.


## Notebooks

You can find our [Pluto notebooks](https://tensor4all.org/T4APlutoExamples/pluto_notebooks/)!
If you are interested in running the tutorials on your local machine, please follow the instructions below.

## Preparation

### Install Julia

Install `julia` command using [juliaup](https://github.com/JuliaLang/juliaup).

On Windows Julia and Juliaup can be installed directly from the Windows store. One can also install exactly the same version by executing


<!-- #region -->
```powershell
PS> winget install julia -s msstore
```

<!-- #endregion -->

on a command line.

Juliaup can be installed on Linux or Mac by executing


<!-- #region -->
```sh
$ curl -fsSL https://install.julialang.org | sh
```

<!-- #endregion -->

in a shell.

You can check that `julia` is installed correctly by simply running `julia` in your terminal:

```julia-repl
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.10.1 (2024-02-13)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```

The REPL greets you with a banner and a `julia>` prompt. Let's display "Hello World":

```julia-repl
julia> println("Hello World")
```

To see the environment in which Julia is running, you can use `versioninfo()`.

```julia-repl
julia> versioninfo()
```

To exit the interactive session, type `exit()` followed by the return or enter key:

```julia-repl
julia> exit()
```

See the official documentation at [The Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) to learn more.

### Install required packages

One can install required packages by running the following command on your shell:


<!-- #region -->
```sh
git clone git@github.com:tensor4all/T4AJuliaTutorials.git
cd T4AJuliaTutorials
julia --project scripts/setup.jl
```

<!-- #endregion -->

Here, the `--project` option activates our project, which is characterized by `Project.toml`, and `scripts/setup.jl` installs dependencies needed to run our notebooks.

<!-- #region -->
Keep in mind that [Quantics.jl](https://github.com/tensor4all/Quantics.jl) is not registered in [Julia's General registry](https://github.com/JuliaRegistries/General/tree/master). To run `qft.jl`, we need to add [T4ARegistry](https://github.com/tensor4all/T4ARegistry).

```sh
julia --project -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url="https://github.com/tensor4all/T4ARegistry.git"))'
```

<!-- #endregion -->

### Print out the status of the project

Having trouble? Try the following command in your Julia's REPL. On GitHub Actions instance we'll get:


```julia
using Dates;
now(UTC);
VERSION # display Julia version
using Pkg;
Pkg.status();
```

