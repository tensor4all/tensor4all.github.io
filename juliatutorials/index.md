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

- [Plots.jl](https://tensor4all.org/T4APlutoExamples/pluto_notebooks/plots.html). Basic tutorial for plotting using Plots.jl.

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
$ curl -fsSL https://install.julialang.org | sh -s -- --yes
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
  | | |_| | | | (_| |  |  Version 1.10.5 (2024-08-27)
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

### How to open Pluto notebooks locally

Our tutorials are written in [Pluto.jl](https://plutojl.org/) notebook. To open Pluto notebooks locally, clone our [tutorial repository T4APlutoExamples](https://github.com/tensor4all/T4APlutoExamples) and navigate to directory `T4APlutoExamples`:

```sh
$ git clone https://github.com/tensor4all/T4APlutoExamples.git
$ cd T4APlutoExamples
$ Manifest.toml   Project.toml    README.md       pluto_notebooks scripts
```

Then we also need to install Pluto.jl. To do this, type `julia --project` in a terminal to open Julia REPL and run the following julia code:

```julia-repl
$ julia --project
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.10.5 (2024-08-27)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> using Pkg; Pkg.instantiate()
  Activating project at `~/tensor4all/T4APlutoExamples`
```

Continued from previous REPL session, launch Pluto server locally via:

```julia-repl
julia> using Pluto; Pluto.run()
[ Info: Loading...
┌ Info:
└ Opening http://localhost:1234/?secret=xxxxxx in your default browser... ~ have fun!
┌ Info:
│ Press Ctrl+C in this terminal to stop Pluto
└
```

This will open http://localhost:1234/?secret=xxxxxx in your default browser and you will see the following main menu of Pluto notebook:

![](https://raw.githubusercontent.com/tensor4all/T4APlutoExamples/refs/heads/main/assets/open_welcome.png)

To open notebooks under `pluto_notebooks` directory, type `pluto_notebooks/welcome.jl` to text area under `Open a notebook` section. Then, press `Open` button.

## Software guide(T4AJuliaTutorials wiki)

[T4AJuliaTutorials/wiki](https://github.com/tensor4all/T4AJuliaTutorials/wiki) provides basic tutorials about software development. For example How to set up a development environment on [Windows](https://github.com/tensor4all/T4AJuliaTutorials/wiki/Instructions-for-setting-up-a-development-environment-on-Windows(PowerShell,-winget)), [macOS](https://github.com/tensor4all/T4AJuliaTutorials/wiki/Instructions-for-setting-up-a-development-environment-on-macOS) and [Linux and WSL2](https://github.com/tensor4all/T4AJuliaTutorials/wiki/Instructions-for-setting-up-a-development-environment-on-Linux(Ubuntu,-WSL2)).