# Julia tutorials

## Steps to view/run the tutorials

You can view [our notebook page](https://tensor4all.org/T4APlutoExamples/pluto_notebooks/) by a web browser.

If you want to run the notebooks interactively,
please install `Julia` following the [instructions](#install-julia).
At the top of each notebook,
you can find instructions on how to run a notebook on a local web browser.
We recommend *not to use Safari browser* because the interactive features may not work properly.

Only if you are interested in advanced topics or developers, you can visit [our advanced topics page](advanced.html).

## Install Julia

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