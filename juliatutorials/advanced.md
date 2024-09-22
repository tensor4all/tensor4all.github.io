# Advanced topics

## How to open Pluto notebooks locally

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

## Developer guide (T4AJuliaTutorials wiki)

[T4AJuliaTutorials/wiki](https://github.com/tensor4all/T4AJuliaTutorials/wiki) provides basic tutorials about software development. For example How to set up a development environment on [Windows](https://github.com/tensor4all/T4AJuliaTutorials/wiki/Instructions-for-setting-up-a-development-environment-on-Windows(PowerShell,-winget)), [macOS](https://github.com/tensor4all/T4AJuliaTutorials/wiki/Instructions-for-setting-up-a-development-environment-on-macOS) and [Linux and WSL2](https://github.com/tensor4all/T4AJuliaTutorials/wiki/Instructions-for-setting-up-a-development-environment-on-Linux(Ubuntu,-WSL2)).