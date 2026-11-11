---
title: Prefer runtime loading over link-time linking
---

# Don't link MPI, HDF5, BLAS or CUDA at build time

*Our native libraries load MPI, HDF5, CUDA and BLAS when the program runs instead of linking them when it is built. This post explains why, describes the two mechanisms we use (`dlopen` and function-pointer injection), and says when to use which.*

by **Hiroshi Shinaoka** (Saitama University), for the tensor4all team

*🌐 [English](https://tensor4all.org/blog/prefer-runtime-loading/) · [日本語](https://tensor4all.org/blog/prefer-runtime-loading-ja/)*

---

## What goes wrong with link-time linking

The usual way to use a C or Fortran library from compiled code is to link it
at build time. That works if you build on the machine where the code runs. It
breaks down once you want to ship a binary, or once your library is called from
Julia or Python.

A binary linked against one MPI implementation, one HDF5 build or one CUDA
version runs only where that same library is installed, and the number of
combinations you would have to ship grows quickly. Building also needs the
toolchain and the headers. Linking MPI from Rust through bindgen needs a C
compiler, the MPI headers and libclang, which is a lot to ask of someone who
only wants to install a package.

Inside Julia or Python there is a second problem. Julia always has a BLAS
loaded, and an HDF5 too once HDF5.jl is in use; a Python session may already
have NumPy, h5py and mpi4py. If our library brings its own copies, the process
ends up with two HDF5 states, or two BLAS implementations with two thread
pools. The host language also knows things a library linked at build time
cannot ask about: which MPI is running, which BLAS is active, and whether its
integers are 32 or 64 bits wide.

So we decide at run time. The program is told which library to use when it
starts. Binaries become easier to distribute, and Julia and Python wrappers
become easier to write, because the host can hand over the MPI runtime, the
BLAS and the HDF5 library it already has.

We use two mechanisms for this, and each fits a different kind of library.

## Mechanism 1: open the library with `dlopen`

The library opens the shared object at run time and looks up the symbols it
needs. We do this for HDF5, MPI and CUDA.

[hdf5-rt](https://github.com/tensor4all/hdf5-rt) is a fork of hdf5-metno
that loads HDF5 with `dlopen`. `init` takes an optional path, so a host can
point it at the HDF5 it already uses (in Julia, for example,
`HDF5_jll.libhdf5_path`). Without a path it tries a list of usual locations for
the platform. After loading, it checks the HDF5 version and refuses versions
that are too old. The library handle is leaked on purpose and never
`dlclose`d: HDF5's cleanup code can loop forever or crash if the library is
unloaded while its internal state is still alive.

[rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) is a fork of rsmpi with an
extra backend, `mpi-rt-sys-backend`, built on the MPIABI of
[MPItrampoline](https://github.com/eschnett/MPItrampoline). MPIABI fixes the
parts of the ABI that differ between MPI implementations: handle types are
integers (`usize` on the Rust side), `MPI_Status` has a fixed layout, and
constants are exported as symbols instead of macros. At run time the backend
loads an [MPIwrapper](https://github.com/eschnett/MPIwrapper) library whose
path is given in `MPI_RT_LIB`, and MPIwrapper forwards to the native MPI. The
Rust build then needs no C compiler, no MPI headers and no libclang; the cost
moves to MPIwrapper, which has to be built once against the system MPI. We want
this so that Rust MPI code can be called from Julia and Python and work with
[MPI.jl](https://github.com/JuliaParallel/MPI.jl) and
[mpi4py](https://mpi4py.readthedocs.io/). The current interoperability test
launches Rust, Python and Julia processes under one `mpiexec` and has them
share `MPI_COMM_WORLD`. The backend is still experimental, and we hope to
contribute it back to rsmpi.

For CUDA, tenferro-rs uses `cudarc` with its `dynamic-loading` feature, so the
CUDA driver, runtime and NVRTC libraries are found when the program runs. The
GPU kernels come as source from
[t4a-cuda-kernels](https://github.com/tensor4all/t4a-cuda-kernels) and are
compiled with NVRTC at run time, so the binary does not link CUDA.

`dlopen` works well when there is one library, the host can tell you where it
is (or it sits somewhere predictable), and its symbols are stable enough to bind
by name.

## Mechanism 2: the host passes function pointers

For BLAS and LAPACK called from Julia or Python, injection is the only stable
method we know of. The host language looks up the functions of the BLAS it is
using and passes their pointers to the Rust library, which calls through them.

`dlopen("libblas")` from our side does not work here, for three reasons.
First, there is no fixed place to look. NumPy and SciPy wheels carry their own
OpenBLAS inside the wheel, and Julia calls BLAS through
[libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline),
which forwards each call to whatever backend is loaded: OpenBLAS by default,
MKL or Accelerate if the user switched. Second, even when a system BLAS
exists, loading it next to the host's BLAS puts two implementations and two
thread pools in one process. Third, LP64 (32-bit integer) and ILP64 (64-bit
integer) BLAS are different ABIs, and the symbol name does not reliably tell
you which one you have. Some ILP64 builds add a `64_` suffix and others keep
the plain names. The host knows; we do not.

### sparse-ir-rs

[sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs), the Rust
implementation of the intermediate representation (IR) for imaginary-time
propagators developed in SpM-lab, is the clearest example we have. Julia,
Python, C++ and Fortran use it through its C API.

Its core does matrix products through a column-major GEMM written in BLAS
terms, `C ← α·op(A)·op(B) + β·C` with explicit leading dimensions. The default
backend is pure-Rust faer. A BLAS can be injected at run time: in Rust with
`set_blas_backend` (LP64) or `set_ilp64_backend` (ILP64), and
`clear_blas_backend` goes back to faer. The C API has
`spir_gemm_backend_new_from_fblas_lp64` and `_ilp64`, which take pointers to
`dgemm` and `zgemm`. A backend handle can also be passed to each call instead
of setting a global default.

The two wrappers use this as follows:

- Python: if SciPy is installed, `pylibsparseir` takes the `dgemm` and `zgemm`
  pointers out of the PyCapsules that `scipy.linalg.cython_blas` exports and
  registers them as an LP64 backend.
- Julia: SparseIR.jl asks libblastrampoline for the forwarded functions with
  `BLAS.lbt_get_forward`, choosing the ILP64 or LP64 interface from
  `BLAS.USE_BLAS64`, and creates the matching backend.

The same Rust code then calls the BLAS that SciPy or Julia uses, with the right
integer width, and the process has one BLAS.

### tenferro-rs, cblas-inject and lapack-inject

Tensor-network code also needs LAPACK for SVD, QR and eigenvalue problems.
[lapack-inject](https://github.com/shinaoka/lapack-inject) exports all 1315
Fortran LAPACK symbols that `lapack-sys` declares, so crates that expect the
symbols from `lapack-src` link against it. Each symbol forwards to a pointer
registered at run time, and every routine has an LP64 and an ILP64
registration function (`register_dgesv_lp64`, `register_dgesv_ilp64`, and so
on). [cblas-inject](https://crates.io/crates/cblas-inject) does the same for
BLAS.

tenferro-rs uses both behind the `provider-inject` feature of `tenferro-cpu`.
tenferro itself calls BLAS and LAPACK with 32-bit integers. A host registers
its provider pointers with `register_blas_gemm_provider_ptrs` and
`register_lapack_provider_ptrs`, stating whether they are LP64 or ILP64, and
the inject crates convert the integer arguments when the provider is ILP64. A
Julia session with a 64-bit libblastrampoline can therefore hand its BLAS to
tenferro unchanged.

## When to use which

| | `dlopen` | Pointer injection |
|---|---|---|
| Who finds the library | Our library | The host language |
| Fits | A library with a known location and stable symbols (HDF5, MPI through MPIwrapper, CUDA) | A library the host already owns and may swap (BLAS and LAPACK in Julia and Python) |
| Examples | hdf5-rt, rsmpi-rt, cudarc | sparse-ir-rs, cblas-inject, lapack-inject |
| What the host provides | A path, or nothing if the defaults work | The function pointers and their integer width |

For BLAS and LAPACK used from Julia or Python we know of no stable alternative
to injection. For the rest, `dlopen` with a path the host can set is a good
default. In both cases the library should expose the hook and include a
pure-Rust fallback (faer, in the case of sparse-ir-rs) so that it runs with
nothing installed, leaving the choice of a faster library to the user.

## Pitfalls we ran into

- Integer width. Pass an ILP64 pointer where LP64 is expected, or the other way
  round, and you get wrong results or a crash instead of an error. In Julia,
  read `BLAS.USE_BLAS64`. The SciPy `cython_blas` pointers are LP64.
- Thread safety and lifetime. The Rust side calls injected functions from
  several threads and keeps the pointers for as long as the backend lives, so
  the provider must be thread-safe and must stay loaded.
- Fortran calling conventions. Every argument is passed by reference, and
  Fortran compilers add hidden length arguments for character arguments such
  as `'N'` or `'T'`. Details like this belong in one tested crate, not in every
  wrapper.
- Unloading. Like HDF5, libraries with global state may not survive
  `dlclose`, so we leak the handle.
- Untested combinations. sparse-ir-rs runs its C API tests with the default
  backend, OpenBLAS LP64 and OpenBLAS ILP64, and rsmpi-rt runs the mpi4py and
  MPI.jl interoperability test. We assume any combination outside these is
  broken until it is tested.

## Agent-written code

Coding agents write most of our Rust code, and we check it by machine, as
described in
[How we check code that nobody reads line by line](https://tensor4all.org/blog/how-we-check-agent-written-code/).
"Do not link BLAS, LAPACK, MPI, HDF5 or CUDA at build time" is a rule a
machine can check, for example by failing CI when a crate's build script emits
a link directive for one of these libraries.

## Links

- [hdf5-rt](https://github.com/tensor4all/hdf5-rt) · [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) · [lapack-inject](https://github.com/shinaoka/lapack-inject) · [cblas-inject](https://crates.io/crates/cblas-inject)
- [tenferro-rs](https://github.com/tensor4all/tenferro-rs) · [sparse-ir-rs](https://github.com/SpM-lab/sparse-ir-rs)

If you have a native library that is hard to ship to Julia or Python users,
tell us on [Matrix](https://tensor4all.org/matrix.html).
