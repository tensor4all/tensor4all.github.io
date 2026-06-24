---
title: Introducing tenferro-rs
---

# From Julia to Rust: a differentiable tensor stack for scientific computing in the agentic AI era

*[tenferro-rs](https://github.com/tensor4all/tenferro-rs) is a Rust-native dense tensor stack: linear algebra, PyTorch-style eager autodiff, JAX-style traced transforms, NumPy-style einsum, FFT, extensible operation crates, and explicit CPU/CUDA backends. The first crates are on crates.io as of June 23, 2026 (JST).*

by **Hiroshi Shinaoka** (Saitama University), for the tensor4all team

*🌐 [English](https://tensor4all.org/blog/introducing-tenferro-rs/) · [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) · [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)*

![tenferro-rs architecture overview](tenferro-architecture.svg)

---

Most tensor-network code has been written in Julia, and ours was no exception.
ITensors and the surrounding ecosystem are good for prototyping: the code stays
close to the math, and it is easy to iterate. Our own work on the IR basis,
sparse modeling, and the [tensor4all](https://tensor4all.org)
tensor-cross-interpolation and quantics stack started there.

Once the codebase gets large, though, Julia development starts to slow down:
type instability that only shows up at run time, compile and precompile times
that stretch the edit/test loop, and the sense that correctness gets harder to
check as the code grows. When we started fitting the tensor-network stack into a
larger system, that became hard to ignore. We began moving the compute engine to
Rust.

That immediately exposed a second problem. The tensor library we wanted to build
on was not there yet. Rust has libraries for individual jobs: ndarray for
arrays, Burn for deep learning, faer for linear algebra. What was missing was a
tensor layer that could cover autodiff through einsum and still feel usable for
scientific computing. The goal was not to replace those libraries. It was to
connect the pieces that already exist.

The Rust ecosystem has changed a lot in the last few years. crates.io went from
602 crates in 2015 to roughly 210,000 in 2026
([data](https://github.com/shinaoka/rust_crate_count)). For dense linear
algebra there is faer; for GPU kernels, [CubeCL](https://github.com/tracel-ai/cubecl);
for generic numerics, `num-traits` and `num-complex`. There are also libraries
at nearby layers: ndarray for arrays, nalgebra and faer for linear algebra, Burn
and candle for deep learning, and numr for a NumPy-style array API. What we
needed was the layer between them: a scientific-computing tensor stack with
column-major storage, dynamic shapes, eager and traced autodiff, einsum, FFT,
CPU/CUDA backends, and extensible operations. That is what tenferro-rs is for.
We build on faer and CubeCL and add the missing parts instead of reinventing
them. Porting SparseIR.jl and Julia tensor-network code made it clearer where
that missing layer was.

That is the background for [tenferro-rs](https://github.com/tensor4all/tenferro-rs).
This post explains why we are building it, and why we chose Rust now that code
is no longer written only by humans.

## Why Rust now, when Julia was fine before?

A couple of years ago I would probably have told students to start with Julia.
Julia code can stay close to the math, memory management is easy, and the
numerical libraries were already there. Rust had more to learn, and the
ecosystem was still missing pieces.

I would not give the same advice now. Not because Rust changed, but because I
am no longer the one writing most of the code.

Fortran, Python, and Julia all developed around lowering the cost for humans to
write, read, and maintain code by hand. Readability, a REPL, notation close to
the math, and a low barrier to entry all matter for that. When AI writes more of
the code, the tradeoff changes. Writing speed matters less. Much of the learning
cost can be handled by the agent. But "it reads like the math" still does not
guarantee correctness: aliasing, mutation, and allocation are not visible from
the surface of a line.

For us, the question stopped being "how fast can a human write this?" and
became "how confident can we be that it is correct?". That reframing is why Rust
became the more practical choice.

Concretely:

- Ownership and types rule out a wide range of errors at compile time. `cargo
  check` answers in seconds, so when the agent gets something wrong, we find out
  before running the program.
- Cargo handles builds, dependency resolution, tests, and benchmarks in one
  place. No CMake, no link-time version conflicts. A from-scratch build of the
  full stack plus dependencies takes a couple of minutes on a laptop, and the
  edit/test loop is tens of seconds.
- Rust controls symbol visibility along module and crate boundaries. An agent
  can only work *inside* a layer; it cannot reach into another crate's internals
  and quietly break the abstraction. In an AI-written codebase of about 130K
  lines, that boundary matters.
- Lifetimes and ownership mechanics can largely be left to the agent, so human
  attention goes to algorithms, design, and correctness. The early learning cost
  that used to count against Rust is less of a problem now.

In C++, Python, and Julia, large codebases tend to come with the worry that they
are becoming too hard to verify. With Rust, that worry is noticeably smaller.

## From a port to a stack

We did not set out to build a general tensor library. We wanted to port the
pieces we needed and spend less time fighting the tools. But as the
implementation progressed, it became clear that autodiff, backends, and the way
new operations are added should not be trapped inside a tensor-network-specific
layer. We designed the shared parts as an independent tensor stack.

- Operation families live in their own crates rather than inside one all-in-one
  tensor type.
- Autodiff rules live outside the tensor type. Following the Julia/ChainRules
  lesson, the derivative rule belongs to the operation itself, not to one
  concrete tensor class. The AD substrate, [tidu-rs](https://github.com/tensor4all/tidu-rs),
  is generic, and the tensor type is just one consumer of it.
- Backends and devices are explicit. Nothing silently moves data between CPU and
  GPU. We also keep separate the question of which backend can execute an
  operation and which devices are available at run time.
- Storage is column-major, matching Fortran, Julia, MATLAB, and LAPACK/BLAS.
  Row-major data can still be handled through strided views, without unnecessary
  eager copies.

That design makes the stack useful outside tensor networks too.

## tenferro-rs in two minutes

The stack gives you typed tensors, immediate (eager) execution with `backward()`,
traced graphs with `grad`/`vjp`/`jvp`/HVP, linear algebra, einsum, FFT, and
explicit CPU and CUDA backends (plus experimental WebGPU).

Here is PyTorch-style eager autodiff for `sum(x²)`, whose gradient is `2x`, copied
verbatim from the repo's
[`eager_autodiff_pytorch_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/eager_autodiff_pytorch_style.rs):

```rust
use tenferro_ad::{EagerRuntime, Tensor};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = EagerRuntime::new();
    let x = runtime.variable_from(Tensor::from_vec_col_major(
        vec![3],
        vec![1.0_f64, 2.0, 3.0],
    )?)?;

    let prediction = x.mul(&x).unwrap();
    let loss = prediction.reduce_sum(&[0])?;

    assert_eq!(loss.shape(), &[]);
    assert_close(loss.materialized()?.as_slice::<f64>().unwrap(), &[14.0]);

    loss.backward()?;

    let grad = x
        .grad()?
        .expect("tracked variable should receive a gradient");
    assert_eq!(grad.shape(), &[3]);
    assert_close(grad.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    x.clear_grad()?;
    assert!(x.grad()?.is_none());

    Ok(())
}
```

In this example, the program checks its own result. That small example reflects
the way we build the library. The same computation can also run as a JAX-style
traced graph, compiled once and reused, with `grad`/`vjp`/`jvp`
(see [`traced_autodiff_jax_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs)).
You can choose the layer you need: typed tensors, eager execution with autodiff,
or traced graphs. Autodiff, CUDA, einsum, FFT, and linalg can all be enabled only
when needed.

## Computations whose sizes depend on data

JAX and XLA are good at optimizing a computation whose shapes are fixed and then
running it quickly again for inputs of the same shape. Once shapes depend on the
data, things get harder. Truncation thresholds, adaptive bond dimensions,
data-dependent iteration counts: if sizes are only known at run time, every new
shape can force a recompilation. If you avoid recompilation by falling back to
eager execution, transforms such as `grad` and `vjp` no longer fit into the same
workflow.

That is the daily reality of tensor networks and much of adaptive scientific
computing. It is most of what we do. For calculations where ranks or bond
dimensions change with the data, reusing the same traced program without
recompilation matters.

tenferro compiles a traced program once and **reuses** it even when concrete
sizes (ranks, thresholds, iteration counts) are decided at run time. `grad`,
`vjp`, and `jvp` remain available. At the same time, static-size computations can
use OpenXLA. `tenferro-xla` lowers graphs to StableHLO and can load PJRT plugins,
so in principle it can reach the same execution speed as JAX.

## Checking correctness from the outside

The obvious worry with AI-written numerical code is that it can look correct
without being correct.

So we do not make reading the code the only basis for trust. We set up separate
checks that do not depend on someone eyeballing the source:

- Correctness is checked against finite-difference and PyTorch reference
  oracles. [tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles)
  is a standalone database and generator for derivative-correctness checks of
  tensor and linear-algebra operations, with invariants, residual checks, and
  provenance checks.
- Performance is measured with the reproducible
  [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) suite
  against PyTorch and JAX. On many targets, tenferro already reaches CPU/GPU
  performance close to them, with more optimization still left. We link the
  benchmark repository rather than freezing numbers here, because that is the
  reproducible reference and the numbers change.
- Design is written down. REPOSITORY_RULES.md, AGENTS.md, design notes, and
  worklogs record the architecture and its constraints for both humans and
  agents. When a failure exposes a missing rule, such as "operation families are
  first-class crates, not a facade" or "no naive CPU loop fallbacks, use faer or
  BLAS", it becomes a new constraint instead of a one-off fix. That keeps the
  design from drifting in a 130K-line codebase.

The oracles and benchmarks live in separate repositories, so the agent writing
the library cannot move the goalposts. We also check rules against code
continuously, so drift is caught early. Add tests with enforced per-file coverage
thresholds, and correctness, performance, and design are checked through separate
mechanisms rather than by review alone.

## Try it

The first tenferro crates are on **crates.io**, published as separate crates so
you can pull in only what you need:

```toml
[dependencies]
tenferro-tensor = "0.1"   # tensors, views, backends
tenferro-ad     = "0.1"   # eager + traced autodiff
# plus tenferro-linalg, tenferro-einsum, tenferro-fft, tenferro-cpu,
# tenferro-gpu, tenferro-xla: add only what you need
```

tenferro-rs is still a v0.1 preview; the API is not a finalized 1.0. But it is
not just an experiment. We already use it as the engine under
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs), our Rust
tensor-network stack (TreeTN, QTT, TCI), and develop it against real scientific
workloads. If your host language is Python, JAX and PyTorch are the natural
choice. tenferro-rs is for people who want similar capabilities directly from
Rust. At this preview stage, user feedback can still shape the design.

We especially want people who have used ndarray, nalgebra/faer, Burn/candle,
PyTorch/JAX, or Julia/Fortran in scientific and HPC work to try it:

- Docs and guides: https://tensor4all.org/tenferro-rs/
- Source: https://github.com/tensor4all/tenferro-rs
- Reproduce the benchmarks: https://github.com/tensor4all/tenferro-benchmark
- Check correctness: https://github.com/tensor4all/tensor-ad-oracles

If you try it on real scientific workloads and find something missing, slow, or
wrong, that report is especially useful.

## Acknowledgments

Thanks to [Jin-Guo Liu](https://giggleliu.github.io/) for the initial tenferro
design, and to [Satoshi Terasaki](https://terasakisatoshi.github.io/) for
development support.

*Disclosure: this post was written collaboratively by AI coding agents and the
author, in the same human-verified workflow it describes.*
