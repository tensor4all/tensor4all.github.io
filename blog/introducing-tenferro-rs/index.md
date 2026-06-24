---
title: Introducing tenferro-rs
---

# From Julia to Rust: a differentiable tensor stack for scientific computing in the agentic AI era

*[tenferro-rs](https://github.com/tensor4all/tenferro-rs) is a Rust-native dense tensor stack: linear algebra, PyTorch-style eager autodiff, JAX-style traced transforms, NumPy-style einsum, FFT, extensible operation crates, and explicit CPU/CUDA backends. The first crates are on crates.io as of June 23, 2026 (JST).*

by **Hiroshi Shinaoka** (Saitama University), for the tensor4all team

*🌐 [English](https://tensor4all.org/blog/introducing-tenferro-rs/) · [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) · [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)*

![tenferro-rs architecture overview](tenferro-architecture.svg)

---

Most tensor-network code is written in Julia, and ours was no exception.
ITensors and the surrounding ecosystem are good for prototyping: the code
stays close to the math, and that matters when you are iterating quickly.
Our own work on the IR basis, sparse modeling, and the
[tensor4all](https://tensor4all.org) tensor-cross-interpolation and quantics
stack grew up there.

But once a numerical codebase gets large, Julia development starts to hurt in
ways that are hard to ignore: type instability that only shows up at run time,
long compile and precompile times that stretch the edit/test loop, and a growing
worry that correctness becomes harder to pin down as the code grows. Porting
our tensor-network stack to a larger system became painful enough that we
started moving the engine to Rust.

As soon as we started, we ran into a second problem. The foundational tensor
library we wanted to build on did not exist yet. Rust has strong pieces:
ndarray for arrays, Burn for deep learning, faer for linear algebra. But an
ecosystem that carries you from autodiff through einsum, and is broad enough to
support real scientific computing, is still underdeveloped. The goal was never
to replace those libraries. It was to build the complementary stack that was
missing.

The foundations have also improved enough that the old objection no longer
holds. crates.io went from 602 crates in 2015 to roughly 210,000 in 2026
([data](https://github.com/shinaoka/rust_crate_count)), and the base is solid:
faer for dense linear algebra, [CubeCL](https://github.com/tracel-ai/cubecl)
for GPU kernels, `num-traits` and `num-complex` for generic numerics. The
pieces also exist as separate layers. ndarray gives you arrays, nalgebra and
faer give you linear algebra, Burn and candle are deep-learning frameworks,
and numr brings a NumPy-style array API. What none of them is, is a
column-major, dynamic-shape, eager-and-traced-autodiff tensor stack with
einsum, FFT, CPU/CUDA, and extensible operations, aimed at scientific computing
rather than deep learning. That middle layer is what tenferro-rs is for. We
build it on faer and CubeCL and contribute back rather than reinvent them, and
porting the missing pieces turns out to be cheap: we did it for SparseIR.jl and
for the Julia tensor-network stack.

That became [tenferro-rs](https://github.com/tensor4all/tenferro-rs). This post
is about why we think it is worth building, and why Rust, now, when the people
writing the code are often AI agents.

## Why Rust now, when Julia was fine before?

A couple of years ago I would have told students to start with Julia, not Rust.
Julia's code looks like the math, the memory is managed, and the numerical
libraries were ready. Rust had a steep learning curve and the ecosystem was not
there yet.

I would not give the same advice now. Not because Rust changed, but because I
am no longer the one writing most of the code.

Fortran, Python, and Julia were all designed to make one thing cheap: a *human*
writing, reading, and maintaining code by hand. Readability, a REPL, notation
close to the math, a gentle on-ramp. When an AI writes the code, those
advantages lose some of their force. Writing speed stops being the bottleneck.
A steep learning curve is absorbed by the agent. And "it reads like the math"
no longer guarantees correctness, because aliasing, mutation, and allocation
are invisible in how a line reads.

For us, the question stopped being "how fast can a human write this?" and
became "how confident can we be that it is correct?". That reframing is the
reason tenferro-rs is in Rust.

## From a port to a stack

We did not set out to build a general tensor library. We set out to port what
we needed and to stop fighting our tools. But a few early design choices quietly
turned a tensor-network port into something broader:

- **Modular crates, not a monolith.** Operation families live in their own crates
  rather than inside one all-in-one tensor type.
- **Autodiff rules live outside the tensor type.** Following the Julia/ChainRules
  lesson, the derivative rule belongs to an operation's *semantics*, not to one
  concrete tensor class. The AD substrate ([tidu-rs](https://github.com/tensor4all/tidu-rs),
  [chainrules-rs](https://github.com/tensor4all/chainrules-rs)) is generic, and the
  tensor type is just one consumer of it.
- **Explicit backends and devices.** Nothing silently moves data between CPU and
  GPU. Capability and runtime availability are separate concerns.
- **Column-major storage**, lining up with Fortran, Julia, MATLAB, and LAPACK/BLAS,
  with strided views to bridge row-major data without eager copies.

These choices are why the result is usable beyond tensor networks.

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

The assertions are the program checking itself, which is fitting for a library
built around verification. The same computation also runs as a
JAX-style traced graph that you compile once and reuse, with `grad`/`vjp`/`jvp`
(see [`traced_autodiff_jax_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs)).
You reach for the lowest layer that solves your problem: typed tensors, eager with
autodiff, or traced graphs. Autodiff, CUDA, einsum, FFT, and linalg are all opt-in.

## Shapes you only know at run time

JAX and XLA are useful until your shapes depend on the data. Truncation
thresholds, adaptive bond dimensions, data-dependent iteration counts: once
sizes are only known at run time, you are recompiling on every new shape, or
falling back to eager and losing the transforms.

That is the daily reality of tensor networks and much of adaptive scientific
computing. It is most of what we do.

tenferro compiles a traced program once and **reuses** it while the concrete
sizes (ranks, thresholds, iteration counts) resolve at execution time, with full
`grad`/`vjp`/`jvp` available throughout, natively in Rust. Static shapes are not
the assumption; they are a special case. (For static-shaped graphs, `tenferro-xla`
can also lower to StableHLO and load PJRT plugins.)

If your shapes are always fixed, you will not see much benefit. If they are not, it is
the first thing we would mention.

## Why Rust now, concretely

Setting the larger argument aside, the practical reasons also matter:

- **Mistakes are caught before execution.** Ownership and types rule out a wide
  range of errors at compile time. `cargo check` answers in seconds, so when the
  agent gets something wrong you find out immediately instead of at run time.
- **Cargo is the build system.** Build, dependency resolution, testing, and
  benchmarking are one tool. No CMake, no link-time version conflicts. A
  from-scratch build of the entire stack plus dependencies is a couple of minutes
  on a laptop, and the edit/test loop is tens of seconds.
- **Crate boundaries are enforcement, not hygiene.** Rust controls symbol
  visibility along the module and crate hierarchy, so an agent can only operate
  *within* a layer. It cannot reach into another crate's internals or quietly
  break the abstraction. For an AI-written codebase (tenferro is about 130K lines),
  that structural constraint is what keeps complexity from compounding.
- **The agent handles the difficult parts.** Lifetimes and ownership mechanics,
  Rust's famous learning cliff, are handled by the agent, so human attention goes
  to the algorithm, the design, and the correctness. The steep early curve that
  used to count against Rust is now paid by the agent.

With Rust, I worry less about whether a large codebase can still be verified.

## Verification, not trust

The obvious worry with AI-written numerical code is that it can look correct
without being correct.

Our answer is to move trust off of reading the code and onto mechanisms that do
not depend on anyone eyeballing the source:

- **Correctness.** We check it against finite-difference and PyTorch reference
  oracles. [tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles) is
  a standalone database and generator for derivative-correctness of tensor and
  linear-algebra operations, backed by invariants, residual checks, and provenance
  checks.
- **Performance.** We measure it with a reproducible suite,
  [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark), against
  PyTorch and JAX. On many targets tenferro already reaches CPU performance
  comparable to them, and we are actively optimizing, with headroom left. We link
  the benchmark repo rather than quote numbers here, because it is the canonical,
  reproducible source and the numbers move.
- **Design consistency.** We write it down instead of keeping it in someone's
  head. A growing source of truth (REPOSITORY_RULES.md, AGENTS.md, design notes,
  worklogs) records the architecture and its constraints, and both the humans and
  the agent follow it. When a failure exposes a missing rule, say "operation families are
  first-class crates, not a facade" or "no naive CPU loop fallbacks, use faer or
  BLAS", it becomes a new written constraint instead of a one-off patch. That is
  how a 130K-line codebase keeps one unified design rather than drifting from
  session to session.

The oracles and benchmarks live in *separate repositories*, so the agent that
writes the library cannot move the goalposts, and we check the rules against
the code continuously so the two cannot quietly diverge. Add a comprehensive test
suite with enforced per-file coverage thresholds, and you establish correctness,
performance, and design systematically and from the outside, rather than asserting
them by eye.

## Try it

The first tenferro crates are on **crates.io** now, published as separate crates
so you can pull in only what you need:

```toml
[dependencies]
tenferro-tensor = "0.1"   # tensors, views, backends
tenferro-ad     = "0.1"   # eager + traced autodiff
# plus tenferro-linalg, tenferro-einsum, tenferro-fft, tenferro-cpu,
# tenferro-gpu, tenferro-xla: add only what you need
```

This is early. It is v0.1, a preview rather than a settled 1.0. But it is not
just an experiment: tenferro-rs is already the engine under
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs), our Rust
tensor-network stack (TreeTN, QTT, TCI), so it is exercised on real scientific
workloads as we build. If your host language is Python, JAX and PyTorch are the
natural choice; tenferro-rs is for projects that want this kind of stack natively
in Rust. It is early enough that your feedback changes where it goes.

We would especially like to hear from people who have used ndarray, nalgebra/faer,
Burn/candle, PyTorch/JAX, or Julia/Fortran in scientific and HPC work:

- **Docs and guides:** https://tensor4all.org/tenferro-rs/
- **Source:** https://github.com/tensor4all/tenferro-rs
- **Reproduce the benchmarks:** https://github.com/tensor4all/tenferro-benchmark
- **Kick the tires on correctness:** https://github.com/tensor4all/tensor-ad-oracles

If you try it on a real scientific workload and something is missing, slow, or
wrong, that is the most useful report for us.

## Acknowledgments

Thanks to [Jin-Guo Liu](https://giggleliu.github.io/) for the initial tenferro
design, and to [Satoshi Terasaki](https://terasakisatoshi.github.io/) for
development support.

*Disclosure: this post was written collaboratively by AI coding agents and the
author, in the same human-verified workflow it describes.*
