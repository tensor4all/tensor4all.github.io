---
title: Software
---
<p class="eyebrow">Software</p>
# Pick by what you want to do

tensor4all maintains two generations of software. The established C++, Python and Julia libraries are stable and documented. The Rust stack is where new development happens, and its lower layers are general purpose tools for computational physics, not only for tensor networks.

<div class="chooser">
  <a class="choice" href="#established">
    <span class="q">I want to use TCI or quantics tensor trains today</span>
    <span class="a">xfac (C++, Python) or the Julia packages</span>
  </a>
  <a class="choice" href="#tensor4all-rs">
    <span class="q">I want the newest tensor network algorithms</span>
    <span class="a">tensor4all-rs, or Tensor4all.jl from Julia</span>
  </a>
  <a class="choice" href="#tenferro-rs">
    <span class="q">I need dense tensors with autodiff in Rust</span>
    <span class="a">tenferro-rs</span>
  </a>
  <a class="choice" href="#kernels">
    <span class="q">I need fast contraction or batched linear algebra in my own code</span>
    <span class="a">tcontract, tlinalg, cpueinsum, strided-rs</span>
  </a>
  <a class="choice" href="#kernels">
    <span class="q">I want to run many tasks over threads and MPI ranks</span>
    <span class="a">hataori</span>
  </a>
  <a class="choice" href="#maintenance">
    <span class="q">I have an existing workflow on an older package</span>
    <span class="a">Maintenance packages</span>
  </a>
</div>

## The Rust stack <span class="new">New</span> {#rust}

A layered set of Rust libraries. Each layer depends only on the layers below it, and each can be used on its own. All of them pass explicit execution contexts instead of starting hidden thread pools, so they compose predictably inside MPI programs, Julia sessions and other host codes.

<figure class="figure">
  <div class="map-scroll"><div class="stack-map" data-focus="all"></div></div>
  <figcaption>Boxes link to the repositories. The <a href="/tensor4all-meta/stack/">stack overview</a> explains each layer, with guides <a href="/tensor4all-meta/stack/users.html">for tensor network users</a> and <a href="/tensor4all-meta/stack/backend.html">for backend developers</a>.</figcaption>
</figure>

### tensor4all-rs: tensor network algorithms {#tensor4all-rs}

<div class="lib">
  <div>
    <p>The next generation implementation of the tensor4all algorithms: tensor cross interpolation, quantics tensor trains with Fourier and shift operators, tree tensor networks of arbitrary topology, patching and HDF5 I/O. A C API serves language bindings, and <a href="https://github.com/tensor4all/Tensor4all.jl">Tensor4all.jl</a> is its Julia frontend for light use and teaching.</p>
    <p>New algorithms land here first. It suits early adopters comfortable with Rust; crates are used as git dependencies for now.</p>
  </div>
  <ul class="facts">
    <li><span class="pill wip">active development</span></li>
    <li><a href="https://tensor4all.org/tensor4all-rs/">Documentation</a></li>
    <li><a href="https://github.com/tensor4all/tensor4all-rs">Source</a></li>
    <li><a href="https://github.com/tensor4all/tensor4all-benchmark">Benchmarks</a></li>
  </ul>
</div>

### tenferro-rs: tensor runtime and autodiff {#tenferro-rs}

<div class="lib">
  <div>
    <p>Typed and dynamic dense tensors with einsum, linear algebra and FFT on CPU, CUDA and XLA backends. Eager mode offers PyTorch style <code>backward()</code>; traced mode offers JAX style graph transforms with VJP, JVP and higher order derivatives. Column major storage and explicit device control make it a natural fit for code that talks to LAPACK, Fortran or Julia.</p>
    <p>Ready for production use with a relatively stable API. Start with <code>tenferro-runtime</code>, <code>tenferro-cpu</code> and the operation crates you need. Background: <a href="{{ '/blog/introducing-tenferro-rs/' | relative_url }}">Introducing tenferro-rs</a> (<a href="{{ '/blog/introducing-tenferro-rs-ja/' | relative_url }}">日本語</a>, <a href="{{ '/blog/introducing-tenferro-rs-zh/' | relative_url }}">中文</a>).</p>
  </div>
  <ul class="facts">
    <li><span class="pill ok">crates.io 0.7.1</span></li>
    <li><a href="https://tensor4all.org/tenferro-rs/">Documentation</a></li>
    <li><a href="https://github.com/tensor4all/tenferro-rs">Source</a></li>
    <li><a href="https://github.com/tensor4all/tenferro-benchmark">Benchmarks vs PyTorch and JAX</a></li>
  </ul>
</div>

### Kernels and infrastructure {#kernels}

These libraries know nothing about tensor networks or about tenferro's tensor types. They take borrowed strided views and an explicit context, which keeps them small and reusable from other codes, including from C.

<div class="cards">
  <div class="card">
    <h3><a href="https://github.com/tensor4all/tcontract-rs">tcontract</a></h3>
    <p>Binary tensor contraction over arbitrary strides, with a TAPP and DLPack C ABI. <a href="https://github.com/tensor4all/tcontract-benchmark">Benchmarks</a>.</p>
    <span class="pill wip">git</span>
  </div>
  <div class="card">
    <h3><a href="https://github.com/tensor4all/tlinalg-rs">tlinalg</a></h3>
    <p>Batched SVD, QR, LU, Cholesky and eigensolvers, faer backed or LAPACK backed, optionally parallelized along the batch dimension. <a href="https://github.com/tensor4all/tlinalg-benchmark">Benchmarks</a>.</p>
    <span class="pill wip">git</span>
  </div>
  <div class="card">
    <h3><a href="https://github.com/tensor4all/cpueinsum-rs">cpueinsum</a></h3>
    <p>N-ary einsum over integer labels with a caller supplied contraction order.</p>
    <span class="pill wip">git</span>
  </div>
  <div class="card">
    <h3><a href="https://github.com/tensor4all/strided-rs">strided-rs</a></h3>
    <p>The strided view types every layer shares, with permutation and elementwise kernels. <a href="https://github.com/tensor4all/strided-rs-benchmark-suite">Benchmarks</a>.</p>
    <span class="pill ok">crates.io 0.4.8</span>
  </div>
  <div class="card">
    <h3><a href="https://github.com/tensor4all/hataori-rs">hataori</a></h3>
    <p>Data parallel map and pmap over serial, Rayon, MPI and hybrid execution.</p>
    <span class="pill wip">git</span>
  </div>
  <div class="card">
    <h3><a href="https://github.com/tensor4all/faer">t4a-faer</a></h3>
    <p>Our fork of <a href="https://github.com/sarah-quinones/faer-rs">faer</a>, the pure Rust linear algebra library by Sarah Quinones. It adds strided and batched entry points, caller owned workspaces, an explicit thread policy and numerical fixes; every change is recorded in its <a href="https://github.com/tensor4all/faer/blob/main/PATCHSET.md">patch ledger</a>.</p>
    <span class="pill ok">crates.io t4a-faer 0.24.4</span>
  </div>
</div>

### Benchmarks {#benchmarks}

Performance claims are backed by public benchmark repositories, kept separate from the libraries so that results can be keyed by commit and hardware.

| Repository | What it measures |
| --- | --- |
| [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) | tenferro-rs against PyTorch and JAX on CPU and GPU |
| [tensor4all-benchmark](https://github.com/tensor4all/tensor4all-benchmark) | Tensor network algorithms in tensor4all-rs: QTT products and MPO-MPO contraction |
| [tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark) | Binary strided contraction in tcontract |
| [tlinalg-benchmark](https://github.com/tensor4all/tlinalg-benchmark) | Batched factorizations in tlinalg |
| [strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) | strided-rs kernels against HPTT, Julia Base and Strided.jl |

<p class="small">Release status changes quickly; the <a href="/tensor4all-meta/stack/#repos">repository table</a> in the stack overview is kept current.</p>

## Established libraries {#established}

Stable, documented and in use in published work. Choose these when you need a dependable workflow now.

<div class="cards">
  <div class="card">
    <span class="tag ok">C++ and Python</span>
    <h3><a href="https://github.com/tensor4all/xfac">xfac</a></h3>
    <p>The original implementation of tensor cross interpolation, with Python bindings.</p>
    <ul>
      <li><a href="https://xfac.readthedocs.io/en/latest/intro.html">Documentation</a></li>
      <li><a href="https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html">Python tutorials</a></li>
    </ul>
  </div>
  <div class="card">
    <span class="tag ok">Julia</span>
    <h3>TCI in Julia</h3>
    <ul>
      <li><a href="https://github.com/tensor4all/TensorCrossInterpolation.jl/">TensorCrossInterpolation.jl</a>: core TCI algorithms</li>
      <li><a href="https://github.com/tensor4all/QuanticsTCI.jl/">QuanticsTCI.jl</a>: quantics TCI for non experts</li>
      <li><a href="https://github.com/tensor4all/QuanticsGrids.jl/">QuanticsGrids.jl</a>: grids and coordinate transforms</li>
      <li><a href="https://github.com/tensor4all/InterpolativeQTT.jl/">InterpolativeQTT.jl</a>: multiscale interpolative QTT</li>
    </ul>
    <a class="go" href="{{ '/juliatutorials/' | relative_url }}">Julia tutorials &rarr;</a>
  </div>
</div>

## Maintenance packages {#maintenance}

Kept working for existing users, not recommended for new projects.

| Package | Status |
| --- | --- |
| [Quantics.jl](https://github.com/tensor4all/Quantics.jl/) | High level QTT operations on ITensors.jl |
| [FastMPOContractions.jl](https://github.com/tensor4all/FastMPOContractions.jl/) | Existing MPO contraction workflows |
| [TCIITensorConversion.jl](https://github.com/tensor4all/TCIITensorConversion.jl/) | Historical; now an extension of TensorCrossInterpolation.jl |

Old `T4A****.jl` repositories were experiments and are not maintained as public libraries.

<script src="https://tensor4all.org/tensor4all-meta/stack/assets/stack-map.js" defer></script>
