---
title: 介绍 tenferro-rs
---

# 从 Julia 到 Rust：智能体 AI 时代面向科学计算的可微张量栈

*[tenferro-rs](https://github.com/tensor4all/tenferro-rs) 是一个 Rust 原生的稠密张量栈：线性代数、PyTorch 风格的 eager 自动微分、JAX 风格的 traced 变换、NumPy 风格的 einsum、FFT、可扩展的运算 crate，以及显式的 CPU/CUDA 后端。首批 crate 已于 2026 年 6 月 23 日（JST）发布到 crates.io。*

作者: **Hiroshi Shinaoka**（埼玉大学），tensor4all 团队

*🌐 [English](https://tensor4all.org/blog/introducing-tenferro-rs/) · [日本語](https://tensor4all.org/blog/introducing-tenferro-rs-ja/) · [简体中文](https://tensor4all.org/blog/introducing-tenferro-rs-zh/)*

![tenferro-rs architecture overview](tenferro-architecture.svg)

---

张量网络是用 Julia 写的，这有充分的理由。ITensors 及其周边生态非常出色，原型开发贴近数学表达，多年来这一直是正确的选择。我们自己的工作（IR 基、稀疏建模，以及 [tensor4all](https://tensor4all.org) 的张量交叉插值（TCI）／quantics 栈）也是在那里成长起来的。

但当代码库变大时，Julia 的开发会以一些具体而熟悉的方式变得痛苦：只在运行时才暴露的类型不稳定、拉长「编辑—测试」循环的编译／预编译时间，以及代码越长越难把正确性钉死的隐忧。我们在把张量网络栈移植到更大的系统时撞上了这堵墙，于是开始把引擎迁移到 Rust。

刚一开始，我们就遇到了第二个意外：我们想要依赖的那一层基础张量库，还没有就绪。Rust 有很强的基础部件：做数组的 ndarray、做深度学习的 Burn、做线性代数的 faer。但一个能从自动微分一路贯通到 einsum、足以支撑真正科学计算的生态，仍不成熟。目标从一开始就不是取代这些库，而是补上那块缺失的、起补充作用的栈。

也值得说说基础到底已经走了多远，因为当年的反对意见已经被推翻。crates.io 从 2015 年的 602 个 crate 增长到 2026 年的约 21 万个（[数据](https://github.com/shinaoka/rust_crate_count)），底座是扎实的：稠密线性代数的 faer、GPU 内核的 [CubeCL](https://github.com/tracel-ai/cubecl)、通用数值的 `num-traits` 和 `num-complex`。作为独立的层，部件也都存在：数组有 ndarray，线性代数有 nalgebra 和 faer，深度学习框架有 Burn 和 candle，NumPy 风格的数组 API 有 numr。但它们之中没有一个，是列主序（column-major）、支持动态 shape、同时具备 eager 与 traced 两种自动微分、带 einsum/FFT/CPU/CUDA 和可扩展运算、并且面向科学计算（而非深度学习）的张量栈。那块「中间层」正是 tenferro-rs 要承担的。我们在 faer 和 CubeCL 之上构建它，并回馈而非重新发明；而移植缺失的部件如今也很廉价：SparseIR.jl 和 Julia 的张量网络栈，我们都这样做过。

于是有了 [tenferro-rs](https://github.com/tensor4all/tenferro-rs)。这篇文章谈的是：我们为什么认为它值得做，以及为什么是现在、在智能体 AI 的时代，选择 Rust。

## 选择语言的标准，刚刚反转了

2023 年初，在那霸的一个晚上，我曾这样说：

> 「我在 notebook 里写原型，Julia 很舒服：代码看起来像数学，易于审阅，内存也帮你打理好。我对 Rust 也有兴趣，但对普通学生门槛太高，而且数值计算库还不齐全。」

而 2026 年的我会这样说：

> 「我不再用 notebook 写原型了。Rust 严格而显式的模型，对智能体编程反而更高效。既然写代码的是智能体，陡峭的学习曲线就不是我的问题。即使代码长一些，让它与数学保持一致也不难。缺的数值库？移植过来就是了。」

变的不是 Rust，变的是「谁来写代码」。

Fortran、Python、Julia 都是为让一件事变便宜而设计的：人类用手写、阅读、维护代码的成本。可读性、REPL、贴近数学的记法、平缓的入门曲线。当 AI 来写代码时，这些优势就失去了力量。写得快不再是瓶颈，陡峭的学习曲线由智能体代劳，而「读起来像数学」也不再保证正确：别名（aliasing）、可变性（mutation）、内存分配（allocation），从一行代码的外观上根本看不出来。

于是选择语言的标准发生反转：从代码「易读」转向「易验证」，从可读性转向可检查性（inspectability）。正是这一重新定位，构成了 tenferro-rs 选择 Rust 的全部理由。

## 从一次移植，到一整套栈

我们并不是要做一个通用张量库。我们只是想移植所需的东西，并停止与工具搏斗。但早期下的几个设计赌注，悄悄把一次张量网络移植，变成了更广的东西：

- **模块化的 crate，而非单体。** 运算族各自住在自己的 crate 里，而不是塞进一个大一统的张量类型内部。
- **自动微分规则，放在张量类型之外。** 遵循 Julia/ChainRules 的经验，导数规则属于运算的语义，而不属于某一个具体的张量类。AD 基础设施（[tidu-rs](https://github.com/tensor4all/tidu-rs)、[chainrules-rs](https://github.com/tensor4all/chainrules-rs)）是通用的，张量类型只是它的一个使用者。
- **后端与设备都是显式的。** 数据不会在 CPU 与 GPU 之间悄悄搬运。能力与运行时可用性是两件不同的事。
- **列主序（column-major）存储。** 与 Fortran、Julia、MATLAB、LAPACK/BLAS 对齐，并通过带步长（strided）的视图，在不做 eager 拷贝的情况下桥接行主序数据。

正因为这些选择，结果才成了一个能超越张量网络去使用的东西。

## 两分钟看懂 tenferro-rs

这套栈提供：带类型的张量、带 `backward()` 的即时（eager）执行、带 `grad`/`vjp`/`jvp`/HVP 的 traced 图、线性代数、einsum、FFT，以及显式的 CPU 与 CUDA（以及实验性的 WebGPU）后端。

下面是 PyTorch 风格的 eager 自动微分，计算 `sum(x²)`（其梯度为 `2x`），原样摘自仓库中的 [`eager_autodiff_pytorch_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/eager_autodiff_pytorch_style.rs)：

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

这些断言（assertion）就是程序在验证它自己，对一个以验证为本的库来说，这很贴切。同样的计算，也能作为 JAX 风格的 traced 图运行：编译一次便可复用，并带有 `grad`/`vjp`/`jvp`（见 [`traced_autodiff_jax_style.rs`](https://github.com/tensor4all/tenferro-rs/blob/main/docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs)）。你可以选用能解决问题的最低层：带类型的张量、带自动微分的 eager，或 traced 图。自动微分、CUDA、einsum、FFT、线性代数都是按需启用（opt-in）。

## 在别处真正困难的地方：运行时才知道的 shape

JAX 和 XLA 很出色，直到你的 shape 开始依赖数据。截断阈值、自适应的键维（bond dimension）、依赖数据的迭代次数：一旦尺寸只有在运行时才知道，你要么对每个新 shape 重新编译，要么退回 eager 而失去这些变换（transforms）。

对张量网络、以及许多自适应的科学计算来说，这就是日常。这正是我们所在的地方。

tenferro 把 traced 程序只编译一次，并在具体尺寸（秩、阈值、迭代次数）于运行时被解析的同时**复用**它；在整个过程中 `grad`/`vjp`/`jvp` 都可用，而且是纯 Rust。静态 shape 不是前提，而是一个特例。（对于静态 shape 的图，`tenferro-xla` 还能 lower 到 StableHLO 并在运行时加载 PJRT 插件。）

如果你的 shape 总是固定的，这一点不会打动你。如果不是，那它正是我们会第一个指给你看的特性。

## 为什么在智能体 AI 时代选 Rust（具体地说）

抛开论点本身，日常层面的理由也站得住脚：

- **错误在执行前就被抓住。** 所有权和类型在编译期排除了一大类错误。`cargo check` 几秒内给出答案，所以当智能体出错时，你当场就知道，而不是等到运行时。
- **Cargo 就是构建系统。** 构建、依赖解析、测试、基准测试统一在一个工具里。不需要 CMake，也没有链接期的版本冲突。从零构建整套栈连同依赖，在笔记本上约两分钟，「编辑—测试」循环是几十秒。
- **crate 边界是强制，而非卫生习惯。** Rust 沿模块与 crate 层级控制符号可见性，所以智能体只能在某一层**内部**操作，无法伸手进入另一个 crate 的内部，也无法悄悄破坏抽象。对一个由 AI 编写的代码库（tenferro 约 13 万行）而言，正是这种结构性约束，遏制了复杂度的滚雪球。
- **困难的部分由智能体吸收。** 生命周期与所有权的机械式复杂性（Rust 那道著名的学习陡坡）由智能体来处理，于是人的注意力转向算法、设计与正确性。过去对 Rust 不利的陡峭起步，如今由一个并不介意的对象来承担。

许多人在 C++、Python、Julia 中养成的那种焦虑，即「代码一大，验证就变难」，会就此消失。

## 是被检验的，而不是凭感觉写的（vibe coding）

对一个 AI 编写的数值库，最自然的担忧，和对所有 AI 代码的担忧是一样的：它会不会只是看起来像样的「slop（垃圾产出）」？

我们的答案，是把信任从「阅读代码」转移到不依赖任何人肉眼检查的机制上：

- **正确性**：以有限差分和 PyTorch 参照 oracle 来检验。[tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles) 是一个独立的数据库与生成器，用于张量及线性代数运算的导数正确性，并由不变量、残差检查和来源（provenance）检查支撑。
- **性能**：用可复现的套件 [tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark)，与 PyTorch 和 JAX 比较来测量。在许多目标上，tenferro 在 CPU 性能上已经达到相当水平，并且仍在持续优化、尚有余量。我们在这里链接基准仓库而不写死数字，因为那里才是权威且可复现的来源，而数字会变。
- **设计的一致性**：写下来，而不是留在某个人的脑子里。一份不断生长的 source of truth（REPOSITORY_RULES.md、AGENTS.md、设计笔记、worklog）记录了架构及其约束，人与智能体都遵循它。当一次失败暴露出缺失的规则时（比如「运算族要做成 first-class crate，而不是一个 facade」「禁止朴素的 CPU 循环回退，要用 faer 或 BLAS」），它会变成一条新的、写下来的约束，而不是一次性的补丁。一个 13 万行的代码库之所以能保持单一而统一的设计、不在会话之间漂移，靠的正是这一点。

oracle 和基准都放在**各自独立的仓库**里，所以编写这个库的智能体无法挪动标杆；规则也会与代码持续比对，使两者不会悄悄背离。再加上一套强制按文件覆盖率阈值的完整测试，你得到的就是一种与 vibe coding 截然相反的开发方式：正确性、性能与设计，都是从外部、系统性地确立的，而不是靠肉眼断言。

## 试一试，并参与把它塑造成形

首批 tenferro crate 今天已在 **crates.io** 上，并且是作为模块化的栈本身发布的，而不是一个单一的捆绑包：

```toml
[dependencies]
tenferro-tensor = "0.1"   # tensors, views, backends
tenferro-ad     = "0.1"   # eager + traced autodiff
# plus tenferro-linalg, tenferro-einsum, tenferro-fft, tenferro-cpu,
# tenferro-gpu, tenferro-xla: add only what you need
```

现在还很早期。它是 v0.1，是预览版，而非定型的 1.0。但它并不是玩具：tenferro-rs 已经是我们的 Rust 张量网络栈 [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs)（TreeTN、QTT、TCI）的引擎，在开发过程中已经经受真实科学计算工作负载的检验。如果你的宿主语言是 Python，JAX 和 PyTorch 是自然的选择；tenferro-rs 面向的是那些希望在 Rust 中原生拥有这类栈的项目。而且它足够早期，早到你的反馈能改变它的走向。

我们尤其想听到这些人的声音：用过 ndarray、nalgebra/faer、Burn/candle、PyTorch/JAX，或在科学计算/HPC 中用过 Julia/Fortran 的人。

- **文档与指南：** https://tensor4all.org/tenferro-rs/
- **源码：** https://github.com/tensor4all/tenferro-rs
- **复现基准：** https://github.com/tensor4all/tenferro-benchmark
- **检验正确性：** https://github.com/tensor4all/tensor-ad-oracles

如果你在真实的科学计算工作负载上试用它，发现有缺失、变慢或出错之处，那正是我们想要的反馈。

## 致谢

感谢 [Jin-Guo Liu](https://giggleliu.github.io/) 对 tenferro 早期设计的贡献，感谢 [Satoshi Terasaki](https://terasakisatoshi.github.io/) 的开发支持。

*披露：本文由 AI 编程智能体与作者协作撰写，并以本文所述的、同样由人来验证的工作流完成。*
