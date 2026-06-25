# Software Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single software map page that presents `xfac/Python` and the stable Julia ecosystem as current recommendations, while marking the Rust/Julia rewrite as active development and legacy packages as maintenance mode.

**Architecture:** This is a static Jekyll Markdown update. Create `software.md` as the canonical software entry point, then update `index.md`, `julia.md`, and the FAQ getting-started answer to point users to that map instead of the old two-way Julia versus C++/Python split.

**Tech Stack:** Jekyll, kramdown Markdown, existing `jekyll-theme-slate`.

---

### Task 1: Add Canonical Software Page

**Files:**
- Create: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/software.md`

- [x] **Step 1: Create `software.md` with the software map**

Use this structure:

```markdown
---
title: Software
---
# Software

The tensor4all software ecosystem is organized by status and workflow. For new users today, we recommend `xfac` for C++ and Python workflows, and the stable Julia packages for Julia workflows.

## Software map

| Status | Library | Best for |
| --- | --- | --- |
| Use now | [`xfac`](https://github.com/tensor4all/xfac) / [Python tutorials](https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html) | C++ and Python workflows; original TCI implementation |
| Use now | [`TensorCrossInterpolation.jl`](https://github.com/tensor4all/TensorCrossInterpolation.jl/) | Core TCI algorithms in Julia |
| Use now | [`QuanticsTCI.jl`](https://github.com/tensor4all/QuanticsTCI.jl/) | Convenient QTCI interface in Julia |
| Use now | [`QuanticsGrids.jl`](https://github.com/tensor4all/QuanticsGrids.jl/) | Quantics grids and coordinate transformations |
| Use now | [`InterpolativeQTT.jl`](https://github.com/tensor4all/InterpolativeQTT.jl/) | Multiscale interpolative QTT construction in Julia |
| Active development | [`tensor4all-rs`](https://github.com/tensor4all/tensor4all-rs) | Next-generation Rust implementation |
| Active development | [`Tensor4all.jl`](https://github.com/tensor4all/Tensor4all.jl) | Julia frontend for `tensor4all-rs` |
| Maintenance | [`Quantics.jl`](https://github.com/tensor4all/Quantics.jl/) | Existing QTT workflows built on ITensors.jl |
| Maintenance | [`FastMPOContractions.jl`](https://github.com/tensor4all/FastMPOContractions.jl/) | Existing MPO contraction workflows |
| Maintenance | [`TCIITensorConversion.jl`](https://github.com/tensor4all/TCIITensorConversion.jl/) | Historical ITensors conversion package |
```

- [x] **Step 2: Add explanatory sections**

Add sections named:

```markdown
## Use now: C++ and Python
## Use now: Julia
## Active development: Rust and future Julia frontend
## Maintenance and legacy packages
## Experimental repositories
```

Expected content:

- `xfac` is the original C++ implementation and remains a recommended current path for C++ and Python users.
- Stable Julia packages are recommended for Julia users today.
- `tensor4all-rs` and `Tensor4all.jl` are under active development and may eventually unify or replace parts of the current Julia ecosystem.
- `TCIITensorConversion.jl` functionality has been absorbed into `TensorCrossInterpolation.jl` as an extension.
- `T4A****.jl` experimental repositories are not recommended for new users and are not planned as maintained public libraries.

### Task 2: Update Existing Site Entry Points

**Files:**
- Modify: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/index.md`
- Modify: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/julia.md`
- Modify: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/faq.md`

- [x] **Step 1: Update `index.md` Code section**

Replace the old paragraph that says the site provides two software libraries with a short paragraph linking to `software.html`.

Expected replacement:

```markdown
The tensor4all software ecosystem includes current C++/Python and Julia libraries, actively developed Rust/Julia components, and maintenance-mode packages. See the [software map](software.html) for recommendations on which library to use.
```

- [x] **Step 2: Update `julia.md`**

Keep `julia.md` as a Julia-specific page, but add a first paragraph:

```markdown
For the full cross-language overview, see the [software map](software.html). This page focuses on Julia packages.
```

Update package status language so the stable Julia packages are clear and `Quantics.jl` is maintenance mode.

- [x] **Step 3: Update `faq.md` getting-started answer**

Replace the first answer's old "both the Julia and C++/Python version" wording with:

```markdown
Start with the [software map](software.html). For new users today, we recommend `xfac` for C++ and Python workflows, and the stable Julia packages for Julia workflows.
```

Preserve the existing C++/Python and Julia tutorial links.

### Task 3: Verify Static Site Content

**Files:**
- Check: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/software.md`
- Check: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/index.md`
- Check: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/julia.md`
- Check: `/Users/hiroshi/projects/tensor4all/tensor4all.github.io/faq.md`

- [x] **Step 1: Search for obsolete two-way split wording**

Run:

```bash
rg -n "two software libraries|both the Julia and C\\+\\+/Python version|One code is called Xfac" index.md faq.md julia.md software.md
```

Expected: no matches.

- [x] **Step 2: Build the Jekyll site if dependencies are available**

Run:

```bash
bundle exec jekyll build
```

Expected: build succeeds. If Bundler dependencies are not installed, report that local static build could not be run.

Result: local static build could not be run because this checkout has no
`Gemfile` or `.bundle/` directory, and `jekyll` is not installed in the current
environment.

- [x] **Step 3: Commit the content update**

Run:

```bash
git add software.md index.md julia.md faq.md docs/superpowers/plans/2026-06-26-software-map.md
git commit -m "Reorganize software documentation"
```
