# Frequently Asked Questions

<!-- - [Q1: What are differences from other tensor network libraries supporting machine learning?](#q1) -->

1. placeholder
{:toc}

---

### How do I get started?
We have an extensive set of online tutorials that teach you how to use our libraries, for both the Julia and C++/Python version of our library.

- **C++/Python**: Install the library using the [installation instructions](https://github.com/tensor4all/xfac/blob/main/README.md#installation), then follow the [tutorial pages](https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html).
- **Julia**: Tutorials are available [online](https://tensor4all.org/juliatutorials/index.html) and can be run in your web browser. If you wish to run notebooks locally on your computer, install julia using the [installation instructions](https://tensor4all.org/juliatutorials/index.html#install-julia), then copy-paste the instructions at the top of each online notebook to download and run that specific notebook.

These tutorials should enable you to implement the most common use cases. In many cases, it is sufficient to copy-paste and slightly modify some tutorial code. For more advanced use cases, it may be useful to look at the code in appendix B of our publication [NunezFernandez2024](http://arxiv.org/abs/2407.02454), and to consult the documentation:

- **C++/Python**: [Documentation](https://xfac.readthedocs.io/en/latest/intro.html)
- **Julia**: [Documentation](https://tensor4all.org/julia.html)

### How do I map a function to a tensor?
Consider a function \(f(x_1, x_2, \ldots, \x_N)\), where each \(x_n\in [0, 1]\). To convert it to a tensor \(A_{\sigma_1, \sigma_2, \ldots, \sigma_L}\), the following two recipes are commonly used:
1. The *natural representation*, where each \(x_n\) is discretized on a grid \(x_n(\sigma_n), \sigma_n \in \{1, \ldots, M_n\}\). This grid may or may not be uniform; for example, we use a Gauss--Kronrod grid to compute integrals.
2. The *quantics representation*, where each \(x_n\) is discretized on a grid with \(2^R\) points, with indices \(\sigma_n^r \in \{0, 1\}\) that correspond to the bits in a binary representation of \(x_n\):
\[
    x_n = (0.\sigma_1^n\sigma_2^n\ldots\sigma_R^n)_2 =
    \frac{\sigma_1^n}{2^1} + \frac{\sigma_2^n}{2^2} + \ldots + \frac{\sigma_R^n}{2^R}.
\]
These bits \(\sigma_r^n\) are then relabeled to \(\sigma_\ell\) in one of several patterns. In the *serial representation*, indices are grouped according to \(n\), such that \(\sigma_1 = \sigma_1^1, \sigma_2 = \sigma_2^1, \ldots, \sigma_R = \sigma_R^1, \sigma_{R+1} = \sigma_1^2), and so on.
In the *interleaved representation*, indices are grouped according to \(r\) instead, such that \(\sigma_1 = \sigma_1^1, \sigma_2 = \sigma_1^2, \ldots, \sigma_N = \sigma_1^N, \sigma_{N+1} = \sigma_2^1\), and so on. Other relabeling patterns are possible.

This map can have dramatic influence on the bond dimension when decomposing the resulting tensor. The optimal choice depends on the function in question. This is discussed in part in ["When is it advisable to use the quantics representation?"](#when-is-it-advisable-to-use-the-quantics-representation) and ["In the quantics representation, which index ordering is optimal?"](#in-the-quantics-representation-which-index-ordering-is-optimal).

See also
- [The introduction to quantics on tensornetwork.org](https://tensornetwork.org/functions/).
- Khoromskij, *O(dlog N)-Quantics Approximation of N-d Tensors in High-Dimensional Numerical Modeling*, [Constr. Approx. 2, 34 (2011)](https://doi.org/10.1007/s00365-011-9131-1).

### Are all tensors/functions compressible?
No: random noise, for example, is not compressible. The tensor train format exploits *structure* for its compression, and random noise lacks any structure that can be used for this purpose.
More generally, the tensor train is a partial factorization of a tensor. If the tensor is a discretized function of many arguments, the tensor train is efficient if the function is close to being factorizable in its arguments. For a function in quantics representation, the tensor is factorizable if the function is factorizable in its length scales. This includes all polynomials and linear combinations of exponential functions ([Lindsey2024](https://arxiv.org/abs/2311.12554)). If multiple factorizable structures are added or multiplied in the function, the function is typically also compressible.

Beyond the arguments above, our understanding of the compressibility of functions in tensor train format is still evolving and the subject of current research.
If you are interested in a specific use case, is often easier to just try out whether some function of interest is compressible using an existing dataset. Instructions for this can be found in ["How can I test whether my data is TCI compressible?"](#how-can-i-test-whether-my-data-is-tci-compressible).

See also
- Lindsey, *Multiscale interpolative construction of quantized tensor trains*, [arXiv:2311.12554](https://arxiv.org/abs/2311.12554).
- Khoromskij, *O(dlog N)-Quantics Approximation of N-d Tensors in High-Dimensional Numerical Modeling*, [Constr. Approx. 2, 34 (2011)](https://doi.org/10.1007/s00365-011-9131-1).

### How can I test whether my data is TCI compressible?
Using our libraries, it is easy to try whether some function or dataset of interest is compressible. [The tutorial on compressing existing data](https://tensor4all.org/T4APlutoExamples/pluto_notebooks/compress.html) demonstrates how to do this for TCI and quantics TCI. Replace the tutorial dataset with your own and set the tolerance to whatever precision you require. Note that the tolerance has to be set well above the noise level in your dataset to get an accurate idea about the compressibility of your dataset. Otherwise, TCI will try to compress your random noise, which may lead to an extreme increase in bond dimension.

### What is the difference between SVD-based and TCI compression?
In tensor network algorithms, tensor trains or MPS are usually constructed using the [*singular value decomposition*](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) and truncation by singular values. It has been shown that this compression is optimal in the L2-norm of the resulting error. However, SVD can only be performed with knowledge of all components of the original tensor. In contrast, TCI is able to construct a tensor train using only a subset of components, such that the full tensor never has to be constructed explicitly. Thus, it is possible to construct a tensor train for tensors that would never fit into memory!

### How does Tensor Cross Interpolation (TCI) work?
Very briefly, TCI is a sweeping optimization algorithm. It starts from a tensor train with very small bond dimension, and then optimizes it using a series of local updates. This initial tensor train is constructed from a set of initial pivots, i.e. initial sample points. 

For local updates, it relies on the *cross interpolation* (CI) factorization instead of SVD. CI extracts a subset of rows (blue) and columns (red) of some matrix to be factorized, and arranges the rows and columns in the following structure:

![](mci.svg)

All red, blue, and purple elements of the original matrix are approximated exactly, i.e. without error.
The elements where a row and column cross (purple) are called  *pivots*. Since the pivots fully determine the subset of rows and columns, optimizing a CI boils down to optimizing the pivots.

CI has the important advantage that all components of the resulting factorization are components of the original matrix. This means that the tensor generalization, TCI, can be constructed and optimized using small slices of the original tensor, and an explicit representation of the full tensor is not required.

TCI as implemented in the tensor4all libraries is explained in detail in [NunezFernandez2024](https://arxiv.org/abs/2407.02454).

See also:
- Oseledets and Tyrtyshnikov, *TT-cross approximation for multidimensional arrays*, [Linear Algebra Appl. 432 (1), 70-88](https://www.sciencedirect.com/science/article/pii/S0024379509003747) (2010).
- Núñez Fernández, Jeannin, Dumitrescu, Kloss, Kaye, Parcollet, and Waintal, *Learning Feynman Diagrams with Tensor Trains*, [Phys. Rev. X 12, 041018](https://link.aps.org/doi/10.1103/PhysRevX.12.041018) (2022), [arXiv:2207.06135](https://arxiv.org/abs/2207.06135).
- Núñez Fernández, Ritter, Jeannin, Li, Kloss, Louvet, Terasaki, Parcollet, von Delft, Shinaoka, Waintal, *Learning Tensor Networks with Tensor Cross Interpolation: New Algorithms and Libraries*, [arXiv:2407.02454](http://arxiv.org/abs/2407.02454).

### What is a "pivot" in the context of TCI?
Roughly, an element of the original tensor that is included in the tensor train approximation. If the pivot was chosen by TCI, it is an element of the original tensor that contributes important information; see also ["How does Tensor Cross Interpolation (TCI) work?"](#how-does-tensor-cross-interpolation-tci-work).

If the nesting condition is met, the pivots are represented exactly by the tensor train, i.e. with 0 error (see [NunezFernandez2024](https://arxiv.org/abs/2407.02454)).

### What is the relation between Tensor Cross Interpolation (TCI) and machine learning?
Tensor Cross Interpolation (TCI) can be seen as a machine learning algorithm, in that it samples a small subset of a large dataset (the tensor to be approximated) to learn a representation (the tensor train) that should generalize to the whole dataset. It is an active learning algorithm in that the sampled subset is not given externally, but dynamically chosen by the algorithm based on what it has learned so far.

### What are differences from other tensor network libraries that support machine learning?
A critical difference is that our library is based on Tensor Cross Interpolation (TCI). TCI allows us to compute tensor networks with a low-rank representation by exploring a tiny subset of the full tensor. This opens up the possibility of applying tensor networks to unconventional applications such as high-dimensional integration, i.e., replacing Monte Carlo integration.

### What are the limitations of your Tensor Cross Interpolation (TCI) algorithms? When do they fail?
As explained [above](#what-is-the-relation-between-tensor-cross-interpolation-tci-and-machine-learning), TCI is a machine learning algorithm that only samples a small subset of tensor components to learn the whole tensor. This leads to an inherent limitation: The fact that all samples seen by the algorithm match some learned structure does not ensure that this structure extends to all other components of the tensor; nor does it imply that no additional structure is present in the components not sampled so far. This limitation cannot, in principle, be avoided by sampling algorithms.

This limitation implies that one has to be careful when applying TCI to tensors or functions that have several disconnected "interesting regions" (with lots of structure) that are separated by "boring regions" (with trivial structure). Since 2-site TCI optimizes pairs of neighbouring tensors, two interesting regions are connected if they can be reached by updating a pair neighbouring indices. If your function of interest has several disconnected interesting regions, it may happen that only the region that contains the [intial pivot](#how-does-tensor-cross-interpolation-tci-work) is approximated correctly, and the algorithm never samples the disconnected structure. Such cases can be fixed by specifying at least one initial pivot in each region, as described in the documentation:
- **C++/Python**: [Set the value of `pivot1` in the `TensorCI2Param` struct.](https://xfac.readthedocs.io/en/latest/api/api1.html#_CPPv4N4xfac14TensorCI2ParamE)
- **Julia**: [Use the argument `initialpivots` of `crossinterpolate2`.](https://tensor4all.org/TensorCrossInterpolation.jl/dev/documentation/#TensorCrossInterpolation.crossinterpolate2-Union{Tuple{N},%20Tuple{ValueType},%20Tuple{Type{ValueType},%20Any,%20Union{NTuple{N,%20Int64},%20Vector{Int64}}},%20Tuple{Type{ValueType},%20Any,%20Union{NTuple{N,%20Int64},%20Vector{Int64}},%20Vector{Vector{Int64}}}}%20where%20{ValueType,%20N})

### My function / tensor is approximated well in one region, but not in another region. Why?
This is a typical symptom of the limitation described in ["What are the limitations of your Tensor Cross Interpolation (TCI) algorithms? When do they fail?"](#what-are-the-limitations-of-your-tensor-cross-interpolation-tci-algorithms-when-do-they-fail). The answer also describes how to solve this problem.

### When is it advisable to use the quantics representation?
In the quantics representation, the tensor indices correspond to bits of the function arguments instead of function arguments themselves. This leads to exponential resolution for linear number of bits. Generally, the quantics representation is useful if a function has few arguments, but high resolution is required in each argument, whilst the natural representation is more useful if a function has many arguments, but only coarse resolution is required.

Of course, this depends on whether the resulting tensor is factorizable in its indices. In the natural representation, factorizing the tensor means factorizing dependencies on different arguments of the function. In the quantics representation, factorizing the tensor means factorizing the different length scales of the function. Therefore, small bond dimension in one representation does *not* imply small bond dimension in the other.

### In the quantics representation, which index ordering is optimal?
This depends entirely on the function to be factorized. Generally, the bond dimension will grow with the distance along the chain that information has to be transferred. Different index orderings can lead to vastly different bond dimensions.
Some rules of thumb:
- If different arguments are highly entangled, but not different length scales, use the fused representation.
- If the function can be factorized both between different arguments and between different length scales, use the interleaved representation.
- If the function is more factorizable between different arguments than between length scales, use the serial representation, or similar.

This is another point where it is worth trying different factorizations on some test dataset that contains a typical case. For example, the paper below compares different index ordering schemes for solving the Vlasov-Poisson equations.

- Ye and Loureiro, *Quantum-inspired method for solving the Vlasov-Poisson equations*, [Phys. Rev. E 3, 106](https://doi.org/10.1103/PhysRevE.106.035208) (2022). ([arXiv:2205.11990](https://arxiv.org/abs/2205.11990))

### Can tensor trains generated with tensor4all tools be manipulated using other tensor libraries such as iTensor?
For the julia version of our libraries, we have implemented a small helper library called [`TCIITensorConversion.jl`](https://github.com/tensor4all/TCIITensorConversion.jl). It allows convenient bidirectional conversion between our TCI/TT objects and iTensor MPS/MPO objects through the constructor:
```julia
tt = TCI.TensorTrain(#=something=#)
mpo = ITensors.MPO(tt)
tt2 = TCI.TensorTrain{Float64, 4}(mpo)
```
If you wish to use other libraries and/or have already written code for conversion between the other library and tensor4all objects, feel free to contact us to set up a similar conversion library.



---