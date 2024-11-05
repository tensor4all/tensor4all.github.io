# Frequently Asked Questions

<!-- - [Q1: What are differences from other tensor network libraries supporting machine learning?](#q1) -->

1. placeholder
{:toc}

---

### How do I get started?
We have an extensive set of online tutorials that teach you how to use our libraries, for both the Julia and C++/Python version of our library.

- **Julia**: Tutorials are available [online](https://tensor4all.org/juliatutorials/index.html) and can be run in your web browser. If you wish to run notebooks locally on your computer, install julia using the [installation instructions](https://tensor4all.org/juliatutorials/index.html#install-julia), then copy-paste the instructions at the top of each online notebook to download and run that specific notebook.
- **C++/Python**: Install the library using the [installation instructions](https://github.com/tensor4all/xfac/blob/main/README.md#installation), then follow the [tutorial pages](https://xfac.readthedocs.io/en/latest/tutorial-python/intro-tutorial.html).

These tutorials should enable you to implement the most common use cases. In many cases, it is sufficient to copy-paste and slightly modify some tutorial code. For more advanced use cases, you may wish to consult the documentation:

- **Julia**: [Documentation](https://tensor4all.org/julia.html)
- **C++/Python**: [Documentation](https://xfac.readthedocs.io/en/latest/intro.html)

### Are all tensors/functions compressible?
No: random noise, for example, is not compressible. The tensor train format exploits *structure* for its compression, and random noise lacks any structure that can be used for this purpose.
More generally, the tensor train is a partial factorization of a tensor. If the tensor is a discretized function of many arguments, the tensor train is efficient if the function is close to being factorizable in its arguments. For a function in quantics representation, the tensor is factorizable if the function is factorizable in its length scales. This includes all polynomials and linear combinations of exponential functions ([Lindsey2024](https://arxiv.org/abs/2311.12554)). If multiple factorizable structures are added or multiplied in the function, the function is typically also compressible.

Beyond the arguments above, our understanding of the compressibility of functions in tensor train format is still evolving and the subject of current research.
If you are interested in a specific use case, is often easier to just try out whether some function of interest is compressible using an existing dataset. Instructions for this can be found in ["How can I test whether my data is TCI compressible?"](#how-can-i-test-whether-my-data-is-tci-compressible).

### How can I test whether my data is TCI compressible?
Using our libraries, it is easy to try whether some function or dataset of interest is compressible. [The tutorial on compressing existing data](https://tensor4all.org/T4APlutoExamples/pluto_notebooks/compress.html) demonstrates how to do this for TCI and quantics TCI. Replace the tutorial dataset with your own and set the tolerance to whatever precision you require. Note that the tolerance has to be set well above the noise level in your dataset to get an accurate idea about the compressibility of your dataset. Otherwise, TCI will try to compress your random noise, which may lead to an extreme increase in bond dimension.

### What is the difference between SVD-based and TCI compression?
In tensor network algorithms, tensor trains or MPS are usually constructed using the [*singular value decomposition*](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) and truncation by singular values. It has been shown that this compression is optimal in the L2-norm of the resulting error. However, SVD can only be performed with knowledge of all components of the original tensor. In contrast, TCI is able to construct a tensor train using only a subset of components, such that the full tensor never has to be constructed explicitly. Thus, it is possible to construct a tensor train for tensors that would never fit into memory!

### How does Tensor Cross Interpolation (TCI) work?
Very briefly, TCI is a sweeping optimization algorithm. It starts from a tensor train with very small bond dimension, and then optimizes it using a series of local updates.

For local updates, it relies on the *cross interpolation* (CI) factorization instead of SVD. CI extracts a subset of rows and columns of some matrix to be factorized, and arranges the rows and columns in the following structure:

![](mci.svg)

CI has the important advantage that all components of the resulting factorization are components of the original matrix. This means that the tensor generalization, TCI, can be constructed and optimized using small slices of the original tensor, and an explicit representation of the full tensor is not required.

TCI is explained in detail in [NunezFernandez2024](https://arxiv.org/abs/2407.02454).

### What is the relation between Tensor Cross Interpolation (TCI) and machine learning?
Tensor Cross Interpolation (TCI) can be seen as a machine learning algorithm, in that it samples a small subset of a large dataset (the tensor to be approximated) to learn a representation (the tensor train) that should generalize to the whole dataset. It is an active learning algorithm in that the sampled subset is not given externally, but dynamically chosen by the algorithm based on what it has learned so far.

### What are differences from other tensor network libraries that support machine learning?

A critical difference is that our library is based on Tensor Cross Interpolation (TCI). TCI allows us to compute tensor networks with a low-rank representation by exploring a tiny subset of the full tensor. This opens up the possibility of applying tensor networks to unconventional applications such as high-dimensional integration, i.e., replacing Monte Carlo integration.

---