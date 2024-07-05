![](tci.svg)

This website collects information from the tensor4all group which is working on tensor network methods.


## Literature

A pedagogical introduction to tensor network methods, which includes an overview of the existing literature and also new algorithms, can be found in:

> Yuriel Núñez Fernández, Marc K. Ritter, Matthieu Jeannin, Jheng-Wei Li, Thomas Kloss, Thibaud Louvet, Satoshi Terasaki, Olivier Parcollet, Jan von Delft, Hiroshi Shinaoka, and Xavier Waintal, 
> *Learning low-rank tensor train representations: new algorithms and libraries*, [arXiv:2407.02454](https://arxiv.org/abs/2407.02454).

Please check the [reference](reference.html) page for more information on TCI and quantics tensor trains.

## Code

We provide two software libraries that implement algorithms from the above manuscript for computing low-rank tensor representations.
The code focuses on recent applications of tensor networks to objects that do not necessarily involve many-body quantum mechanics. 
It also contain known and new variants of the tensor cross interpolation (TCI) algorithm for unfolding tensors into tensor trains.
One code is called Xfac (written in C++ with Python bindings), and a second implementation with similar functionality is based on Julia:

* [Xfac (C++ / Python)](https://xfac.readthedocs.io/en/latest/intro.html)
* [Julia code](julia.html)

## Workshops

We organize [workshops](workshop/index.html) to discuss the development of new methods.

## People

Check the [about](about.html) page to see who is involved in the tensor4all collaboration.
