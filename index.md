![](tci.svg)

This website collects information from the tensor4all group which is working on tensor network methods based on tensor cross interpolation (TCI) and related tensor learning algorithms such as quantics/quantized tensor trains.

## Join our community!

<div style="background-color: #e8f4fd; border: 2px solid #2196F3; border-radius: 8px; padding: 20px; margin: 20px 0; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
  <h3 style="color: #1976D2; margin-top: 0; font-size: 1.4em;">🤝 Join our community!</h3>
  <p style="font-size: 1.1em; margin: 15px 0; color: #424242;">
    Connect with researchers working on tensor networks and related methods. Join our <a href="https://tensor4all.discourse.group" style="color: #1976D2; text-decoration: none; font-weight: bold;">Discourse forum</a> for discussions, or subscribe to our <a href="https://groups.google.com/g/tensor4all" style="color: #1976D2; text-decoration: none; font-weight: bold;">Google Groups mailing list</a> for meeting announcements.
  </p>
  <p style="font-size: 0.9em; margin: 10px 0; color: #666;">
    Don't have a Google account? Contact us at <a href="mailto:tensor4all-admin@googlegroups.com" style="color: #1976D2; text-decoration: none;">tensor4all-admin@googlegroups.com</a> to join the mailing list.
  </p>
  <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; margin-top: 15px;">
    <a href="https://tensor4all.discourse.group" style="display: inline-block; background-color: #2196F3; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; transition: background-color 0.3s;" onmouseover="this.style.backgroundColor='#1976D2'" onmouseout="this.style.backgroundColor='#2196F3'">
      Join Forum →
    </a>
    <a href="https://groups.google.com/g/tensor4all" style="display: inline-block; background-color: white; color: #2196F3; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; border: 2px solid #2196F3; transition: background-color 0.3s;" onmouseover="this.style.backgroundColor='#e8f4fd'" onmouseout="this.style.backgroundColor='white'">
      Join Mailing List →
    </a>
  </div>
</div>

## Literature

A pedagogical introduction to tensor network methods, which includes an overview of the existing literature and also new algorithms, can be found in:

> Yuriel Núñez Fernández, Marc K. Ritter, Matthieu Jeannin, Jheng-Wei Li, Thomas Kloss, Thibaud Louvet, Satoshi Terasaki, Olivier Parcollet, Jan von Delft, Hiroshi Shinaoka, Xavier Waintal, "Learning tensor networks with tensor cross interpolation: New algorithms and libraries", [SciPost Phys. 18, 104 (2025)](https://www.scipost.org/SciPostPhys.18.3.104) · published 20 March 2025, [arXiv:2407.02454](https://arxiv.org/abs/2407.02454).

Please check the [reference](reference.html) page for more information on TCI and quantics tensor trains.

## Code

We provide two software libraries that implement algorithms from the above manuscript for computing low-rank tensor representations.
The code focuses on recent applications of tensor networks to objects that do not necessarily involve many-body quantum mechanics. 
It also contain known and new variants of the tensor cross interpolation (TCI) algorithm for unfolding tensors into tensor trains.
One code is called Xfac (written in C++ with Python bindings), and a second implementation with similar functionality is based on Julia:

* [Xfac (C++ / Python)](https://xfac.readthedocs.io/en/latest/intro.html)
* [Julia](julia.html)

<a id="onlinemeeting"></a>
## Monthly online meeting
We have a monthly online meeting to discuss the development of new methods and applications. Zoom links will be provided through a mailing list (49 registered users as of November 4th, 2024). Please contact [us](<mailto:tensor4all-admin@googlegroups.com>) if you would like to join the mailing list.

Planned meetings:

* January 20th, 2025 (talk: Prof. Engin Danis, "Tensor-Train Finite Difference Method for Compressible Flows", 8:00 AM ET / 14:00 CET / 22:00 JST) - [Details](https://tensor4all.discourse.group/t/online-by-prof-engin-danis-tensor-train-finite-difference-method-for-compressible-flows-january-20th/21)

Previous meetings:

* July 8th, 2025 (talk: Leonhard Hoelscher)
* June 6th, 2025 (talk: Maksymilian Środa)
* May 6th, 2025
* March 1st, 2025
* February 18th, 2025
* January 13th, 2025 (talk: Yuehaw Khoo)
* November 26th, 2024
* October 29th, 2024
* September 30th, 2024
* August 26th, 2024
* July 1st, 2024
* June 3rd, 2024
* April 22th, 2024
* March 25th, 2024
* February 22nd, 2024

## Workshops

We organize [workshops](workshop/index.html) to discuss the development of new methods.

## Lectures
We have a [lectures](lecture/index.html) page to share lectures on TCI and related tensor learning algorithms.

## People

Check the [about](about.html) page to see who is involved in the tensor4all collaboration.

## Frequently Asked Questions
We have a [FAQ](faq.html) page to answer common questions.
