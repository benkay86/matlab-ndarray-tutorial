# Ndarray and Python

[Python](https://www.python.org/) is a very popular programming language for "big data," machine learning, and numerical analysis.  [NumPy](https://numpy.org/) is the _de facto_ matrix library for Python, and the inspiration for Rust's ndarray.  This portion of the tutorial demonstrates how to efficiently convert between Python's numpy arrays and Rusts's ndarray arrays.

You should familiarize yourself with how to use [ndarray in Rust](../rust/README.md) before working through this part of the tutorial.

## Introduction

Python is a dynamically-typed, interpreted language with a [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) much like Matlab.  Many programmers find it easier to use than compiled languages like C, C++, and Rust... or at least easier to get started with.  As with Matlab, this ease of use comes with a performance tradeoff.

Python's numpy matrix algebra library is actually a Python extension module written in C that calls into the highly-optimized 
[blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [lapack](https://en.wikipedia.org/wiki/LAPACK) libraries.  This allows one to achieve quite good performance in Python for vectorized matrix operations like dot product, decomposition, etc, since numpy calls into blas/lapack to do the actual operation, similar to Matlab.  As with Matlab, Python runs into performance issues composing more and more complex matrix operations together, especially when memory contraints and parallelism are involed. This is where Rust and ndarray can help.

Fortunately, the memory layout of numpy's arrays is almost identical to the memory layout of Rust's ndarray arrays.  This makes it easy to implement low- or zero-cost conversions between array types using Rust's [PyO3](https://pyo3.rs) crate for generating Python bindings and [numpy](https://github.com/PyO3/rust-numpy) crate for interfacing with numpy arrays.

## Building an Extension Module

Numpy is a Python extension module written in C to make numerical operations in Python more efficient.  We can build on numpy and ndarray by writing our own Python extension in Rust to make more complex numerical operations run faster!

See the example in [src/lib.rs](./src/lib.rs).


## Getting Help

Refer to the [pyo3 guide](https://pyo3.rs) and [api documentation](https://docs.rs/pyo3).  Also refer to the [Github page for Rusts's numpy](https://github.com/PyO3/rust-numpy) and its [api documentation](https://docs.rs/numpy).