# Ndarray and Rust for Matlab Users

This tutorial demonstrates how to implement common [Matlab](https://www.mathworks.com/products/matlab.html) patterns in [Rust](https://www.rust-lang.org/) using the [ndarray](https://docs.rs/ndarray) ecosystem of linear algebra tools.  It begins with simple matrix creation and progresses through basic statistics and parallel matrix operations.

## Quick Start

Clone this repository with:

```
git clone https://github.com/benkay86/ndarray-matlab-tutorial.git
```

Compile and run the `build_test` example to verify that everything works.

```
cd ndarray-matlab-tutorial
cargo run --bin build_test
```
```
... cargo/rustc output ...
Running `target/debug/build_test`
Testing BLAS backend... OK.
Testing LAPACK backend... OK.
```

You will need to [install Rust](https://www.rust-lang.org/tools/install), C, and fortran compilers (typically [gcc](https://gcc.gnu.org/) and [gfortran](https://gcc.gnu.org/wiki/GFortran)).  You will also need a [blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)/[lapack](https://en.wikipedia.org/wiki/LAPACK) backend such as [openblas](https://www.openblas.net/).  These dependencies are provided on most modern operating systems.  If you have difficulty getting openblas to work, open [Cargo.toml](Cargo.toml) and edit the last two lines to change `system` to `static` (this will significantly increase compilation time).

## Introduction

In brief, writing small programs in Rust takes more time and effort than writing them in Matlab.  As the programs get larger and more sophisticated, writing and maintaining them in Rust often becomes easier than in Matlab.  Programs written in Rust will generally run faster and use less memory than programs written in Matlab.

### What is Rust/Ndarray?

Matlab is a domain-specific computer programming language for numerical computation.  Rust is a general-purpose programming language suitable for anything from [writing an operating system kernel](https://security.googleblog.com/2021/04/rust-in-linux-kernel.html) to [writing a website](https://github.com/seed-rs/seed).

[Ndarray](https://docs.rs/ndarray) is an ecosystem of libraries for numerical computation in Rust.  The name is short for "N-Dimensional Array."  Both Matlab and ndarray use the heavily-optimized [blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [lapack](https://en.wikipedia.org/wiki/LAPACK) projects for numerical computation on the backend, and will therefore have similar performance for basic numerical operations (e.g. matrix multiplication).

[Rust's ndarray](https://github.com/rust-ndarray/ndarray) is closely related to another general-purpose programming language library: [Python's numpy](https://numpy.org/).  If you are already familiar with numpy then you can skip this tutorial and read this [documentation on ndarray for numpy users](https://docs.rs/ndarray/0.15.3/ndarray/doc/ndarray_for_numpy_users/index.html) instead.

### How are Matlab and ndarray different?

- Matlab is a proprietary software for numerical computation.  Ndarray is open source.
- In addition to basic linear algebra, Matlab offers all kinds of paid add-on packages for digital signal processing, statistics, machine learning, etc.  Ndarray lacks many of these features.
- The Matlab environment has a graphical user interface with a read-evaluate-print loop (REPL).  Ndarray code must be compiled.
- The Matlab language is designed with a focus on linear algrebra, and not much else.  The Rust language is general-purpose, therefore ndarray's syntax is necessarily more verbose.
- Matlab is an excellent environment for rapidly prototyping numerical programs.  Rust has a steeper learning curve, but it excels at memory-intensive operations, parallelism, and integration into production environments.

### Why Rust/ndarray?

- Rust's memory model enforces [strict aliasing](https://cvw.cac.cornell.edu/vector/coding_aliasing) for Fortran-like performance in vectorized loops.
- Rust makes it easy to write highly-performant parallel (i.e. multi-threaded) code, even nested parallel-for loops.
- Rust and ndarray give you precise control over memory allocation for manipulating large data sets.
- The Rust compiler makes you more productive by safeguarding against common errors _before_ you run your code.
- Rust supports integration with many other programming languages including [C](https://doc.rust-lang.org/nomicon/ffi.html), [C++](https://github.com/dtolnay/cxx), and [Python](https://github.com/PyO3/pyo3).

### Why _not_ Rust/ndarray?

- Rust is a relatively young programming language circa 2015.  Some [core](https://github.com/rust-lang/rust/issues/44265) [features](https://github.com/rust-lang/rust/issues/31844) are not even implemented yet!
- Ndarray does not have feature parity with [numpy](https://numpy.org/) or Matlab yet.
- Rust does not have the popularity/mindshare of Python, C++, or R.
- Rust code will always be more daunting to newcomers than tweaking a Python or Matlab script.
- Matlab's IDE is a more productive environment for rapid prototyping.

## How To Use This Tutorial

This tutorial does not cover the basics of programming in Rust.  The [Rust Book](https://doc.rust-lang.org/book/) is a great place to learn Rust.

Work through the examples in [`src/bin`](./src/bin).  As you work, make frequent reference to the [ndarray](https://docs.rs/ndarray) and [ndarray-linalg](https://docs.rs/ndarray-linalg) documentation so that you will become skilled at using these information sources.  You are encouraged to edit, tweak, and experiment on the examples.

To run each example:

```
cargo run --bin name_of_example
```

Consider working the examples beginning in the following order.  Once you have mastered these core examples you can proceed with the remaining examples in any order you wish.

1. [Matrix Creation](./src/bin/matrix_creation.rs)
2. [Clear](./src/bin/clear.rs)ing Memory
3. [Index](./src/bin/index.rs)ing Matrices
4. Checking [Size](./src/bin/size.rs)
5. [Slicing](./src/bin/slicing.rs) Matrices
6. Matrix [Transpose](./src/bin/transpose.rs) (and broadcast)
7. [Assign](./src/bin/assign.rs) to Matrices
8. [Concatenate](./src/bin/concatenate.rs) Matrices
9. Matrix [Math](./src/bin/math.rs)
10. [For](./src/bin/for.rs) Loops
11. [Fold](./src/bin/fold.rs)ing

## Getting Help

The reader is assumed to be familiar with [Matlab](https://www.mathworks.com/products/matlab.html) (or [Octave](https://www.gnu.org/software/octave/index)) and possess basic knowledge of [Rust](https://www.rust-lang.org/).  For general questions about these languages, refer to the reference documentation ([Matlab](https://www.mathworks.com/help/matlab/), [Rust](https://docs.rs/std/)) or post in the general user forums ([Matlab](https://www.mathworks.com/matlabcentral/), [Rust](https://users.rust-lang.org/)).

For help with ndarray refer to the [ndarray](https://docs.rs/ndarray) and [ndarray-linalg](https://docs.rs/ndarray-linalg) documentation (especially the [ArrayBase](https://docs.rs/ndarray/0.15.1/ndarray/struct.ArrayBase.html) struct).  Rust's ndarray is very similar to [Python's](https://www.python.org/) [numpy](https://numpy.org/), so you may also find relevant solutions involving numpy.  If you think you have found a bug, or if something seems very unclear, open an issue on the Github page for [ndarray](https://github.com/rust-ndarray/ndarray) or [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg).  If you have issues with compiling or linking, try reading the comments in [Cargo.toml](Cargo.toml) for an explanation of the various dependencies.

If you discover an error in this tutorial, have a question about an example, or would like to suggest an improvement, please open an issue on [Github](https://github.com/benkay86/ndarray-matlab-tutorial/issues) or e-mail the primary author [benjamin@benkay.net](mailto:benjamin@benkay.net).

## Concepts

### Representing Data in Memory

Matlab represents data in 2-dimensional arrays, or matrices.  For example:

```
>> mat2d = [1,2,3; 4,5,6; 7,8,9]

mat2d =

     1     2     3
     4     5     6
     7     8     9
```

The way in which Matlab and Rust's ndarray store data is actually very similar since both are designed to use a [blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) linear algebra library behind-the-scenes.

In the case of the 2-dimensional matrix above, we humans conceptualize the data as existing along two dimensions, or axes.  Computers, however, do not innately understand this concept.  To a computer the data always exist as one contiguous block of memory, regardless of how many dimensions the matrix has!  To a computer, the data look like this ([diagram taken from Stack Overflow](https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman)):

![diagram of matrix stride](stride.png)

The second row and third column of the data are at coordinates `[1,2]` in ndarray (which counts up from zero) or `(2,3)` in Matlab (which counts from one).  To access the data at that location, the computer will calculate an offset (in [bytes](https://en.wikipedia.org/wiki/Byte)) from the beginning of the array using the _stride_ of each dimension and the _size_ of the data.  If the data are [double-precision float](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) then the size is 8.  In the example above the rows are stored contiguously, so the stride walking across a row from column to column (red arrow) is 1.  The columns are stored one after another, so the stride walking down a column from row to row (green arrow) is 3, the number of rows.  The offset for the second row, third column `[1,2]` is then `8 * (1*3 + 2*1) = 40` bytes.

[Further reading.](https://ajcr.net/stride-guide-part-1/)