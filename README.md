# Ndarray for Matlab Users

This tutorial demonstrates how to implement common [Matlab](https://www.mathworks.com/products/matlab.html) patterns in [Rust](https://www.rust-lang.org/) using the [ndarray](https://docs.rs/ndarray) ecosystem of linear algebra tools.  It begins with simple matrix creation and progresses through basic statistics and parallel matrix operations.

## Quick Start

Clone this repository with:

```
git clone https://github.com/benkay86/matlab-ndarray-tutorial.git
```

Compile and run the `build_test` example to verify that everything works.

```
cd matlab-ndarray-tutorial
cargo run --bin build_test
```
```
... cargo/rustc output ...
Running `target/debug/build_test`
Testing BLAS backend... OK.
Testing LAPACK backend... OK.
```

You will need to [install Rust](https://www.rust-lang.org/tools/install), C, and fortran compilers (typically [gcc](https://gcc.gnu.org/) and [gfortran](https://gcc.gnu.org/wiki/GFortran)).  You will also need a [blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)/[lapack](https://en.wikipedia.org/wiki/LAPACK) backend such as [openblas](https://www.openblas.net/).  These dependencies are provided on most modern operating systems.  If you have difficulty getting openblas to work, open [Cargo.toml](Cargo.toml) and edit the last two lines to change `system` to `static` (this will significantly increase compilation time).

<hr />

## [Proceed with the Tutorial](rust/README.md)

<hr />

## Next Steps

This tutorial is divided into [workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html) by language.  We strongly recommend beginning with the [rust](./rust) workspace.

- [rust/](./rust)
	+ The best place to get started.
	+ Review pros and cons of ndarray vs Matlab.
	+ Examples of ndarray using pure Rust.
	+ Work through basic examples in order.
	+ Refer to advanced examples as needed.
- [python/](./python)
	+ Pass data seamlessly between Python's numpy and ndarray.
	+ Write numerical Python extensions in Rust.
- matlab/
	+ Need examples of how to write mex extensions in Rust.
	+ Your help writing this tutorial is welcome!

## See Also

- [Official ndarray examples](https://github.com/rust-ndarray/ndarray-examples)

## Getting Help
	
If you discover an error in this tutorial, have a question about an example, or would like to suggest an improvement, please open an issue on [Github](https://github.com/benkay86/matlab-ndarray-tutorial/issues) or e-mail the primary author [benjamin@benkay.net](mailto:benjamin@benkay.net).

### Upgrading ndarray 0.14 --> 0.15

If you have recently pulled this tutorial from ndarray version 0.14 to 0.15, you may get get bizarre trait errors like this one:

```
the trait FromPyObject<'_> is not implemented for PyReadonlyArray<'_, f64, Dim<IxDynImpl>>
```

If some of your code uses traits from ndarray 0.15 and other parts use the same trait from ndarray 0.14, Rust will see these as different, incompatible traits.  To fix this, update your `Cargo.lock` file to make sure all packages depend on the same version of ndarray.  Simply go to the root directory of this respository and run:

```
cargo update
```