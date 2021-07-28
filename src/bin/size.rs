//! Checking the size of a matrix.

extern crate blas_src;
use ndarray::{array, Axis};

fn main() {
    // Generate a 2d matrix to check the size of.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // sz = size(mat)
    // nrows = size(mat,1)
    // ncols = size(mat,2)
    // ```
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("The matrix:\n{:?}", mat);
    // We can get the shape (analagous to size) as a slice.
    let sz = mat.shape();
    println!("Shape of matrix: {:?}", sz);
    println!(
        "{:?}-dimensional matrix has {:?} rows and {:?} columns.",
        sz.len(),
        sz[0],
        sz[1]
    );
    // Or we can just get the number of dimenions.
    let ndim = mat.ndim();
    println!("Matrix has {:?} dimensions.", ndim);
    // Or the length of just one axis.  The first axis (rows) is `Axis(0)`.
    let nrows = mat.len_of(Axis(0));
    println!("Matrix has {:?} rows.", nrows);
    // Columns are `Axis(1)`, and so on.
    let ncols = mat.len_of(Axis(1));
    println!("Matrix has {:?} columns.", ncols);
    // We can also get the overall length, i.e. total number of elements.
    //
    // ```matlab
    // nels = numel(mat)
    // ```
    let nels = mat.len();
    println!("Matrix has {:?} total elements.\n", nels);

    // Vectors are 1-dimensional arrays.  Matlab doesn't really have 1d ararys.
    // A Matlab vector has 2 dimensions, but one of them is collapsed.  This
    // gives us row vectors and column vectors.  We can do this in Rust/ndarray
    // too.
    //
    // ```matlab
    // v = [1,2,3] % row vector
    // size(v) % 1 3
    // v = [1;2;3] % column vector
    // size(v) % 3 1
    // ```
    let v = array![[1., 2., 3.]]; // technically a 2d array
    println!("Row matrix dimensions: {:?}", v.shape());
    let v = array![[1.], [2.], [3.]]; // also a 2d array
    println!("Column matrix dimensions: {:?}", v.shape());
    // However, unlike Matlab, ndarray often uses single-dimension arrays to
    // represent vectors.  For example, `solve_into()` in <build_test.rs> and
    // `diag()` in <matrix_creation.rs> take a 1d array.
    let v = array![1., 2., 3.]; // truly a 1d array
    println!(
        "1d array has {:?} dimension and size: {:?}",
        v.ndim(),
        v.shape()
    );
}
