//! Test program to make sure the requisite build dependencies are installed.
//! Run the test with `cargo run --bin 01_build_test`.
//! See README.md and the comments in Cargo.toml for help.

// Tell the compiler this binary requires the link dependencies from
// openblas-src (i.e. to link to blas and lapack backends).
extern crate blas_src;

// Bring needed traits and macros into scope.
use ndarray::array;
use ndarray_linalg::Solve;

fn main() {
    // Perform matrix multiplication to make sure ndarray successfully linked
    // against a BLAS backend.  In Matlab:
    //
    // ```matlab
    // A = [1,2,3; 4,5,6];
    // B = [1,2; 3,4; 5,6];
    // X = A * B;
    // ```
    print!("Testing BLAS backend...");
    let a = array![[1., 2., 3.], [4., 5., 6.]];
    let b = array![[1., 2.], [3., 4.], [5., 6.]];
    let x = a.dot(&b);
    assert!(x.abs_diff_eq(&array![[22., 28.], [49., 64.]], 1e-9));
    println!(" OK.");

    // Use an LAPACK solver to make sure ndarray-linalg successfully linked
    // against an LAPACK backend.  This is vaguely similar to Matlab:
    //
    // ```matlab
    // A = [3,2,-1; 2,-2,4; -2,1,-2];
    // b = [1; -2; 0];
    // x = A \ b;
    // ```
    print!("Testing LAPACK backend...");
    let a = array![[3., 2., -1.], [2., -2., 4.], [-2., 1., -2.]];
    let b = array![1., -2., 0.];
    let x = a.solve_into(b).unwrap();
    assert!(x.abs_diff_eq(&array![1., -2., -2.], 1e-9));
    println!(" OK.");
}
