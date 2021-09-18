//! Time how long it takes to do something to measure performance.
//!
//! Be sure to compile in release mode to enable compiler optimizations.
//! cargo run --release --bin 20_benchmark

extern crate blas_src;
use ndarray::{Array, Axis};
use ndarray_linalg::{LeastSquaresResult, LeastSquaresSvdInto};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use rust_examples::pseudoinverse::PseudoInverse;

fn main() {
    // In <19_least_squares.rs> we introduced several different ways of
    // computing the pseudoinverse of a matrix.
    //
    // Let's benchmark the difference by computing the pseudoinverse of a
    // 512 x 512 random matrix 64 times with each algorithm.
    //
    // ```matlab
    // x = normrnd(0, 1, 512, 512, 64);
    // tic
    // for j=1:64
    //     pinv(x(:,:,j));
    // end
    // msec = toc * 1000 / 64
    // ```
    let x = Array::<f64, _>::random((512, 512, 64), StandardNormal);
    println!("Time to compute pseudoinverse of 512 x 512 matrix averaged over 64 iterations:");
    let start = std::time::Instant::now();
    x.axis_iter(Axis(2)).for_each(|x| {
        let LeastSquaresResult {
            solution: _x_pinv, ..
        } = x
            .to_owned()
            .least_squares_into(Array::<f64, _>::eye(x.shape()[0]))
            .unwrap();
    });
    let elapsed = start.elapsed();
    println!(
        "dgesvd (Matlab's pinv): {:?} ms",
        elapsed.as_millis() / x.shape()[2] as u128
    );
    let start = std::time::Instant::now();
    x.axis_iter(Axis(2)).for_each(|x| {
        x.pinv_svd().unwrap();
    });
    let elapsed = start.elapsed();
    println!(
        "dgesdd (SVD divide-and-conquer): {:?} ms",
        elapsed.as_millis() / x.shape()[2] as u128
    );
    let start = std::time::Instant::now();
    x.axis_iter(Axis(2)).for_each(|x| {
        x.pinv().unwrap();
    });
    let elapsed = start.elapsed();
    println!(
        "QR decomposition: {:?} ms",
        elapsed.as_millis() / x.shape()[2] as u128
    );
}
