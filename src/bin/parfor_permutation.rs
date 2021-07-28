//! Example of using a parallel iterator to perform a permutation test.
//!
//! See also <parfor.rs> for simpler examples of parallel iteration.
//!
//! The aim of this example is to show how parfor works in Rust.
//! See <least_squares.rs> for examples of how to solve least squares systems.
//! To keep this example readable, the linear least squares functionality is
//! abstracted away into <../lls.rs>.
//!
//! See also <nested_parfor.rs> for an example of nested parallel computation,
//! and <spmd.rs> for the "harder" version of this same example showing how
//! the magic of `par_mapv_inplace()` works.

extern crate blas_src;
use matlab_ndarray_tutorial::lls;
use ndarray::{array, Array, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand_distr::StandardNormal;

fn main() {
    // Generate a very simple linear least squares system with one predictor.
    //
    // ```matlab
    // % Simulate 32 observations with random error.
    // n_obs = 32;
    // x(:,2) = linspace(0, 3, n);
    // b = [1]
    // y = x * b + normrnd(0, 1, n_obs, 1);
    // ```
    let n_obs = 32;
    let x = Array::linspace(0., 3., n_obs)
        .into_shape((n_obs, 1))
        .unwrap();
    // Try setting b to zero and see what happens to the p-value.
    let b = array![1.];
    println!("true beta: {:?}", b[0]);
    let y = x.dot(&b) + &Array::<f64, _>::random(n_obs, StandardNormal);

    // Pre-solve the linear system.
    // ```matlab
    // % pseudocode using functionality from lls module
    // solved_model = presolve(x);
    // ```
    let solved_model = lls::pre_solve(x).unwrap();

    // Compute t-values for the un-permuted model.
    //
    // ```matlab
    // % pseudocode using functionality from lls module
    // t0 = tvalues(solved_model, y);
    // ```
    let t0 = solved_model.tvalues_with(&y);
    println!("t-value: {:?}", t0[0]);

    // Manually specify number of threads if desired, defaults to # of cpus.
    // rayon::ThreadPoolBuilder::default().num_threads(4).build_global().unwrap();

    // Pre-allocate matrix of permuted t-values.
    //
    // ```matlab
    // n_perm = 1024;
    // t = zeros(1, n_perm);
    // ```
    let n_perm = 1024; // number of permutations
    let mut t = Array::<f64, _>::zeros(n_perm);

    // Perform the permutations in parallel.
    //
    // ```matlab
    // for i=1:nperm
    //     yp = y(randperm(n_obs));
    //     t(i) = tvalues(solved_model, yp)
    // end
    // ```
    //
    // In Rust, rather than use parfor (which does exist), we will instead use
    // the more efficient maping interface.  The following is exactly the same
    // as calling `mapv_inplace()` (see <../for.rs>), but uses multiple threads.
    t.par_mapv_inplace(|_| {
        // Sample permuted observations.
        let yp = y.sample_axis(Axis(0), y.len(), SamplingStrategy::WithoutReplacement);
        // Compute t-values with permuted observations and store result in
        // the variable `t`.
        solved_model.tvalues_with(&yp)[0]
    });

    // Two-tailed test.
    // Sum the number of simulations in which |t0| > |t|.
    //
    // ```matlab
    // t0 = abs(t0);
    // sum = 0;
    // for i=1:n_perm
    //     if (t0 > abs(t))
    //         sum = sum + 1
    //     end
    // end
    // p = 1 - sum / n_perm
    // ```
    let t0 = t0[0].abs();
    let sum = t.fold(0, |sum, t| if t0 > t.abs() { sum + 1 } else { sum });
    // Compute the p-value.
    let p = 1. - (sum as f64) / (n_perm as f64);
    println!("p-value: {:?}", p);
}
