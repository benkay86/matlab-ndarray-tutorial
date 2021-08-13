//! Example of using a parallel for loop to perform a permutation test.
//!
//! First see <parfor_permutation.rs>.  In this more complex example, we eschew
//! the convenience of `par_mapv_inplace()` and spawn tasks on a threadpool
//! manually.  This is roughly analagous to Matlab's spmd primitives.

extern crate blas_src;
use matlab_ndarray_tutorial::lls;
use ndarray::{array, Array, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand_distr::StandardNormal;

fn main() {
    // Generate matrix of observations y and predictors x with known regression
    // coefficient b.
    // See <parfor.rs> for explanations.
    let n_obs = 32;
    let x = Array::linspace(0., 3., n_obs)
        .into_shape((n_obs, 1))
        .unwrap();
    // Try setting b to zero and see what happens to the p-value.
    let b = array![1.];
    println!("true beta: {:?}", b[0]);
    let y = x.dot(&b) + &Array::<f64, _>::random(n_obs, StandardNormal);

    // Pre-solve the linear system.
    let solved_model = lls::pre_solve(x).unwrap();

    // Compute t-values for the un-permuted model.
    let t0 = solved_model.tvalues_with(&y);
    println!("t-value: {:?}", t0[0]);

    // Pre-allocate matrix of permuted t-values.
    let n_perm = 1024; // number of permutations
    let mut t = Array::<f64, _>::zeros(n_perm);

    // The global Rayon thread pool is automatically initialized for us.
    // Uncomment the following line if you want to manually initialize the
    // global thread pool with 4 threads.
    // rayon::ThreadPoolBuilder::default().num_threads(4).build_global().unwrap();

    // Closure to perform each permutation.
    let permute = |el: &mut f64| {
        // Sample permuted observations.
        let yp = y.sample_axis(Axis(0), y.len(), SamplingStrategy::WithoutReplacement);
        // Compute t-values with permuted observations and store result
        // the element el of t we are working on.
        *el = solved_model.tvalues_with(&yp)[0];
    };

    // Inline helper function to recursively perform permutations in parallel.
    // (Needed because Rust does not support recursive closures.)
    fn par_permute<'a, I, F>(permute: &F, mut iter_mut: I)
    where
        I: Iterator<Item = &'a mut f64> + Send + Sync,
        F: Fn(&mut f64) -> () + Send + Sync,
    {
        // Recurse until we exhaust the elements in t.
        if let Some(el) = iter_mut.next() {
            // Rayon's join() will attempt to execute the first closure on the
            // first idle thread in the threadpool.  If there is a second idle
            // thread then it will execute the second closure in parallel.
            // Otherwise the first thread will execute the second closure after
            // it finishes executing the first closure.
            rayon::join(
                || {
                    // Perform this permutation on a worker thread.
                    permute(el);
                },
                || {
                    // Perform the remaining permutations recursively, on the same
                    // or possibly other worker threads.
                    par_permute(permute, iter_mut);
                },
            );
        }
    }

    // Execute the helper function to perform the permutations in parallel.
    // The main thread will block until all permutations are done.
    par_permute(&permute, t.iter_mut());

    // Two-tailed test.
    // Sum the number of simulations in which |t0| > |t|.
    let t0 = t0[0].abs();
    let sum = t.fold(0, |sum, t| if t0 > t.abs() { sum + 1 } else { sum });
    // Compute the p-value.
    let p = 1. - (sum as f64) / (n_perm as f64);
    println!("p-value: {:?}", p);
}
