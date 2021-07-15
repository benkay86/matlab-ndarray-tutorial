//! Example of nested parallel computations.
//!
//! This is an example where Rust really outshines Matlab.  You can have code
//! like this in Matlab:
//!
//! ```matlab
//! x = zeros(100, 200);
//! parfor i=1:size(x,1)
//!     parfor j=1:size(x,2)
//!          % Do something computationally expensive.
//!          x(i,j) = f(i,j);
//!     end
//! end
//! ```
//!
//! But Matlab will only execute the outer parfor loop in parallel.  The inner
//! parfor loop will *not* be run in parallel!  Sometimes you can work around
//! this by "flattening" the inner loop into the outer loop, but this affects
//! code readability and may not always be practical.
//!
//! This is an "advanced" example of parallelism in Rust.  First work through
//! <parfor.rs> example.

extern crate blas_src;
use ndarray::{Array, Axis, array, Zip};
use matlab_ndarray_tutorial::lls;
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand_distr::StandardNormal;

fn main() {
    // Building on <parfor.rs> we will parallelize a simple permutation test.
    //
    // To create an opportunity for nested parallelism, instead of performing
    // a single permutation test in parallel, we will perform multiple tests.

    // Generate matrix of observations y and predictors x with known regression
    // coefficient b.
    // See <parfor.rs> for explanations.
    let n_test = 32; // number of permutation tests to perform
    let n_obs = 32; // number of observations
    let n_perm = 1024; // number of permutations to perform for each test
    let x = Array::<f64, _>::random((n_obs, n_test), StandardNormal);
    // Try setting b to zero and see what happens to the p-value.
    let b = array![1.];
    println!("true beta: {:?}", b[0]);
    let y = &x * &b + &Array::<f64, _>::random((n_obs, n_test), StandardNormal);

    // Pre-allocate array of p-values for each permutation test.
    let mut p_values = Array::<f64, _>::zeros(n_test);

    // In <parfor.rs> we ran all parallel computations on a single, global
    // thread pool.  This time we want to think about what order to run our
    // computations in very carefully.
    // Each outer test requires considerable memory in which to store its
    // SolvedModel structure.  If we had infinite memory we could compute in any
    // order and it would take the same amount of time.  Since we only have
    // finite memory, it will be more efficient to prioritize computing the
    // inner permutations, and to only work on 1 or 2 of the outer tests at a
    // time.
    // We can sort this out dynamically by querying the number of available CPU
    // cores at runtime and then executing our tasks on different thread pools.
    let (pool_test, pool_perm) = {
        let ncpus = num_cpus::get();
        let ncpus_test = if ncpus < 5 { 1 } else { 2 };
        let ncpus_perm = std::cmp::max(1, ncpus - ncpus_test);
        // Outer thread pool for running each test.
        // Use a limited number of cpus.
        let pool_test = rayon::ThreadPoolBuilder::default().num_threads(ncpus_test).build().unwrap();
        // Inner thread pool for running the permutations.
        // Use the remaining cpus.
        let pool_perm = rayon::ThreadPoolBuilder::default().num_threads(ncpus_perm).build().unwrap();
        (pool_test, pool_perm)
    };

    // Run parallel operations in the outer, test pool.
    pool_test.install(|| {
        Zip::from(&mut p_values).and(x.gencolumns()).and(y.gencolumns()).par_apply(|p, x, y| {
            // Make design matrix x 2-dimensional.
            let x = x.insert_axis(Axis(1));
            // Pre-solve the linear system.
            let solved_model = lls::pre_solve(x).unwrap();
            // Compute t-values for the un-permuted model.
            let t0 = solved_model.tvalues_with(&y);
            // Pre-allocate matrix of permuted t-values.
            let mut t = Array::<f64, _>::zeros(n_perm);

            // Run parallel operations in the inner, permutation pool.
            pool_perm.install(|| {
                t.par_mapv_inplace(|_| {
                    // Sample permuted observations.
                    let yp = y.sample_axis(Axis(0), y.len(), SamplingStrategy::WithoutReplacement);
                    // Compute t-values with permuted observations and store
                    // result in the variable `t`.
                    solved_model.tvalues_with(&yp)[0]
                });
            });

            // Two-tailed test.
            // Sum the number of simulations in which |t0| > |t|.
            let t0 = t0[0].abs();
            let sum = t.fold(0, |sum, t| {
                if t0 > t.abs() {
                    sum + 1
                }
                else {
                    sum
                }
            });
            // Compute the p-valu and store it in the variable `p_values`.
            *p = 1. - (sum as f64) / (n_perm as f64);
        });
    });

    // Examine the 32 p-values we computed in parallel.
    println!("p-values: {:?}", p_values);
}
