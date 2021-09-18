//! Parallel for loops.
//!
//! First work through the examples in <for.rs>.

extern crate blas_src;
use ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use ndarray::{array, Array, Axis, Zip};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

fn main() {
    // Matlab uses a global worker pool to process parallel algorithms.  You can
    // manually specify the number of workers and start the pool, otherwise
    // Matlab will do this automatically for you the first time you call parfor.
    //
    // ```matlab
    // parpool(4) % start parallel pool with 4 workers
    // ```
    //
    // Rust and ndarray use the [Rayon](https://github.com/rayon-rs/rayon)
    // threadpool to implement parallel for loops.  Similar to Matlab, a global
    // threadpool is started automatically the first time you need it.  The size
    // of the threadpool defaults to the number of CPU cores.
    // Uncomment the following lines to start the global threadpool manually.
    // In this example we use one less than the number of CPUs.
    // rayon::ThreadPoolBuilder::default().num_threads(num_cpus::get() - 1).build_global().unwrap();

    // Sum all the elements of an array in parallel.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // sum = 0;
    // parforfor x=1:numel(mat)
    //    sum = sum + mat(x)
    // end
    // ```
    //
    // We can achieve roughly the same effect in Rust by getting a Rayon
    // parallel iterator over the array using `par_iter()`.  This works almost
    // the same as an orderinary iterator with `iter()` -- except in parallel!
    // See <https://docs.rs/rayon> for more on parallel iterators.
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("mat = \n{:?}", mat);
    // We need a mutex to synchronize access to the sum across threads.
    let sum = std::sync::Mutex::<f64>::new(0.);
    mat.par_iter().for_each(|el| {
        let mut sum = sum.lock().unwrap(); // acquire the mutex
        *sum = *sum + el; // increase the sum
    });
    let sum = sum.into_inner(); // get the sum out of the mutex
    println!("sum (mutex) = {:?}", sum);
    // For a reduction operation like summation, we can write a lock-free
    // version of this algorithm using the built-in `sum()` method on parallel
    // iterators.
    let sum: f64 = mat.par_iter().sum();
    println!("sum (lock free) = {:?}", sum);
    // More generally this is referred to as folding or reduction.  Conceptually
    // we start with an identity value (zero) and perform some associative
    // operation (i.e. addition) on elements of the array until we are left with
    // a single value (the sum). Note the mapping from &f64 to f64 so that we
    // can sum by value.
    let sum = mat.par_iter().map(|x| *x).reduce(
        || 0., // closure producing the identity value
        |x, y| x + y, // fold operation
    );
    println!("sum reduction (lock free) = {:?}\n", sum);

    // There is also a parallel version of the map.  Here was start with a
    // matrix of zeros and increment each element in parallel.
    let mut mat = Array::<f64, _>::zeros((2, 3));
    println!("zeros =\n{:?}", mat);
    mat.par_mapv_inplace(|el| el + 1.);
    println!("ones =\n{:?}\n", mat);

    // We can also iterate over rows/columns of an array in parallel.
    // Here we assign the column number to each column of an array.
    // The interface for parallel iteration is, unfortunately, inconsistent.
    // There is no `ArrayBase::par_axis_iter()`, rather, we get an ordinary
    // iterator with `axis_iter()` and then transform it into a parallel
    // iterator using `into_par_iter()`.  (Not all iterators support this,
    // but the axis iterator does.)
    let mut mat = Array::<f64, _>::zeros((2, 3));
    println!("zeros =\n{:?}", mat);
    mat.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(col_idx, mut col)| {
            col.fill(col_idx as f64);
        });
    println!("column indices =\n{:?}\n", mat);

    // We can also zip (iterate in lockstep) over two arrays in parallel.
    // For example, we can compute the sum of each column in an array and store
    // it in a second array.  This works the same way as in <./for.rs> except
    // that instead of `for_each()` we call `par_for_each()`.  Easy!
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("mat = \n{:?}", mat);
    let mut col_sums = Array::<f64, _>::zeros(3);
    Zip::from(&mut col_sums)
        .and(mat.axis_iter(Axis(1)))
        .par_for_each(|col_sum, col| *col_sum = col.sum());
    println!("col_sums = \n{:?}", col_sums);
}
