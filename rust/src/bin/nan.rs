//! Find and filter NaN (not a number) values with ndarray.
//!
//! Note that this only works for floating-point numbers (i.e. f32, f64).
//! Integers and booleans do not have an NaN value.

extern crate blas_src;
use ndarray::{array, Array, Axis, Dim, Ix};

fn main() {
    // A vector containing some not-a-number values.
    let vec = array![1., 2., f64::NAN, 4., f64::NAN, 6.];
    println!("vec =\n{:?}\n", vec);

    // Does the vector contain an nans?
    //
    // ```matlab
    // vec = [1, 2, NaN, 4, NaN, 6]
    // any(isnan(vec))
    // ```
    println!(
        "Does vec contain nans? {:?}.\n",
        vec.iter().any(|x| x.is_nan())
    );

    // Find indices of all the not-nan values.  Count how many nans there are.
    // Create a copy without the nans.
    //
    // ```matlab
    // real_indices = find(~isnan(vec)) % indices of not-nan values
    // length(vec) - length(real_indices) % # of nans
    // vec_filtered = vec(real_indices) % vector with nans filtered out
    // ```
    let real_indices: Array<Ix, Dim<[Ix; 1]>> = vec
        .indexed_iter() // Create an indexed iterator.
        .filter(|(_, x)| !x.is_nan()) // Filter out nans.
        .map(|(i, _)| i) // Extract reamining indices.
        .collect();
    println!("real_indices =\n{:?}", real_indices);
    println!("There are {} nans in vec.", vec.len() - real_indices.len());
    // Note `unwrap()` on next line is guaranteed not to panic because
    // `real_indices` is contiguous in memory.
    let vec_filtered = vec.select(Axis(0), real_indices.as_slice().unwrap());
    println!("vec_filtered = \n{:?}\n", vec_filtered);

    // Create a copy without the nans using a consuming iterator.  This is more
    // memory efficient that what is possible in Matlab.
    // Note mapping of `&f64` to `f64` by `*x`.  This is inelegant but
    // necessary to collect our iterator over `&f64` to a vector of `f64`.
    let vec_filtered: Array<f64, Dim<[Ix; 1]>> = vec
        .into_iter() // could use iter() if we don't want to consume vec
        .filter(|x| !x.is_nan())
        .map(|x| x)
        .collect();
    println!("vec_filtered = \n{:?}", vec_filtered);
}
