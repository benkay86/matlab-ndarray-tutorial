//! Taking random samples of vectors/matrices, with or without replacement.
//!
//! ```matlab
//! mat = [1, 2, 3; 4, 5, 6; 7, 8, 9]
//! row_indices = randperm(size(mat,1), 2)
//! sample_rows = mat(row_indices, :)
//! col_indices = randi(size(mat,2), 2)
//! sample_cols = mat(:, col_indices)
//! ```

extern crate blas_src;
use ndarray::{array, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand::distributions::Distribution;

fn main() {
    // Generate a matrix from which to take random samples.
    println!("3x3 matrix to be sampled:");
    let mat = array![[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]];
    println!("{:?}", mat);

    // Do it the easy way using ndarray_rand::RandomExt::sample_axis.
    // Does not keep the indices used for sampling.
    println!("Sampling 2 rows without replacement:");
    // Axis(0) denotes rows.  We will sample 2 rows.
    let sample_rows = mat.sample_axis(Axis(0), 2, SamplingStrategy::WithoutReplacement);
    println!("{:?}", sample_rows);
    println!("Sampling 2 columns with replacement:");
    // Axis(1) denotes columns.
    let sample_cols = mat.sample_axis(Axis(1), 2, SamplingStrategy::WithReplacement);
    println!("{:?}", sample_cols);

    // Do it manually using modules directly from the rand crate.
    // This allows us to save the random indices for re-use.
    // Note that if you just want to control the random number generator you
    // can use ndarray_rand::RandomExt::sample_axis_with().
    //
    // We need to explicitly specify the random number generator.
    // For this example we'll use the default RNG.
    let mut rng = rand::thread_rng();
    println!("Sampling 2 rows without replacement:");
    // Use rand::seq::index__sample() to sample 2 integers from 0 to to the
    // number of rows, without replacement.
    let row_indices = rand::seq::index::sample(&mut rng, mat.len_of(Axis(0)), 2).into_vec();
    println!("{:?}", row_indices);
    // Sample the rows using the random row indices we generated.
    let sample_rows = mat.select(Axis(0), &row_indices);
    println!("{:?}", sample_rows);
    println!("Sampling 2 columns with replacement:");
    // Sample from a uniform distribution of integers over the desired range.
    // This will sample with replacement.
    let col_indices: Vec<_> = {
        let distribution = rand::distributions::Uniform::from(0..mat.len_of(Axis(1)));
        (0..2).map(|_| distribution.sample(&mut rng)).collect()
    };
    println!("{:?}", col_indices);
    // Sample the columns using the random column indices we generated.
    let sample_cols = mat.select(Axis(1), &col_indices);
    println!("{:?}", sample_cols);
}
