//! The basics of initializing matrices.
//!
//! To create a matrix without initializing it, see <uninitialized.rs>.

// Tell the compiler this binary requires the link dependencies from
// openblas-src (i.e. to link to blas and lapack backends).
extern crate blas_src;

// Bring needed traits and macros into scope.
use ndarray::{array, Array, Dim, Ix};

fn main() {
    // Use ndarray's `array!` macro to initialize a matrix with literal values.
    // Note the use of the `1.`, which is short for `1.0`, instead of `1`.  This
    // tells Rust that we want a floating point literal rather than an integer
    // literal.  In Matlab:
    //
    // ```matlab
    // mat = [1,3,4; 4,5,6]
    // ```
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("Double-precision 2x3 array:\n{:?}", mat);
    // Rust defaults to double-precision (f64) float.  We can use
    // single-precision (i.e. 32-bit) by specifying the type of the array.  The
    // first type parameter, `f32`, specifies the type of data stored in the
    // array.  The second parameter, `Dim<[Ix; 2]>` specifies the number of
    // dimensions/axes.  If we wanted a 3d array it would be `Dim<[Ix; 3]>`.
    //
    // Note there is currently no way (at least right now) to specify the shape
    // of the array (2x3 in this case) at compile time.
    let mat: Array<f32, Dim<[Ix; 2]>> = array![[1., 2., 3.], [4., 5., 6.]];
    println!("Single-precision 2x3 array:\n{:?}\n", mat);

    // In Rust, as in Matlab, we can fill an array with zeros.  Note the `_` to
    // save keystrokes by letting the compiler infer the dimensionality of the
    // array.
    //
    // Note the peculiar ::<> notation is known as the
    // [turbofish](https://matematikaadit.github.io/posts/rust-turbofish.html).
    //
    // ```matlab
    // mat = zeros(2,3)
    // ```
    let mat = Array::<f64, _>::zeros((2, 3));
    println!("2x3 array of zeros:\n{:?}", mat);
    // We can also fill an array with ones.
    //
    // ```matlab
    // mat = ones(2,3)
    // ```
    let mat = Array::<f64, _>::ones((2, 3));
    println!("2x3 array of ones:\n{:?}", mat);
    // Or whatever value we want.
    //
    // ```matlab
    // mat = zeros(2,3) + 42
    // ```
    let mat = Array::from_elem((2, 3), 42.);
    println!(
        "2x3 array of the life, universe, and everything:\n{:?}",
        mat
    );
    // We can even fill the array by visiting each element with a function or
    // closure.
    let mat = Array::from_shape_fn((2, 3), |(x, y)| 3. * x as f64 + y as f64);
    println!("2x3 array of offsets:\n{:?}\n", mat);

    // Create a one-dimensional array over a range.  Note that rust ranges
    // (unlike Matlab) do *not* include the last element.
    //
    // ```matlab
    // mat = 0:3:9
    // ```
    let mat = Array::range(0., 12., 3.);
    println!("Zero through 9 in increments of 3:\n{:?}", mat);
    // But ndarray's `linspace()` *does* include the last element.
    //
    // ```matlab
    // mat = linspace(0,9,4)
    // ```
    let mat = Array::linspace(0., 9., 4);
    println!("Zero through 9 in increments of 3:\n{:?}\n", mat);
    // See also `ArrayBase::geomspace()` and `logspace()`.`

    // It's convenient to be able to generate an identity matrix.
    //
    // ```matlab
    // mat = eye(2)
    // ```
    let mat = Array::<f64, _>::eye(2);
    println!("2x2 identity matrix:\n{:?}", mat);
    // Sometimes you also want to make a diagonal matrix from a vector.
    //
    // ```matlab
    // mat = diag([1,2])
    // ```
    let mat = Array::from_diag(&array![1., 2.]);
    println!("2x2 diagonal matrix matrix:\n{:?}", mat);
}
