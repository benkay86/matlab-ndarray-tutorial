//! Writing functions that do linear algebra.
//!
//! Writing functions that take a concrete type such as
//! `Array<f64, Dim<[Ix; 1]>>` is easy.  The function can borrow the array `&`,
//! mutably borrow the array `&mut` to modify it in-place, or take ownership of
//! the array just as with any other variable in Rust.
//!
//! In general you want your functions to actually work with `ArrayBase` instead
//! of `Array`.  This will allow your function to take views of an array (e.g.
//! obtained by slicing) instead of just arrays.
//!
//! The primary difficulty is with writing concise, generic functions that can
//! operate on multiple numeric types while satisfying the complicated trait
//! systems of ndarray and ndarray_linalg.  Here we will focus on floating
//! point types.

extern crate blas_src;

// Most concrete and simple example.
// We borrow a 1d array of f64 and add a scalar of type f64.
// We return the sum as a 1d array of f64.
fn array_scalar_add1(
    arr: &ndarray::Array<f64, ndarray::Dim<[ndarray::Ix; 1]>>,
    ref scal: f64
) -> ndarray::Array<f64, ndarray::Dim<[ndarray::Ix; 1]>>
{
    arr + *scal
}

// More generic example, still using f64 as the underlying numeric type.
//
// We borrow an array of f64.  Instead of `Array` we use `ArrayBase`.  This
// lets us work with slices and views of arrays, not just owned arrays.
// However, the type `Data` is still constrained to use `f64` for the scalar
// elements of whatever kind of array it is.
//
// Note that the returned value is still an owned array.  That is because we
// create the returned array on the function stack and must then pass ownership
// of it to the caller.
//
// We are also generic over the dimensionality of the array.  So now we can take
// 1d, 2d, 3d, ..., Nd arrays.  The returned array will have the same
// dimensionality `Dim` as the input array.
//
// Note that the generic parameters `Data` and `Dim` can have any name (we could
// have done `MyData` and `MyDim` instead).
fn array_scalar_add2<Data, Dim>(
    arr: &ndarray::ArrayBase<Data, Dim>,
    ref scal: f64
) -> ndarray::Array<f64, Dim>
where
    Data: ndarray::Data<Elem = f64>,
    Dim: ndarray::Dimension,
{
    arr + *scal
}

// Most generic example.
// We use `ArrayBase` to be generic over owned arrays and views.
// We use the `Dimensions` trait to be generic over dimensionality.
// In addition, we are now generic over the scalar element type.
//
// Instead of specifying the concrete type `f64`, we constrain the type using
// trait bounds requiring that the type in the array `N1`:
//
// - Can participate in scalar operations
// - Can have a value of type `N2` added to it such that the sum is of type `N1`
//
// And we require the scalar argument type `N2`:
//
// - Can participate in scalar operations
// - Is trivially copyable
//
// In practice this means our function accepts integers and floats.
fn array_scalar_add3<Data, Dim, N1, N2>(
    arr: &ndarray::ArrayBase<Data, Dim>,
    ref scal: N2
) -> ndarray::Array<N1, Dim>
where
    Data: ndarray::Data<Elem = N1>,
    Dim: ndarray::Dimension,
    N1: ndarray::ScalarOperand + std::ops::Add<N2, Output = N1>,
    N2: ndarray::ScalarOperand + Copy,
{
    arr + *scal
}

// As we can see above, working with generic trait bounds on numeric types can
// get quite verbose.  What if we need to multiply as well as adding?  What if
// we need the type to be compatible with an LAPACK solver?  The following trait
// alias effectively constrains us to the floating point types `f32` and `f64`.
//
// If you are writing a low-level function that does something very simple like
// our `array_scalar_add()` example then you should ideally use the least
// restrictive trait bounds such as in `array_scalar_add3()`, however you can
// use a comprehensive trait alias like `Float` for convenience.  You can always
// go back and relax the trait bounds later while maintaining backwards-
// compatibility of your API.
//
// If you are writing a high-level function that might do many different linear
// algebra operations then using a `Float` is not only convenient, but also
// probably a good idea.  You will be less likely to need to make breaking
// changes to your API in the future if you find your complicated math ends up
// needing another trait bound.
trait Float:
    // Needed for compute-by-value semantics.
    Copy +
    // Needed for array-scalar operations, like adding a scalar to an array.
    ndarray::ScalarOperand +
    // Collection of `std::ops::Add`, `std::ops::Mul`, etc.
    // Only need to uncomment if not using `ndarray_linalg::Scalar`.
    // ndarray::LinalgScalar +
    // Needed for comparison like `<` and `==`.
    std::cmp::PartialOrd +
    // Needed for basic math in `ndarray_linalg`, which does not respect
    // `ndarray::LinalgScalar`.
    ndarray_linalg::Scalar +
    // Needed for any call into an LAPACK function, e.g. solvers.
    ndarray_linalg::Lapack
{ }
impl<F> Float for F where F:
    Copy +
    ndarray::ScalarOperand +
    // ndarray::LinalgScalar +
    std::cmp::PartialOrd +
    ndarray_linalg::Scalar<Real=F> + // get rid of Real=F to allow complex #'s
    ndarray_linalg::Lapack
{ }

// Similar to `array_scalar_add3()` but uses the `Float` trait alias above to
// cut down on typing.  The tradeoff is that now we can only operate on floating
// point types -- no integers allowed.
fn array_scalar_add4<Data, Dim, F>(
    arr: &ndarray::ArrayBase<Data, Dim>,
    ref scal: F
) -> ndarray::Array<F, Dim>
where
    Data: ndarray::Data<Elem = F>,
    Dim: ndarray::Dimension,
    F: Float,
{
    arr + *scal
}

// So far we haven't used any literal numbers (e.g. `2.`) in our function.
// Let's rewrite `array_scalar_add2()` to always add the number 2 to whatever
// array is passed in.
fn array_scalar_add5<Data, Dim>(arr: &ndarray::ArrayBase<Data, Dim>)
-> ndarray::Array<f64, Dim>
where
    Data: ndarray::Data<Elem = f64>,
    Dim: ndarray::Dimension,
{
    arr + 2.
}

/// Generic function that adds the number 2 to an array.
/// Use `Float` trait alias and `f()` cast defined above for convenience.
fn array_scalar_add6<Data, Dim, F>(arr: &ndarray::ArrayBase<Data, Dim>)
    -> ndarray::Array<F, Dim>
where
    Data: ndarray::Data<Elem = F>,
    Dim: ndarray::Dimension,
    F: Float,
{
    arr + F::from_real(F::real(2))
}

fn main() {
    // Make an array and a scalar to add togehter.
    let arr = ndarray::array![1., 2., 3.];
    let scal = 2.;

    // Test out each of our functions above.
    println!("arr: {:?}\nscal: {:?}\n", arr, scal);
    println!("array_scalar_add1(arr, scal) =\n{:?}", array_scalar_add1(&arr, scal));
    println!("array_scalar_add2(arr, scal) =\n{:?}", array_scalar_add2(&arr, scal));
    println!("array_scalar_add3(arr, scal) =\n{:?}", array_scalar_add3(&arr, scal));
    println!("array_scalar_add4(arr, scal) =\n{:?}", array_scalar_add4(&arr, scal));
    println!("array_scalar_add5(arr, scal) =\n{:?}", array_scalar_add5(&arr));
    println!("array_scalar_add6(arr, scal) =\n{:?}", array_scalar_add6(&arr));
}
