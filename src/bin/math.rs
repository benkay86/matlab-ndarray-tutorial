//! Matrix math with ndarray.

extern crate blas_src;
use core::ops::Mul;
use core::ops::Add;
use ndarray::{array, Axis};

fn main() {
    // Generate a matrix and a vector to work with.
    //
    // ```matlab
    // mat_a = [1,2,3; 4,5,6; 7,8,9]
    // vec_b = [1, 2, 3]
    // ```
    let mat_a = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let vec_b = array![1., 2., 3.];
    println!("Matrix a:\n{:?}", mat_a);
    println!("Vector b:\n{:?}\n", vec_b);

    // Matrix multiplication is done with the `dot()` method.
    //
    // ```matlab
    // mat_a * mat_a
    // ```
    println!("a.dot(&a) =\n{:?}", mat_a.dot(&mat_a));
    // In the case of vectors you get the dot product.
    //
    // ```matlab
    // dot(vec_b, vec_b)
    // ```
    println!("b.dot(&b) =\n{:?}", vec_b.dot(&vec_b));
    // For a vector.dot(matrix) operation the 1d vector is automatically treated
    // as a row vector.
    //
    // ```matlab
    // vec_b * mat_a
    // ```
    println!("b.dot(&a) = \n{:?}", vec_b.dot(&mat_a));
    // For a matrix.dot(vector) operation the 1d vector is automatically treated
    // as a column vector.  There is no need to transpose it.  However, note
    // that the result is a 1d array (not a column vector).
    //
    // ```matlab
    // mat_a * vec_b'
    // ```
    println!("a.dot(&b) = \n{:?}", mat_a.dot(&vec_b));
    // If we want to do a column x row vector matrix multiplication then we need
    // to actually make b into a 2d matrix.
    //
    // ```matlab
    // vec_b' * vec_b
    // ```
    let mat_b = vec_b.clone().insert_axis(Axis(0));
    println!("2d row vector b:\n{:?}", mat_b);
    println!("b.t().dot(&b)\n{:?}\n", mat_b.t().dot(&mat_b));

    // Element-wise multiplication is indicated with the * operator.
    // The same rules apply for element-wise addition and subtraction.
    //
    // ```matlab
    // a .* a
    // a + a
    // ```
    println!("&a * &a =\n{:?}", &mat_a * &mat_a);
    println!("&a + &a =\n{:?}", &mat_a + &mat_a);
    // If desired, we can also use `mul()`, `add()`, etc syntax.  The
    // appropriate `core::ops::Mul` or other trait must be in scope for this to
    // work (see `use` statement above).
    //
    // ```matlab
    // b .* b
    // b + b
    // ```
    println!("(&b).mul(&b) =\n{:?}", (&vec_b).mul(&vec_b));
    println!("(&b).add(&b) =\n{:?}", (&vec_b).add(&vec_b));
    // Note the use of immutable borrows `&`.  Without these, element-wise
    // operations consume the operands.  Sometimes this is desired for memory
    // efficiency, but in this example we want to reuse mat_a and vec_b later.
    // Uncomment the following line to get a compiler error.
    // let _product = mat_a * vec_b; // mat_a and vec_b consumed, can't be used later
    //
    // With vector x matrix or matrix x vector element-wise multiplication, the
    // vector is treated as a row vector and automatically broadcast to the size
    // of the matrix by default.
    //
    // ```matlab
    // a .* b
    // ```
    println!("&a * &b =\n{:?}", &mat_a * &vec_b);
    // If we want to broadcast as a column vector instead we must do so
    // explicitly, or else use the 2d matrix form of b and transpose it.
    // Broadcasting is usually more memory efficient.
    //
    // ```matlab
    // a .* b'
    // ```
    println!("&a * &vec_b.broadcast().t() =\n{:?}", &mat_a * &vec_b.broadcast((3,3)).unwrap().t());
    println!("&a * &mat_b.t() =\n{:?}\n", &mat_a * &mat_b.t());

    // We can also do element-wise addition, multiplication, etc.
    //
    // ```matlab
    // a + 2
    // b * 2
    // ```
    println!("&a + 2 =\n{:?}", &mat_a + 2.); // or: (&mat_a).add(2.)
    println!("&b * 2 =\n{:?}", &vec_b * 2.); // or: (&vec_b).mul(2.)
    // We must "map" more complex element-wise operations onto the array.
    // The map method takes a closure or function pointer.
    // `map()` visits the elements of the array by reference and `mapv()` by
    // value.  `mapv()` is more efficient for trivially-copyable elements.
    //
    // ```matlab
    // abs(mat_a)
    // sin(vec_b)
    // ```
    println!("abs(a) = \n{:?}", mat_a.mapv(f32::abs)); // function pointer
    println!("sin(b) = \n{:?}", vec_b.mapv(|b| f32::sin(b))); // closure
    // Unlike Matlab, in Rust we have the option of mapping in-place or
    // mapping into a new matrix without creating a copy.
    //
    // ```
    // a = sin(abs(a))
    // ```
    let mut mat_a = mat_a.mapv_into(f32::abs);
    mat_a.mapv_inplace(f32::sin);
    println!("sin(abs(a)) = \n{:?}\n", mat_a);

    // We can also do some simple reductions.
    // More in this in <reduction.rs>.
    // First let's sum *all* the elements of a matrix.
    //
    // matlab```
    // sum(sum(a))
    // ```
    let mat_a = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    println!("a.sum() = {:?}", mat_a.sum());
    // Or we can sum along an axis.
    //
    // matlab```
    // sum(a) % sum columns
    // sum(a,2) % sum rows
    // ```
    println!("a.sum_axis(Axis(0)) =\n{:?}", mat_a.sum_axis(Axis(0)));
    println!("a.sum_axis(Axis(1)) =\n{:?}", mat_a.sum_axis(Axis(1)));
    // We can also do mean and standard deviation.  For standard deviation we
    // must specify the degrees of freedom, typically 0 for a population
    // standard deviation and 1 for a sample standard deviation.
    //
    // matlab```
    // std(a) % standard deviation of columns
    // mean(b) % mean all elements in vector
    // ```
    println!("a.std_axis(Axis(0)) = \n{:?}", mat_a.std_axis(Axis(0), 1.));
    println!("b.mean() = {:?}", vec_b.mean());
}
